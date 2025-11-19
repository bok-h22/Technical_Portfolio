import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple

from cornac.metrics import NDCG, Recall, Precision

from src.utils import set_seed, save_results_by_model
from src.data_loader import load_data
from src.baseline_models import run_model as run_baseline_model
from src.user_profiling import compute_long_tail_items, compute_item_pops
from src.reranker import Reranker 
from src.experiment import run_experiment

TODAY_DATE = "20251118"

def train_baselines(config: Dict, eval_method: Any):
    logging.info("--- 1. 베이스라인 모델 학습 시작 (GridSearch) ---")
    
    run_baseline_model(
        eval_method=eval_method,
        model_names=config['models'],
        dataset_name=config['dataset_name'],
        save_dir=config['results_dir'],
        seed=config['seed'],
        grid_search=True,  
        model_params_map=None, 
        verbose=True
    )
    logging.info("--- 베이스라인 학습 완료 (GridSearch) ---")

def find_optimal_lambda(
    results_df: pd.DataFrame, 
    beta: float = 0.5, 
    ndcg_metric: str = 'NDCG@10', 
    div_metric: str = 'CompositeDiv'
) -> Dict[str, Dict[str, float]]:
    max_acc = results_df[ndcg_metric].max()
    max_div = results_df[div_metric].max()
    
    if max_acc == 0 or max_div == 0:
        print("경고: NDCG 또는 CompositeDiv의 최대값이 0임")
        return {}

    beta_sq = beta ** 2
    df_calc = results_df.copy()
    
    df_calc['Acc_Norm'] = df_calc[ndcg_metric] / max_acc
    df_calc['Div_Norm'] = df_calc[div_metric] / max_div

    numerator = (1 + beta_sq) * df_calc['Acc_Norm'] * df_calc['Div_Norm']
    denominator = (beta_sq * df_calc['Acc_Norm']) + df_calc['Div_Norm']
    
    df_calc['F_beta_Score'] = np.divide(
        numerator, denominator, 
        out=np.zeros_like(numerator), 
        where=denominator != 0
    )
    
    optimal_params = defaultdict(dict)
    df_optimal = df_calc[df_calc['diversification'] != 'None']
    
    grouped = df_optimal.groupby(['model', 'diversification', 'personalize'])
    
    for names, group in grouped:
        model, div_mode, per_mode = names
        max_score_row = group.loc[group['F_beta_Score'].idxmax()]
        mode_key = f"{div_mode}_{per_mode}"
        
        optimal_params[model][mode_key] = max_score_row['lambda_param']
        
    return dict(optimal_params)

def run_reranking(config: Dict, 
                  eval_method: Any, 
                  item_info: Dict,
                  optimal_lambda_map: Dict[str, Dict[str, float]] = None):
    logging.info("--- 2. 리랭킹 실험 시작 ---")
    
    train_set = eval_method.train_set
    test_set = eval_method.test_set
    test_user_indices = set(test_set.uir_tuple[0]) 
    metrics = [NDCG(k=10), Recall(k=10), Precision(k=10)]

    logging.info("리랭커 데이터 준비 (아이템 인기도 계산)")
    item_pops = compute_item_pops(train_set)
    long_tail_items = compute_long_tail_items(
        item_pops, 
        head_percentage=config['analysis']['long_tail_head_percentage']
    )
    reranker = Reranker(
        rerank_config={**config['reranking']['params'], 
                       "final_k": config['reranking']['final_k']},
        user_profile_config=config['analysis'],
        train_set=train_set,
        item_info=item_info,
        long_tail_items=long_tail_items
    )

    loaded_models, _ = run_baseline_model(
        eval_method=eval_method,
        model_names=config['models'],
        dataset_name=config['dataset_name'],
        save_dir=config['results_dir'],
        seed=config['seed'],
        grid_search=False, 
        model_params_map=config['model_params'], 
        verbose=True
    )
    
    if not loaded_models:
        logging.error("로드된 베이스라인 모델이 없음")
        return

    logging.info("--- 메인 실험 루프 시작 ---")
    
    rerank_modes = config['reranking']['modes']
    pers_modes = config['reranking']['personalize']
    
    for model in loaded_models:
        model_name = getattr(model, "name", str(model))
        
        for per_mode in pers_modes:
            for div_mode in rerank_modes:
                mode_key = f"{div_mode}_{per_mode}"
                lambda_to_use = config['reranking']['params']['lambda_rel']                 
                if optimal_lambda_map and model_name in optimal_lambda_map and mode_key in optimal_lambda_map[model_name]:
                    lambda_to_use = optimal_lambda_map[model_name][mode_key]
                    logging.info(f"-> 최적 람다 적용: {model_name} | {mode_key} -> λ={lambda_to_use:.2f}")
                
                if div_mode == "None" and per_mode != pers_modes[0]:
                    continue
                    
                current_pers_mode = per_mode if div_mode != "None" else "NP"
                reranker.lambda_rel = lambda_to_use
                
                logging.info(f"실험 진행: 모델 {model_name}, 모드 {div_mode}, 개인화 {current_pers_mode}")
                
                avg_results, user_results, recommendations = run_experiment(
                    model=model, 
                    test_user_indices=test_user_indices, 
                    train_set=train_set, 
                    test_set=test_set,
                    item_info=item_info, 
                    metrics=metrics, 
                    reranker=reranker, 
                    rerank_options={
                        **config['reranking']['params'], 
                        "diversification_mode": div_mode,
                        "personalize_mode": current_pers_mode,
                        "lambda_rel": lambda_to_use,
                        "final_k": config['reranking']['final_k'],
                        "topk_candidates": config['reranking']['top_k_candidates']
                    }
                )
                
                logging.info(f"--- [{model_name}] 결과 ({div_mode}, {current_pers_mode}) ---")
                log_res = {k: f"{v:.4f}" for k, v in avg_results.items() if "CatalogCoverage" not in k}
                logging.info(f"평균 결과: {log_res}")
                
                save_results_by_model(
                    model_name=model_name, 
                    result_dict={"avg": avg_results, "user": user_results, "recs": recommendations},
                    dataset=config['dataset_name'],
                    diversification=div_mode, 
                    personalize=current_pers_mode, 
                    save_dir=config['results_dir']
                )

    logging.info("--- 모든 리랭킹 실험 완료 ---")

def main():
    parser = argparse.ArgumentParser(description="리랭킹 실험 실행기")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/movielens_1m.yaml", 
        help="설정 YAML 파일 경로."
    )
    parser.add_argument(
        "step", 
        choices=["train", "rerank", "all"]
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"설정 파일을 찾을 수 없음: {config_path}")
        return
        
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"YAML 설정 파일 구문 분석 오류 발생: {e}")
            return
            
    logging.info(f"설정 파일 로드 완료: {config_path}")

    set_seed(config['seed'])

    logging.info("데이터 로드 및 분할 진행...")
    try:
        _, eval_method, item_info = load_data(
            dataset_name=config['dataset_name'],
            path=config['data_path'],
            split_config=config['data_split'],
            seed=config['seed']
        )
    except FileNotFoundError as e:
        logging.error(f"데이터 로드 실패: {e}")
        return
    except Exception as e:
        logging.error(f"데이터 로드 중 예상치 못한 오류 발생: {e}")
        return
    
    results_dir = Path(config['results_dir']) / config['dataset_name']
    sweep_file = results_dir / f"{TODAY_DATE}_{config['dataset_name']}_parameter_sweep_results.csv" 
    
    if sweep_file.exists():
        sweep_df = pd.read_csv(sweep_file)
        optimal_lambda_map = find_optimal_lambda(sweep_df, beta=0.5) 
    else:
        logging.warning("람다 스윕 결과 파일이 없어 최적화된 람다를 찾을 수 없음. 기본값 사용.")
        optimal_lambda_map = None

    if args.step == "train" or args.step == "all":
        train_baselines(config, eval_method)
        
    if args.step == "rerank" or args.step == "all":
        run_reranking(config, eval_method, item_info, optimal_lambda_map)

    logging.info("--- 프로세스 완료 ---")

if __name__ == "__main__":
    main()