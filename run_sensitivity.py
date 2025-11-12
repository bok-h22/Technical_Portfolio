import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Set, List

from cornac.metrics import NDCG, Recall, Precision

from src.utils import set_seed
from src.data_loader import load_data
from src.baseline_models import run_model as run_baseline_model
from src.user_profiling import compute_long_tail_items, compute_item_pops
from src.reranker import Reranker
from src.experiment import run_experiment

logging.getLogger('tensorflow').setLevel(logging.ERROR)

def harmonic_mean(x, y):
    if x + y == 0:
        return 0.0
    return (2 * x * y) / (x + y)

def run_lambda_sweep(config: Dict, eval_method: Any, item_info: Dict):
    logging.info("--- 1. 람다 민감도 분석 시작 ---")
    
    train_set = eval_method.train_set
    test_set = eval_method.test_set
    test_user_indices = set(test_set.uir_tuple[0])
    metrics = [NDCG(k=10), Recall(k=10), Precision(k=10)]
    
    logging.info("Reranker와 사용자 프로필 초기화")
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
    
    import pandas as pd
    import numpy as np

    if 'lambda_sweep_values' in config['reranking']:
        lambda_values = np.array(config['reranking']['lambda_sweep_values'])
        logging.info(f"설정 파일의 람다 스윕 값: {lambda_values}")
    else:
        lambda_values = np.arange(0.1, 1.0, 0.1).round(2)
        logging.warning(f"설정 파일에 'lambda_sweep_values'가 없어 기본값 사용: {lambda_values}")
    
    rerank_modes_to_sweep = [m for m in config['reranking']['modes'] if m != "None"]
    pers_modes_to_sweep = config['reranking']['personalize']
    
    all_run_results = []

    for model in loaded_models:
        model_name = getattr(model, "name", str(model))
        logging.info(f"--- 베이스라인 (None) 실행 시작: {model_name} ---")
        
        avg_results_base, _, _ = run_experiment(
            model=model, test_user_indices=test_user_indices, 
            train_set=train_set, test_set=test_set,
            item_info=item_info, metrics=metrics, reranker=reranker,
            rerank_options={
                **config['reranking']['params'],
                "diversification_mode": "None",
                "personalize_mode": "NP",
                "final_k": config['reranking']['final_k'],
                "topk_candidates": config['reranking']['top_k_candidates']
            }
        )
        run_info_base = {
            "model": model_name, "diversification": "None", 
            "personalize": "NP", "lambda_param": np.nan
        }
        run_info_base.update(avg_results_base)
        cc_base = avg_results_base.get("CatalogCoverage", 0.0)
        aclt_base = avg_results_base.get("ACLT", 0.0)
        run_info_base["CompositeDiv"] = harmonic_mean(cc_base, aclt_base)
        all_run_results.append(run_info_base)

    for model in loaded_models:
        model_name = getattr(model, "name", str(model))
        for per_mode in pers_modes_to_sweep:
            for div_mode in rerank_modes_to_sweep:
                for lamb in lambda_values:
                    
                    logging.info(f"--- 스윕 실행 시작: {model_name} | {div_mode} | {per_mode} | λ={lamb:.1f} ---")
                    
                    reranker.lambda_rel = lamb
                    current_rerank_params = config['reranking']['params'].copy()
                    current_rerank_params['lambda_rel'] = lamb
                    
                    avg_results_rerank, _, _ = run_experiment(
                        model=model, test_user_indices=test_user_indices,
                        train_set=train_set, test_set=test_set,
                        item_info=item_info, metrics=metrics, reranker=reranker,
                        rerank_options={
                            **current_rerank_params, 
                            "diversification_mode": div_mode,
                            "personalize_mode": per_mode,
                            "final_k": config['reranking']['final_k'],
                            "topk_candidates": config['reranking']['top_k_candidates']
                        }
                    )
                    
                    run_info_rerank = {
                        "model": model_name, "diversification": div_mode,
                        "personalize": per_mode, "lambda_param": lamb
                    }
                    run_info_rerank.update(avg_results_rerank)
                    cc_rerank = avg_results_rerank.get("CatalogCoverage", 0.0)
                    aclt_rerank = avg_results_rerank.get("ACLT", 0.0)
                    run_info_rerank["CompositeDiv"] = harmonic_mean(cc_rerank, aclt_rerank)
                    all_run_results.append(run_info_rerank)

    results_df = pd.DataFrame(all_run_results)
    
    save_dir_path = Path(config['results_dir']) / config['dataset_name']
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    today_date = pd.Timestamp.now().strftime("%Y%m%d")
    output_path = save_dir_path / f"{today_date}_{config['dataset_name']}_parameter_sweep_results.csv"
    
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    logging.info(f"\n✅ 람다 스윕 결과 저장 완료: {output_path}")
    logging.info("\n--- [ 전체 실험 결과 요약 ] ---")
    logging.info(results_df.round(4).to_string())

def main():
    parser = argparse.ArgumentParser(description="Lambda 파라미터 민감도 분석 실행")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/movielens_1m.yaml", 
        help="설정 YAML 파일 경로."
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

    run_lambda_sweep(config, eval_method, item_info)

    logging.info("--- 민감도 분석 완료 ---")

if __name__ == "__main__":
    main()