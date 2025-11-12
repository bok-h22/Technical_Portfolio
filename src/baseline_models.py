import os
import json
import logging
from typing import List, Tuple, Any, Dict

from cornac.eval_methods import BaseMethod
from cornac.models import WMF, EASE, VAECF, NeuMF
from cornac.metrics import NDCG, Recall, Precision, MRR
from cornac.experiment import Experiment
from cornac.hyperopt import GridSearch, Discrete

import tensorflow as tf
import torch

os.environ['TF_MLIR_ENABLE_V1_PASSES'] = '1'

def get_gridsearch_space(model_name: str) -> List[Discrete]:
    if model_name == "WMF":
        return [
            Discrete(name="k", values=[50, 100, 200]),
            Discrete(name="learning_rate", values=[0.001, 0.005]),
            Discrete(name="lambda_u", values=[0.01, 0.1, 0.5]),
            Discrete(name="lambda_v", values=[0.01, 0.1])
        ]
    if model_name == "EASE":
        return [Discrete(name="lamb", values=[500, 1000, 2500, 5000])]
    if model_name == "VAECF":
        return [
            Discrete(name="k", values=[20, 50, 100]),
            Discrete(name="learning_rate", values=[0.001, 0.005, 0.01])
        ]
    if model_name == "NeuMF":
        return [
            Discrete(name="num_factors", values=[8, 16, 32]),
            Discrete(name="layers", values=[[32, 16, 8], [64, 32, 16, 8]]),
            Discrete(name="lr", values=[0.001, 0.005]),
            Discrete(name="num_epochs", values=[10, 20])
        ]
    return []

def create_model_instance(model_name: str, params: Dict, seed: int, verbose: bool) -> Any:
    base_params = {"seed": seed, "verbose": verbose}

    if model_name == "WMF":
        default_params = {"k": 200, "max_iter": 100, "learning_rate": 0.001, 
                          "lambda_u": 0.01, "lambda_v": 0.01}
        final_params = {**default_params, **params, **base_params}
        return WMF(**final_params)

    if model_name == "EASE":
        default_params = {"lamb": 500}
        final_params = {**default_params, **params, **base_params}
        return EASE(**final_params)

    if model_name == "VAECF":
        default_params = {"autoencoder_structure": [20], "act_fn": "tanh", 
                          "likelihood": "mult", "n_epochs": 100, 
                          "batch_size": 100, "beta": 1.0}
        final_params = {**default_params, **params, **base_params}
        return VAECF(**final_params)

    if model_name == "NeuMF":
        default_params = {"num_factors": 8, "layers": [64, 32, 16, 8], "act_fn": "relu", 
                          "num_epochs": 10, "num_neg": 4, "batch_size": 256, "lr": 0.001}
        final_params = {**default_params, **params, **base_params}
        return NeuMF(**final_params)
        
    raise ValueError(f"알 수 없는 모델 이름: {model_name}")

def run_model(
    eval_method: BaseMethod, 
    model_names: List[str],
    dataset_name: str,
    save_dir: str,
    seed: int,
    grid_search: bool = False,
    model_params_map: Dict = None,
    verbose: bool = False
) -> Tuple[List[Any], Experiment]: 
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    models = []
    
    for model_name in model_names:
        cornac_model_name = "EASE" if model_name == "EASEᴿ" else model_name
        if grid_search:
            logging.info(f"[{cornac_model_name}] GridSearch 설정")
            
            base_model_class = globals().get(cornac_model_name)
            if not base_model_class:
                logging.warning(f"모델 클래스 {cornac_model_name}를 찾을 수 없음. 건너뛰기")
                continue
            base_model = base_model_class(seed=seed, verbose=verbose)
            
            space = get_gridsearch_space(cornac_model_name)
            if not space:
                logging.warning(f"{cornac_model_name}에 대한 GridSearch 공간이 정의되지 않음. 건너뛰기")
                continue

            gs_model = GridSearch(
                model=base_model, 
                space=space, 
                metric=NDCG(k=10), 
                eval_method=eval_method
            )
            models.append(gs_model)
        
        else:
            logging.info(f"[{model_name}] 파라미터 로드 진행")
            
            params = {}
            if model_params_map and model_name in model_params_map:
                params = model_params_map[model_name]
                logging.info(f"[{model_name}] 설정 파일에서 파라미터 로드 완료: {params}")
            else:
                load_path = os.path.join(dataset_save_dir, f"{model_name}_best_params.txt")
                try:
                    with open(load_path, 'r') as f:
                        params = json.load(f)
                    logging.info(f"[{model_name}] 파일 '{load_path}'에서 파라미터 로드 완료: {params}")
                except FileNotFoundError:
                    logging.warning(f"[{model_name}] 파라미터 파일 '{load_path}'이 없고 설정값도 없음. 모델 기본값 사용")
            
            try:
                model_instance = create_model_instance(
                    cornac_model_name, params, seed, verbose
                )
                model_instance.name = model_name 
                models.append(model_instance)
            except Exception as e:
                logging.error(f"{model_name} 인스턴스 생성 실패: {e}")

    if not models:
        logging.warning("실행할 모델이 설정되거나 로드되지 않음")
        return [], None

    metrics = [NDCG(k=10), Recall(k=10), Precision(k=10), MRR()]
    
    exp = Experiment(
        eval_method=eval_method,
        models=models,
        metrics=metrics,
        verbose=verbose
    )
    
    if grid_search:
        logging.info(f"--- GridSearch 실행 시작: {model_names} ---")
    else:
        logging.info(f"--- 실험 실행 시작 (최적 파라미터): {model_names} ---")
        
    exp.run()
    
    if grid_search:
        logging.info("--- GridSearch 완료. 최적 파라미터 저장 ---")
        
        for model_obj in models:
            if isinstance(model_obj, GridSearch):
                model_name = model_obj.model.name 
                save_model_name = "EASEᴿ" if model_name == "EASE" else model_name

                best_params = model_obj.best_params
                save_path = os.path.join(dataset_save_dir, f"{save_model_name}_best_params.txt")
                
                try:
                    with open(save_path, 'w') as f:
                        json.dump(best_params, f, indent=4)
                    logging.info(f"[{save_model_name}] 최적 파라미터 저장 완료: {save_path}")
                    logging.info(f"  > 파라미터: {best_params}")
                    logging.info(f"  > 최고 {model_obj.metric.name} 점수: {model_obj.best_score:.4f}")
                except Exception as e:
                    logging.error(f"{save_model_name} 파라미터 저장 실패: {e}")
    else:
        logging.info("--- 실험 완료 ---")

    return models, exp