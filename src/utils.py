import os
import random
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import tensorflow
import torch

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "experiment.log", mode='w'),
        logging.StreamHandler()
    ]
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tensorflow.random.set_seed(seed)
    logging.info(f"랜덤 시드 {seed} 설정 완료")

def compute_normalize_scores(
    scores: np.ndarray, 
    sorted: bool = False,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    if scores.size == 0:
        return scores
        
    min_s, max_s = np.nanmin(scores), np.nanmax(scores)

    if max_s == min_s:
        scaled_scores = np.full_like(scores, (min_val + max_val) / 2)
    else:
        scaled_0_to_1 = (scores - min_s) / (max_s - min_s)
        scaled_scores = scaled_0_to_1 * (max_val - min_val) + min_val
    if sorted:
        return np.sort(scaled_scores)[::-1]
    
    return scaled_scores

def compute_pos_items(csr_row: Any, rating_threshold: float = 1.0) -> List[int]:
    if not hasattr(csr_row, 'indices') or not hasattr(csr_row, 'data'):
        logging.warning("유효하지 않은 CSR 행 형식")
        return []
        
    return [
        item_idx for item_idx, rating in zip(csr_row.indices, csr_row.data) 
        if rating >= rating_threshold
    ]

def save_results_by_model(
    model_name: str, 
    result_dict: Dict, 
    dataset: str, 
    diversification: str, 
    personalize: str, 
    save_dir: str
):
    save_dir_path = Path(save_dir) / dataset
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    pers_str = f"_{personalize}" if diversification != "None" else ""
    file_key = f"{diversification}{pers_str}"
    
    for key, result in result_dict.items():
        path = save_dir_path / f"{model_name}_result_{file_key}_{key}.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logging.error(f"피클 파일 저장 실패: {path}, 오류: {e}")
            
    logging.info(f"결과 저장 완료: 모델 {model_name} (모드: {file_key}), 경로 {save_dir_path}")