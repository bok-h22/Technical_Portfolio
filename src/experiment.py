import numpy as np
import logging
from tqdm import tqdm
from typing import List, Dict, Set, Any, Tuple

from cornac.metrics.ranking import RankingMetric 

from .metrics import rr_at_k, novelty, catalog_coverage, aplt, aclt
from .user_profiling import compute_long_tail_items
from .utils import compute_pos_items, compute_normalize_scores

from .reranker import Reranker 

def run_experiment(
    model: Any, 
    test_user_indices: Set[int], 
    train_set: Any, 
    test_set: Any, 
    item_info: Dict[int, List[str]],
    metrics: List[RankingMetric],
    reranker: Reranker, 
    rerank_options: Dict
) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], Dict[int, Dict[int, float]]]:
    div_mode = rerank_options.get("diversification_mode", "None")
    pers_mode = rerank_options.get("personalize_mode", "NP")
    final_k = rerank_options.get("final_k", 10)
    top_k_candidates = rerank_options.get("topk_candidates", 1000)
    rerank_flag = (div_mode != "None")
    
    model_name = getattr(model, "name", str(model))
    
    log_msg = f"실험 시작: 모델 {model_name}, 모드 {div_mode}, 개인화 {pers_mode}"
    logging.info(log_msg)

    item_pops = reranker.item_pops
    long_tail_items = reranker.long_tail_items
    
    metric_names = [m.name for m in metrics] + ["MRR@10", "Novelty", "CatalogCoverage", "APLT"]
    user_results = {name: {} for name in metric_names}
    recommendation_items = {}
        
    for user_idx in tqdm(test_user_indices, 
                         desc=f"평가 진행 중: {model_name} ({div_mode}/{pers_mode})", 
                         unit="user"):
        
        train_pos_items = compute_pos_items(train_set.csr_matrix.getrow(user_idx))
        test_pos_items = compute_pos_items(test_set.csr_matrix.getrow(user_idx))
        
        if not test_pos_items:
            continue 

        u_gt_pos_mask = np.zeros(test_set.num_items, dtype="int")
        u_gt_pos_mask[test_pos_items] = 1
        
        u_gt_neg_mask = np.ones(test_set.num_items, dtype="int")
        u_gt_neg_mask[train_pos_items] = 0 
        
        item_indices = np.nonzero(u_gt_pos_mask + u_gt_neg_mask)[0]
        
        try:
            item_rank, item_scores = model.rank(user_idx=user_idx, item_indices=item_indices)
        except Exception as e:
            logging.error(f"모델 랭킹 실패: 사용자 {user_idx}, 오류 {e}")
            continue
            
        scaled_scores = compute_normalize_scores(item_scores, sorted=True)
        final_rec_list = []
        final_rec_scores = [] 

        if rerank_flag:
            final_rec_list, final_rec_scores = reranker.rerank(
                user_idx, 
                list(item_rank[:top_k_candidates]), 
                list(scaled_scores[:top_k_candidates]),
                div_mode,
                pers_mode
            )
        else:
            final_rec_list = list(item_rank[:final_k])
            final_rec_scores = list(scaled_scores[:final_k])
        
        if not final_rec_list:
            logging.warning(f"사용자 {user_idx}의 추천 목록이 비어있음")
            continue
            
        recommendation_items[user_idx] = dict(zip(final_rec_list, final_rec_scores))
        
        gt_pos_set = set(test_pos_items)
        results_this_user = {
            **{metric.name: metric.compute(gt_pos=test_pos_items, 
                                           pd_rank=final_rec_list, k=final_k) 
               for metric in metrics},
            
            "MRR@10": rr_at_k(gt_pos=gt_pos_set, pd_rank=final_rec_list, k=final_k),
            "Novelty": novelty(pd_rank=final_rec_list, item_pops=item_pops),
            "CatalogCoverage": catalog_coverage(pd_rank=final_rec_list, item_info=item_info),
            "APLT": aplt(pd_rank=final_rec_list, long_tail_items=long_tail_items),
        }
        for name, value in results_this_user.items(): 
            user_results[name][user_idx] = value

    logging.info("평가 결과 집계 시작")
    avg_results = {
        name: np.mean(list(scores.values())) 
        for name, scores in user_results.items() if scores
    }
    
    aclt_score = aclt(
        recommendation_items=recommendation_items,
        long_tail_items=long_tail_items
    )
    avg_results["ACLT"] = aclt_score
    
    logging.info(f"실험 완료: 모델 {model_name}, 모드 {div_mode}, 개인화 {pers_mode}")
    return avg_results, user_results, recommendation_items