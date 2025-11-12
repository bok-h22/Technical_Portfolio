from itertools import chain
from typing import List, Dict, Set, Any
import numpy as np

def rr_at_k(pd_rank: List[int], gt_pos: Set[int], k: int) -> float:
    """
    """
    for i, item in enumerate(pd_rank[:k]):
        if item in gt_pos:
            return 1.0 / (i + 1)
    return 0.0

def catalog_coverage(pd_rank: List[int], item_info: Dict[int, List[str]]) -> float:
    """
    """
    recommended_categories = {
        category
        for iidx in pd_rank
        for category in item_info.get(iidx, [])
    }
    
    all_categories = set(chain.from_iterable(item_info.values()))
    
    if not all_categories:
        return 0.0
        
    return len(recommended_categories) / len(all_categories)

def novelty(pd_rank: List[int], item_pops: Dict[int, int]) -> float:
    """
    [Castells et al., 2022. Novelty and diversity in recommender systems]
    """
    if not pd_rank:
        return 0.0
        
    total_interaction = sum(item_pops.values())
    if total_interaction == 0:
        return 0.0

    self_information = 0.0
    for item in pd_rank:
        item_pop = item_pops.get(item)
        if item_pop and item_pop > 0:
            self_information += -np.log2(item_pop / total_interaction)
            
    return self_information / len(pd_rank)

def arp(pd_rank: List[int], item_pops: Dict[int, int]) -> float:
    """
    [Yin et al., 2012. Challenging the long tail recommendation]
    """
    if not pd_rank:
        return 0.0
        
    rec_pop = [item_pops.get(pd, 0) for pd in pd_rank]
    return np.mean(rec_pop)

def aplt(pd_rank: List[int], long_tail_items: Set[int]) -> float:
    """
    [Abdollahpouri et al., 2017. Controlling popularity bias...]
    """
    if not pd_rank:
        return 0.0
        
    long_tail_count = sum(1 for item_id in pd_rank if item_id in long_tail_items)
    return long_tail_count / len(pd_rank)

def aclt(
    recommendation_items: Dict[int, Dict[int, float]],
    long_tail_items: Set[Any]
) -> float:
    total_long_tail_count = len(long_tail_items)
    if total_long_tail_count == 0:
        return 0.0

    all_recommended_items = set()
    for rec_dict in recommendation_items.values():
        all_recommended_items.update(rec_dict.keys())

    covered_long_tail_items = all_recommended_items.intersection(long_tail_items)

    aclt_score = len(covered_long_tail_items) / total_long_tail_count
    
    return aclt_score