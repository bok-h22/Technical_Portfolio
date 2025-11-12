from collections import Counter, defaultdict
from itertools import chain, groupby
from operator import itemgetter
from typing import Dict, List, Set, Any
import numpy as np
import logging

from .metrics import novelty 

def compute_item_pops(train_set: Any) -> Dict[int, int]:
    item_pops = Counter(train_set.uir_tuple[1])
    return dict(item_pops)

def compute_long_tail_items(
    item_pops: Dict[int, int], 
    head_percentage: float = 0.2
) -> Set[int]:
    if not item_pops:
        return set()
    
    # 인기도 순 정렬
    sorted_items = sorted(item_pops.items(), key=lambda item: item[1], reverse=True)
    num_head_items = int(len(sorted_items) * head_percentage)
    long_tail_tuples = sorted_items[num_head_items:]
    
    return {item_id for item_id, count in long_tail_tuples}

def compute_user_preference(
    train_set: Any, 
    item_info: Dict[int, List[str]], 
    method: str = "MLE", 
    alpha_value: float = 1.0
) -> Dict[int, Dict[str, float]]:
    user_category_counts = defaultdict(lambda: defaultdict(int))
    all_categories = set(chain.from_iterable(item_info.values()))
    K = len(all_categories) 
    
    if K == 0:
        logging.warning("item_info에 카테고리 정보가 없음. 빈 선호도 반환")
        return {}

    for uid_idx, iid_idx, _ in zip(*train_set.uir_tuple):
        if iid_idx in item_info:
            for category in item_info[iid_idx]:
                user_category_counts[uid_idx][category] += 1
    
    user_preferences = {}
    
    for uid_idx, category_counts in user_category_counts.items():
        total_count = sum(category_counts.values())
        
        if method == "MLE":
            if total_count == 0: continue
            user_preferences[uid_idx] = {
                cat: category_counts.get(cat, 0) / total_count
                for cat in all_categories
            }
        
        elif method == "MAP":
            denominator = total_count + (alpha_value * K)
            if denominator == 0: continue
            user_preferences[uid_idx] = {
                cat: (category_counts.get(cat, 0) + alpha_value) / denominator
                for cat in all_categories
            }
        else:
            raise ValueError(f"알 수 없는 선호도 계산 방법: {method}")
            
    return user_preferences

def compute_user_novelty(
    train_set: Any, 
    item_pops: Dict[int, int],
) -> Dict[int, float]:
    user_item_pairs = sorted(zip(train_set.uir_tuple[0], train_set.uir_tuple[1]))
    user_novelty_raw = {}

    for u_idx, group in groupby(user_item_pairs, key=itemgetter(0)):
        item_list = [item[1] for item in group]
        user_novelty_raw[u_idx] = novelty(item_list, item_pops)
    
    return user_novelty_raw

def compute_user_gini_impurity(
    user_profile: Dict[int, Dict[str, float]]
) -> Dict[int, float]:
    if not user_profile:
        return {}
        
    uids = list(user_profile.keys())
    prob_matrix = np.array([list(dist.values()) for dist in user_profile.values()])
    gini_impurity_values = 1 - np.sum(np.square(prob_matrix), axis=1)
    
    return dict(zip(uids, gini_impurity_values))