import numpy as np
import logging
from typing import List, Dict, Set, Any, Tuple
from itertools import chain

from .user_profiling import (
    compute_user_preference,
    compute_item_pops,
    compute_user_novelty,
    compute_user_gini_impurity
)

class Reranker:
    def __init__(self, 
                 rerank_config: Dict, 
                 user_profile_config: Dict,
                 train_set: Any, 
                 item_info: Dict[int, List[str]],
                 long_tail_items: Set[int]):
        self.k = rerank_config.get("final_k", 10)
        self.lambda_rel = rerank_config.get("lambda_rel", 0.5)
        self.lambda_div_mix = rerank_config.get("lambda_div_mix", 0.5)
        self.beta_param = rerank_config.get("beta_param", 10.0) 
        self.item_info = item_info
        self.long_tail_items = long_tail_items
        
        logging.info("다양성 데이터 준비 시작")
        self.diversification_data = self._prepare_diversification_data(
            train_set, item_info, user_profile_config
        )
        
        self.item_pops = self.diversification_data["item_pops"]
        self.max_raw_pop = max(self.item_pops.values()) if self.item_pops else 1
        
        logging.info("Reranker 초기화 완료 및 사용자 프로필 준비 완료")

    def _prepare_diversification_data(self, train_set, item_info, config) -> Dict:
        user_prefs = compute_user_preference(
            train_set, item_info, 
            method=config.get("user_pref_method", "MAP"),
            alpha_value=config.get("user_pref_alpha", 1.0)
        )
        item_pops = compute_item_pops(train_set)
        user_novelty_raw = compute_user_novelty(train_set, item_pops) 
        user_gini_raw = compute_user_gini_impurity(user_prefs)

        gini_factors = user_gini_raw
        log1p_gini_values = np.array([np.log1p(v) for v in gini_factors.values() if v > 0])
        mean_log_gini = np.mean(log1p_gini_values) if log1p_gini_values.size > 0 else 0
        std_log_gini = np.std(log1p_gini_values) + 1e-9 
        
        scaled_gini_factors = {}
        for u, v in gini_factors.items():
            log_v = np.log1p(v)
            z_score = (log_v - mean_log_gini) / std_log_gini 
            scaled_v = z_score * 0.5 + 1.0 
            clipped_v = np.clip(scaled_v, 0.0 + 1e-9, 2.0 - 1e-9)
            scaled_gini_factors[u] = clipped_v
        
        novelty_factors = user_novelty_raw
        log1p_novelty_values = np.array([np.log1p(v) for v in novelty_factors.values() if v > 0])
        mean_log_novelty = np.mean(log1p_novelty_values) if log1p_novelty_values.size > 0 else 0
        std_log_novelty = np.std(log1p_novelty_values) + 1e-9

        scaled_novelty_factors = {}
        for u, v in novelty_factors.items():
            log_v = np.log1p(v)
            z_score = (log_v - mean_log_novelty) / std_log_novelty
            scaled_v = z_score * 0.5 + 1.0 
            clipped_v = np.clip(scaled_v, 0.0 + 1e-9, 2.0 - 1e-9)
            scaled_novelty_factors[u] = clipped_v
        
        logging.info("다양성 데이터 준비 완료.")

        return {
            "item_pops": item_pops,
            "user_preferences": user_prefs,
            "user_gini": scaled_gini_factors, 
            "user_novelty": scaled_novelty_factors 
        }

    def rerank(self, 
               user_idx: int, 
               topk_items: List[int], 
               topk_scores: List[float],
               div_mode: str, 
               pers_mode: str
               ) -> Tuple[List[int], List[float]]:
        user_gini = self.diversification_data["user_gini"].get(user_idx, 1.0)
        gini_factor = user_gini if pers_mode == "P" else 1.0

        user_novelty = self.diversification_data["user_novelty"].get(user_idx, 1.0)
        novelty_factor = user_novelty if pers_mode == "P" else 1.0
        
        user_preference = self.diversification_data.get("user_preferences", {}).get(user_idx, {})
        
        final_rank, final_scores = [], []
        candidate_items, candidate_scores = list(topk_items), list(topk_scores)
        
        while len(final_rank) < self.k and candidate_items:
            best_item_local_idx, best_combined_score = -1, -np.inf
            
            recommended_topics = set(chain.from_iterable(self.item_info.get(i, []) for i in final_rank))
            num_long_tail_in_rank = sum(1 for iid in final_rank if iid in self.long_tail_items)
            
            dynamic_beta = 1 / (self.beta_param ** num_long_tail_in_rank)

            for i, item_id in enumerate(candidate_items):
                original_score = candidate_scores[i]
                
                i_div_score = 0.0
                if "I_DIV" in div_mode:
                    i_div_score = self._compute_i_div(item_id, recommended_topics, user_preference, gini_factor)
                
                a_div_score = 0.0
                if "A_DIV" in div_mode:
                    a_div_score = self._compute_a_div(item_id, dynamic_beta, novelty_factor)
                
                if div_mode == "I_DIV":
                    diversification_score = i_div_score
                elif div_mode == "A_DIV":
                    diversification_score = a_div_score
                elif div_mode == "I_DIV+A_DIV":
                    diversification_score = (self.lambda_div_mix * i_div_score) + \
                                             ((1 - self.lambda_div_mix) * a_div_score)
                else:
                    diversification_score = 0
                
                combined_score = (self.lambda_rel * original_score) + \
                                 ((1 - self.lambda_rel) * diversification_score)
                
                if combined_score > best_combined_score:
                    best_combined_score, best_item_local_idx = combined_score, i
            
            if best_item_local_idx != -1:
                selected_item = candidate_items.pop(best_item_local_idx)
                selected_score = candidate_scores.pop(best_item_local_idx)
                final_rank.append(selected_item)
                final_scores.append(selected_score)
            else:
                break
                
        return final_rank, final_scores 

    def _compute_i_div(self, item_id: int, recommended_topics: Set[str], 
                       user_preference: Dict[str, float], gini_factor: float) -> float:
        item_topics = self.item_info.get(item_id, [])
        if not item_topics:
            return 0.0
            
        etc_score = sum(
            user_preference.get(topic, 0.0) 
            for topic in item_topics 
            if topic not in recommended_topics
        )
        
        return gini_factor * etc_score

    def _compute_a_div(self, item_id: int, dynamic_beta: float, 
                       novelty_factor: float) -> float:
        lower_bound = 0.5 
        n_i = self.item_pops.get(item_id, 0)
        
        log_numerator = np.log1p(dynamic_beta * n_i)
        log_denominator = np.log1p(dynamic_beta * self.max_raw_pop)
        
        if log_denominator < 1e-9:
            normalized_log_pop = 0.0
        else:
            normalized_log_pop = log_numerator / log_denominator
            
        scaled_log_pop = normalized_log_pop * (1 - lower_bound)
        item_unpop_score = 1 - scaled_log_pop
    
        return novelty_factor * item_unpop_score