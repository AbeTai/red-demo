import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Any

class EvaluationMetrics:
    """
    推薦システム用の評価指標を提供するクラス
    """
    
    @staticmethod
    def precision_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        Precision@Kを計算
        
        Args:
            recommended_items: 推薦アイテムのリスト（関連度順）
            relevant_items: ユーザーにとって関連するアイテムの集合
            k: 評価対象となる上位推薦数
            
        Returns:
            Precision@K値
        """
        if k == 0:
            return 0.0
        
        top_k_items: List[str] = recommended_items[:k]
        relevant_top_k: List[str] = [item for item in top_k_items if item in relevant_items]
        
        return len(relevant_top_k) / k
    
    @staticmethod
    def recall_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        Recall@Kを計算
        
        Args:
            recommended_items: 推薦アイテムのリスト（関連度順）
            relevant_items: ユーザーにとって関連するアイテムの集合
            k: 評価対象となる上位推薦数
            
        Returns:
            Recall@K値
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_items: List[str] = recommended_items[:k]
        relevant_top_k: List[str] = [item for item in top_k_items if item in relevant_items]
        
        return len(relevant_top_k) / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        NDCG@K（正規化割引累積利得）を計算
        
        Args:
            recommended_items: 推薦アイテムのリスト（関連度順）
            relevant_items: ユーザーにとって関連するアイテムの集合
            k: 評価対象となる上位推薦数
            
        Returns:
            NDCG@K値
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        # DCG@Kを計算
        dcg: float = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                # バイナリ関連性：関連ありなら1、なしなら0
                relevance: float = 1.0
                dcg += relevance / np.log2(i + 2)  # +2はlog2(1) = 0のため
        
        # IDCG@K（理想的DCG）を計算
        # バイナリ関連性の場合、理想ランキングは全ての関連アイテムを最初に配置
        ideal_length: int = min(k, len(relevant_items))
        idcg: float = 0.0
        for i in range(ideal_length):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def hit_rate_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        Hit Rate@K（上位k件に少なくとも1つの関連アイテムが含まれるか）を計算
        
        Args:
            recommended_items: 推薦アイテムのリスト（関連度順）
            relevant_items: ユーザーにとって関連するアイテムの集合
            k: 評価対象となる上位推薦数
            
        Returns:
            Hit Rate@K値（0または1）
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k_items: List[str] = recommended_items[:k]
        for item in top_k_items:
            if item in relevant_items:
                return 1.0
        
        return 0.0
    
    @staticmethod
    def coverage(all_recommended_items: List[str], all_items: Set[str]) -> float:
        """
        カバレッジ（推薦された全アイテムの割合）を計算
        
        Args:
            all_recommended_items: 全ユーザーに対する推薦アイテムのリスト
            all_items: 利用可能な全アイテムの集合
            
        Returns:
            カバレッジ値
        """
        if len(all_items) == 0:
            return 0.0
        
        unique_recommended: Set[str] = set(all_recommended_items)
        return len(unique_recommended) / len(all_items)
    
    @staticmethod
    def evaluate_recommendations(
        user_recommendations: Dict[int, List[str]],
        user_relevant_items: Dict[int, Set[str]],
        all_items: Set[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        全ユーザーの推薦結果を評価
        
        Args:
            user_recommendations: user_idから推薦アイテムリストへのマッピング
            user_relevant_items: user_idから関連アイテム集合へのマッピング
            all_items: 利用可能な全アイテムの集合
            k: 評価対象となる上位推薦数
            
        Returns:
            評価指標を含む辞書
        """
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        ndcg_scores: List[float] = []
        hit_rate_scores: List[float] = []
        all_recommended_items: List[str] = []
        
        for user_id, recommended_items in user_recommendations.items():
            relevant_items: Set[str] = user_relevant_items.get(user_id, set())
            
            if len(relevant_items) > 0:  # 関連アイテムを持つユーザーのみ評価
                precision_scores.append(
                    EvaluationMetrics.precision_at_k(recommended_items, relevant_items, k)
                )
                recall_scores.append(
                    EvaluationMetrics.recall_at_k(recommended_items, relevant_items, k)
                )
                ndcg_scores.append(
                    EvaluationMetrics.ndcg_at_k(recommended_items, relevant_items, k)
                )
                hit_rate_scores.append(
                    EvaluationMetrics.hit_rate_at_k(recommended_items, relevant_items, k)
                )
            
            all_recommended_items.extend(recommended_items[:k])
        
        # 平均指標を計算
        results: Dict[str, float] = {
            f'precision_at_{k}': np.mean(precision_scores) if precision_scores else 0.0,
            f'recall_at_{k}': np.mean(recall_scores) if recall_scores else 0.0,
            f'ndcg_at_{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            f'hit_rate_at_{k}': np.mean(hit_rate_scores) if hit_rate_scores else 0.0,
            'coverage': EvaluationMetrics.coverage(all_recommended_items, all_items)
        }
        
        return results
    
    @staticmethod
    def evaluate_all_metrics(
        user_recommendations: Dict[int, List[str]],
        user_relevant_items: Dict[int, Set[str]],
        all_items: Set[str],
        interaction_data = None,
        item_features: Dict[str, Dict[str, Any]] = None,
        item_genres: Dict[str, str] = None,
        k: int = 5,
        include_serendipity: bool = False
    ) -> Dict[str, float]:
        """
        標準指標とセレンディピティ指標の両方を評価
        
        Args:
            user_recommendations: user_idから推薦アイテムリストへのマッピング
            user_relevant_items: user_idから関連アイテム集合へのマッピング
            all_items: 利用可能な全アイテムの集合
            interaction_data: インタラクションデータ（セレンディピティ指標用）
            item_features: アイテム特徴量（セレンディピティ指標用）
            item_genres: アイテムジャンルマッピング（セレンディピティ指標用）
            k: 評価対象となる上位推薦数
            include_serendipity: セレンディピティ指標を含めるかどうか
            
        Returns:
            全評価指標を含む辞書
        """
        # 標準指標を計算
        results = EvaluationMetrics.evaluate_recommendations(
            user_recommendations, user_relevant_items, all_items, k
        )
        
        # セレンディピティ指標を追加
        if include_serendipity and interaction_data is not None:
            try:
                from .serendipity_metrics import SerendipityMetrics
                
                # カタログアイテム集合を準備
                catalog_items = all_items
                
                # ジャンル情報がない場合はダミーを作成
                if item_genres is None:
                    item_genres = {item: 'Unknown' for item in all_items}
                
                # セレンディピティ指標を評価
                serendipity_results = SerendipityMetrics.evaluate_serendipity_metrics(
                    user_recommendations=user_recommendations,
                    interaction_data=interaction_data,
                    item_features=item_features or {},
                    item_genres=item_genres,
                    catalog_items=catalog_items
                )
                
                # 結果をマージ
                results.update(serendipity_results)
                
            except ImportError:
                print("Warning: SerendipityMetrics not available")
        
        return results