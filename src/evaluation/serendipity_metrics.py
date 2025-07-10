import numpy as np
import pandas as pd
import polars as pl
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from collections import Counter
import math
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from itertools import combinations

class SerendipityMetrics:
    """
    セレンディピティ推薦用の評価指標を提供するクラス
    """
    
    @staticmethod
    def intra_list_diversity(
        recommended_items: List[str],
        item_features: Dict[str, Dict[str, Any]],
        similarity_metric: str = 'cosine',
        feature_columns: Optional[List[str]] = None
    ) -> float:
        """
        Intra-list Diversity（リスト内多様性）を計算
        推薦リスト内のアイテム間の平均ペアワイズ距離
        
        Args:
            recommended_items: 推薦アイテムのリスト
            item_features: アイテムの特徴量辞書
            similarity_metric: 類似度メトリクス ('cosine', 'euclidean')
            feature_columns: 使用する特徴量カラム（Noneの場合は全て使用）
            
        Returns:
            平均ペアワイズ距離（多様性スコア）
        """
        if len(recommended_items) < 2:
            return 0.0
        
        # アイテムペアの全組み合わせを生成
        item_pairs: List[Tuple[str, str]] = list(combinations(recommended_items, 2))
        
        if not item_pairs:
            return 0.0
        
        distances: List[float] = []
        
        for item1, item2 in item_pairs:
            if item1 in item_features and item2 in item_features:
                # 特徴量ベースの距離計算
                if feature_columns:
                    features1 = [item_features[item1].get(col, 0) for col in feature_columns]
                    features2 = [item_features[item2].get(col, 0) for col in feature_columns]
                else:
                    # ジャンルベースの単純な距離（異なれば1、同じなら0）
                    genre1 = item_features[item1].get('genre', '')
                    genre2 = item_features[item2].get('genre', '')
                    distance = 1.0 if genre1 != genre2 else 0.0
                    distances.append(distance)
                    continue
                
                # ベクトル間の距離計算
                if similarity_metric == 'cosine':
                    similarity = cosine_similarity([features1], [features2])[0][0]
                    distance = 1.0 - similarity
                elif similarity_metric == 'euclidean':
                    distance = euclidean_distances([features1], [features2])[0][0]
                else:
                    raise ValueError(f"サポートされていない類似度メトリクス: {similarity_metric}")
                
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    @staticmethod
    def catalog_coverage(
        all_recommended_items: List[str],
        catalog_items: Set[str]
    ) -> float:
        """
        カタログカバレッジを計算
        全アイテムのうち推薦されるアイテムの割合
        
        Args:
            all_recommended_items: 全ユーザーに対する推薦アイテムの全リスト
            catalog_items: カタログ内の全アイテム集合
            
        Returns:
            カタログカバレッジ（0-1）
        """
        if len(catalog_items) == 0:
            return 0.0
        
        unique_recommended: Set[str] = set(all_recommended_items)
        covered_items: Set[str] = unique_recommended & catalog_items
        
        return len(covered_items) / len(catalog_items)
    
    @staticmethod
    def genre_coverage(
        recommended_items: List[str],
        item_genres: Dict[str, str]
    ) -> float:
        """
        ジャンルカバレッジを計算
        推薦されるアイテムが属するジャンルの多様性
        
        Args:
            recommended_items: 推薦アイテムのリスト
            item_genres: アイテムとジャンルのマッピング
            
        Returns:
            ユニークジャンル数 / 全ジャンル数
        """
        if not recommended_items:
            return 0.0
        
        # 推薦アイテムのジャンルを取得
        recommended_genres: Set[str] = set()
        for item in recommended_items:
            if item in item_genres:
                recommended_genres.add(item_genres[item])
        
        # 全ジャンル数を取得
        all_genres: Set[str] = set(item_genres.values())
        
        if len(all_genres) == 0:
            return 0.0
        
        return len(recommended_genres) / len(all_genres)
    
    @staticmethod
    def shannon_entropy(
        recommended_items: List[str],
        item_categories: Dict[str, str]
    ) -> float:
        """
        Shannon Entropy を計算
        推薦リストのカテゴリ分布の均等性を測定
        
        Args:
            recommended_items: 推薦アイテムのリスト
            item_categories: アイテムとカテゴリのマッピング
            
        Returns:
            Shannon Entropy値
        """
        if not recommended_items:
            return 0.0
        
        # カテゴリ分布を取得
        category_counts: Counter = Counter()
        for item in recommended_items:
            if item in item_categories:
                category_counts[item_categories[item]] += 1
        
        if len(category_counts) == 0:
            return 0.0
        
        # 確率分布を計算
        total_items: int = len(recommended_items)
        entropy: float = 0.0
        
        for count in category_counts.values():
            probability: float = count / total_items
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    @staticmethod
    def gini_coefficient(
        item_recommendation_counts: Dict[str, int]
    ) -> float:
        """
        Gini係数を計算
        推薦の偏りを測定（0に近いほど均等）
        
        Args:
            item_recommendation_counts: アイテムごとの推薦回数
            
        Returns:
            Gini係数（0-1）
        """
        if not item_recommendation_counts:
            return 0.0
        
        counts: List[int] = list(item_recommendation_counts.values())
        counts.sort()
        
        n: int = len(counts)
        cumulative_sum: float = sum((i + 1) * count for i, count in enumerate(counts))
        
        total_sum: float = sum(counts)
        
        if total_sum == 0:
            return 0.0
        
        gini: float = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
        
        return gini
    
    @staticmethod
    def novelty_score(
        recommended_items: List[str],
        item_popularity: Dict[str, float]
    ) -> float:
        """
        Novelty（新規性）スコアを計算
        アイテムの人気度の逆数として定義
        
        Args:
            recommended_items: 推薦アイテムのリスト
            item_popularity: アイテムの人気度（高いほど人気）
            
        Returns:
            平均新規性スコア
        """
        if not recommended_items:
            return 0.0
        
        novelty_scores: List[float] = []
        
        for item in recommended_items:
            if item in item_popularity:
                popularity = item_popularity[item]
                # 人気度が0の場合を避けるため、小さな値を追加
                novelty = -math.log2(max(popularity, 1e-10))
                novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    @staticmethod
    def unexpectedness_score(
        user_id: int,
        recommended_items: List[str],
        user_profile: Dict[str, float],
        item_features: Dict[str, Dict[str, float]],
        similarity_threshold: float = 0.5
    ) -> float:
        """
        Unexpectedness（予期しなさ）スコアを計算
        ユーザープロファイルとの類似度の逆数
        
        Args:
            user_id: ユーザーID
            recommended_items: 推薦アイテムのリスト
            user_profile: ユーザープロファイル（特徴量の重み）
            item_features: アイテムの特徴量
            similarity_threshold: 期待値として扱う類似度の閾値
            
        Returns:
            予期しなさスコア
        """
        if not recommended_items or not user_profile:
            return 0.0
        
        unexpected_scores: List[float] = []
        
        for item in recommended_items:
            if item in item_features:
                # ユーザープロファイルとアイテム特徴量の類似度計算
                profile_vector = list(user_profile.values())
                item_vector = list(item_features[item].values())
                
                # ベクトル長を合わせる
                min_len = min(len(profile_vector), len(item_vector))
                profile_vector = profile_vector[:min_len]
                item_vector = item_vector[:min_len]
                
                if profile_vector and item_vector:
                    similarity = cosine_similarity([profile_vector], [item_vector])[0][0]
                    # 類似度が低いほど予期しない（unexpectedなス）
                    unexpectedness = max(0.0, similarity_threshold - similarity)
                    unexpected_scores.append(unexpectedness)
        
        return np.mean(unexpected_scores) if unexpected_scores else 0.0
    
    @staticmethod
    def compute_item_popularity(
        interaction_data: Union[pl.DataFrame, pd.DataFrame],
        item_column: str = 'artist',
        interaction_column: str = 'play_count'
    ) -> Dict[str, float]:
        """
        アイテムの人気度を計算
        
        Args:
            interaction_data: インタラクションデータ
            item_column: アイテムカラム名
            interaction_column: インタラクション値カラム名
            
        Returns:
            アイテムの人気度辞書（正規化済み）
        """
        if isinstance(interaction_data, pl.DataFrame):
            # Polarsの場合
            popularity_df = interaction_data.group_by(item_column).agg(
                pl.col(interaction_column).sum().alias('total_interactions')
            )
            popularity_dict = dict(zip(
                popularity_df[item_column].to_list(),
                popularity_df['total_interactions'].to_list()
            ))
        else:
            # Pandasの場合
            popularity_dict = interaction_data.groupby(item_column)[interaction_column].sum().to_dict()
        
        # 正規化（0-1の範囲に）
        max_popularity = max(popularity_dict.values()) if popularity_dict else 1
        normalized_popularity = {
            item: pop / max_popularity 
            for item, pop in popularity_dict.items()
        }
        
        return normalized_popularity
    
    @staticmethod
    def create_user_profile(
        user_id: int,
        interaction_data: Union[pl.DataFrame, pd.DataFrame],
        item_features: Dict[str, Dict[str, float]],
        user_column: str = 'user_id',
        item_column: str = 'artist',
        interaction_column: str = 'play_count'
    ) -> Dict[str, float]:
        """
        ユーザープロファイルを作成
        
        Args:
            user_id: ユーザーID
            interaction_data: インタラクションデータ
            item_features: アイテム特徴量
            user_column: ユーザーカラム名
            item_column: アイテムカラム名
            interaction_column: インタラクション値カラム名
            
        Returns:
            ユーザープロファイル（特徴量の重み付き平均）
        """
        # ユーザーのインタラクションデータを取得
        if isinstance(interaction_data, pl.DataFrame):
            user_interactions = interaction_data.filter(
                pl.col(user_column) == user_id
            )
            user_items = user_interactions[item_column].to_list()
            user_weights = user_interactions[interaction_column].to_list()
        else:
            user_interactions = interaction_data[interaction_data[user_column] == user_id]
            user_items = user_interactions[item_column].tolist()
            user_weights = user_interactions[interaction_column].tolist()
        
        if not user_items:
            return {}
        
        # 重み付き平均でプロファイルを計算
        profile: Dict[str, float] = {}
        total_weight: float = sum(user_weights)
        
        for item, weight in zip(user_items, user_weights):
            if item in item_features:
                for feature, value in item_features[item].items():
                    if feature not in profile:
                        profile[feature] = 0.0
                    profile[feature] += (weight / total_weight) * value
        
        return profile
    
    @staticmethod
    def evaluate_serendipity_metrics(
        user_recommendations: Dict[int, List[str]],
        interaction_data: Union[pl.DataFrame, pd.DataFrame],
        item_features: Dict[str, Dict[str, Any]],
        item_genres: Dict[str, str],
        catalog_items: Set[str]
    ) -> Dict[str, float]:
        """
        セレンディピティ関連の全指標を評価
        
        Args:
            user_recommendations: ユーザーごとの推薦リスト
            interaction_data: インタラクションデータ
            item_features: アイテム特徴量
            item_genres: アイテムジャンルマッピング
            catalog_items: カタログアイテム集合
            
        Returns:
            評価指標の辞書
        """
        # 全推薦アイテムを収集
        all_recommended: List[str] = []
        for recommendations in user_recommendations.values():
            all_recommended.extend(recommendations)
        
        # アイテム人気度を計算
        item_popularity = SerendipityMetrics.compute_item_popularity(interaction_data)
        
        # 推薦回数をカウント
        item_rec_counts = Counter(all_recommended)
        
        # 各指標を計算
        results: Dict[str, float] = {}
        
        # 1. Intra-list Diversity（各ユーザーの平均）
        diversity_scores: List[float] = []
        for user_id, recommendations in user_recommendations.items():
            diversity = SerendipityMetrics.intra_list_diversity(
                recommendations, {item: {'genre': item_genres.get(item, '')} for item in recommendations}
            )
            diversity_scores.append(diversity)
        results['intra_list_diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # 2. Catalog Coverage
        results['catalog_coverage'] = SerendipityMetrics.catalog_coverage(all_recommended, catalog_items)
        
        # 3. Genre Coverage（各ユーザーの平均）
        genre_coverage_scores: List[float] = []
        for recommendations in user_recommendations.values():
            coverage = SerendipityMetrics.genre_coverage(recommendations, item_genres)
            genre_coverage_scores.append(coverage)
        results['genre_coverage'] = np.mean(genre_coverage_scores) if genre_coverage_scores else 0.0
        
        # 4. Shannon Entropy（各ユーザーの平均）
        entropy_scores: List[float] = []
        for recommendations in user_recommendations.values():
            entropy = SerendipityMetrics.shannon_entropy(recommendations, item_genres)
            entropy_scores.append(entropy)
        results['shannon_entropy'] = np.mean(entropy_scores) if entropy_scores else 0.0
        
        # 5. Gini Coefficient
        results['gini_coefficient'] = SerendipityMetrics.gini_coefficient(item_rec_counts)
        
        # 6. Novelty Score（各ユーザーの平均）
        novelty_scores: List[float] = []
        for recommendations in user_recommendations.values():
            novelty = SerendipityMetrics.novelty_score(recommendations, item_popularity)
            novelty_scores.append(novelty)
        results['novelty_score'] = np.mean(novelty_scores) if novelty_scores else 0.0
        
        # 7. Unexpectedness Score（ユーザープロファイルが利用可能な場合）
        unexpected_scores: List[float] = []
        for user_id, recommendations in user_recommendations.items():
            if item_features:  # 特徴量が利用可能な場合のみ
                user_profile = SerendipityMetrics.create_user_profile(
                    user_id, interaction_data, item_features
                )
                if user_profile:
                    unexpectedness = SerendipityMetrics.unexpectedness_score(
                        user_id, recommendations, user_profile, item_features
                    )
                    unexpected_scores.append(unexpectedness)
        results['unexpectedness_score'] = np.mean(unexpected_scores) if unexpected_scores else 0.0
        
        return results