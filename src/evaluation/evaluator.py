import polars as pl
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Union
import json
import importlib
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.data_splitter import TimeSeriesDataSplitter
from src.evaluation.model_param_manager import ModelParamManager, DynamicParamExtractor
from src.models.base_recommender import BaseRecommender

class ModelEvaluator:
    """
    モデル評価オーケストレーター
    """
    
    def __init__(self, data_splitter: TimeSeriesDataSplitter = None):
        self.data_splitter = data_splitter or TimeSeriesDataSplitter()
        self.metrics = EvaluationMetrics()
    
    def evaluate_model(
        self,
        model_class: type,
        model_params: Dict[str, Any],
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        user_relevant_items: Dict[int, Set[str]],
        k: int = 5,
        csv_path: str = None,
        include_serendipity: bool = False
    ) -> Dict[str, float]:
        """
        単一モデルの評価（汎用版）
        
        Args:
            model_class: 評価するモデルクラス（BaseRecommenderを継承）
            model_params: モデルのパラメータ
            train_df: 訓練データ
            test_df: テストデータ
            user_relevant_items: user_idから関連アイテムへのマッピング
            k: 生成する推薦数
            csv_path: CSVファイルパス（モデル初期化用）
            include_serendipity: セレンディピティ指標を含めるかどうか
            
        Returns:
            評価指標を含む辞書
        """
        # モデルを初期化
        if csv_path:
            model = model_class(csv_path=csv_path)
        else:
            model = model_class()
        
        # BaseRecommenderを継承していることを確認
        if not isinstance(model, BaseRecommender):
            raise TypeError(f"モデルクラス {model_class.__name__} は BaseRecommender を継承している必要があります")
        
        # モデル種別を取得
        model_type = model.get_model_type()
        
        # パラメータ管理システムを更新（新しいモデルタイプの場合）
        DynamicParamExtractor.update_param_manager_from_model(model)
        
        # パラメータを検証
        validated_params = ModelParamManager.validate_model_params(
            model_params, model_type, model
        )
        
        # 訓練データを準備
        model.prepare_data(train_df)
        
        # 訓練パラメータを抽出
        training_params = ModelParamManager.extract_training_params(
            validated_params, model_type
        )
        
        # 推論パラメータを抽出
        inference_params = ModelParamManager.extract_inference_params(
            validated_params, model_type
        )
        
        # モデルを訓練
        model.train_model(**training_params)
        
        # テストユーザーに対する推薦を生成
        user_recommendations = {}
        test_users = list(user_relevant_items.keys())
        
        for user_id in test_users:
            try:
                # 統一インターフェースで推薦を生成
                recommendations = model.get_recommendations(
                    user_id=user_id,
                    n_recommendations=k,
                    **inference_params
                )
                
                if isinstance(recommendations, list):
                    # アーティスト名を抽出（(artist, score)タプルまたは文字列）
                    recommended_items = [
                        item[0] if isinstance(item, tuple) else item 
                        for item in recommendations
                    ]
                    user_recommendations[user_id] = recommended_items
                else:
                    # エラーケースの処理
                    print(f"ユーザー {user_id} の推薦生成でエラー: {recommendations}")
                    user_recommendations[user_id] = []
                        
            except Exception as e:
                print(f"ユーザー {user_id} の推薦生成中にエラーが発生: {e}")
                user_recommendations[user_id] = []
        
        # 全アイテムを取得（カバレッジ計算用）
        all_items = set(train_df['artist'].unique().to_list())
        
        # ジャンル情報を取得（セレンディピティ指標用）
        item_genres = {}
        if 'genre' in train_df.columns:
            genre_df = train_df.select(['artist', 'genre']).unique()
            item_genres = dict(zip(genre_df['artist'].to_list(), genre_df['genre'].to_list()))
        
        # 推薦を評価
        results = self.metrics.evaluate_all_metrics(
            user_recommendations=user_recommendations,
            user_relevant_items=user_relevant_items,
            all_items=all_items,
            interaction_data=train_df,
            item_features=None,  # 必要に応じて特徴量を追加
            item_genres=item_genres,
            k=k,
            include_serendipity=include_serendipity
        )
        
        return results
    
    def compare_models(
        self,
        model_configs: List[Dict[str, Any]],
        df: pl.DataFrame,
        split_date: int = None,
        train_ratio: float = 0.8,
        k: int = 5,
        csv_path: str = None,
        include_serendipity: bool = False
    ) -> List[Dict[str, Any]]:
        """
        複数モデルの比較
        
        Args:
            model_configs: モデル設定のリスト
            df: 全データセット
            split_date: 分割用のオプション日付
            train_ratio: split_dateが指定されない場合の訓練比率
            k: 推薦数
            csv_path: CSVファイルパス
            
        Returns:
            評価結果のリスト
        """
        # Split data
        train_df, test_df, user_relevant_items = self.data_splitter.prepare_evaluation_data(
            df, split_date, train_ratio
        )
        
        print(f"データ分割: 訓練 {len(train_df)}件、テスト {len(test_df)}件")
        print(f"テストデータを持つユーザー: {len(user_relevant_items)}人")
        
        results = []
        
        for config in model_configs:
            model_class = config['model_class']
            model_params = config['model_params']
            model_name = config['model_name']
            
            print(f"\n{model_name}を評価中...")
            
            try:
                metrics = self.evaluate_model(
                    model_class=model_class,
                    model_params=model_params,
                    train_df=train_df,
                    test_df=test_df,
                    user_relevant_items=user_relevant_items,
                    k=k,
                    csv_path=csv_path,
                    include_serendipity=include_serendipity
                )
                
                result = {
                    'model_name': model_name,
                    'model_params': model_params,
                    'metrics': metrics
                }
                
                results.append(result)
                
                print(f"{model_name}の結果:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                
            except Exception as e:
                print(f"{model_name}の評価中にエラーが発生: {e}")
                continue
        
        return results