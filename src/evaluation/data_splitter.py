import polars as pl
import pandas as pd
from typing import Tuple, Dict, Set, Any
import numpy as np

class TimeSeriesDataSplitter:
    """
    推薦システム評価用の時系列データ分割クラス
    """
    
    def __init__(self, date_column: str = 'interaction_date') -> None:
        self.date_column: str = date_column
    
    def split_by_date(self, df: pl.DataFrame, split_date: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        日付によるデータ分割
        
        Args:
            df: インタラクションデータを含むDataFrame
            split_date: 分割基準日（YYYYMMDD形式）
            
        Returns:
            (train_df, test_df)のタプル
        """
        train_df: pl.DataFrame = df.filter(pl.col(self.date_column) < split_date)
        test_df: pl.DataFrame = df.filter(pl.col(self.date_column) >= split_date)
        
        return train_df, test_df
    
    def split_by_ratio(self, df: pl.DataFrame, train_ratio: float = 0.8) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        日付を基準とした比率によるデータ分割
        
        Args:
            df: インタラクションデータを含むDataFrame
            train_ratio: 訓練データの比率
            
        Returns:
            (train_df, test_df)のタプル
        """
        # ユニークな日付を取得してソート
        dates: pl.DataFrame = df.select(pl.col(self.date_column)).unique().sort(self.date_column)
        date_list: list = dates[self.date_column].to_list()
        
        # 分割ポイントを計算
        split_index: int = int(len(date_list) * train_ratio)
        split_date: int = date_list[split_index] if split_index < len(date_list) else date_list[-1]
        
        return self.split_by_date(df, split_date)
    
    def get_test_relevant_items(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Dict[int, Set[str]]:
        """
        Get relevant items for each user in test set
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            Dict mapping user_id to set of relevant items in test set
        """
        # Get users that appear in both training and test sets
        train_users = set(train_df['user_id'].unique().to_list())
        test_users = set(test_df['user_id'].unique().to_list())
        common_users = train_users & test_users
        
        # Get relevant items for each user in test set
        user_relevant_items = {}
        for user_id in common_users:
            test_items = test_df.filter(pl.col('user_id') == user_id)['artist'].unique().to_list()
            user_relevant_items[user_id] = set(test_items)
        
        return user_relevant_items
    
    def prepare_evaluation_data(self, df: pl.DataFrame, split_date: int = None, train_ratio: float = 0.8) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[int, Set[str]]]:
        """
        Prepare data for evaluation
        
        Args:
            df: Full dataset
            split_date: Optional specific date to split on (YYYYMMDD format)
            train_ratio: Ratio for training data if split_date not provided
            
        Returns:
            Tuple of (train_df, test_df, user_relevant_items)
        """
        if split_date is not None:
            train_df, test_df = self.split_by_date(df, split_date)
        else:
            train_df, test_df = self.split_by_ratio(df, train_ratio)
        
        user_relevant_items = self.get_test_relevant_items(train_df, test_df)
        
        return train_df, test_df, user_relevant_items
    
    def get_data_stats(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Dict[str, any]:
        """
        Get statistics about the data split
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            Dict with statistics
        """
        train_users = set(train_df['user_id'].unique().to_list())
        test_users = set(test_df['user_id'].unique().to_list())
        common_users = train_users & test_users
        
        train_items = set(train_df['artist'].unique().to_list())
        test_items = set(test_df['artist'].unique().to_list())
        common_items = train_items & test_items
        
        stats = {
            'train_interactions': len(train_df),
            'test_interactions': len(test_df),
            'train_users': len(train_users),
            'test_users': len(test_users),
            'common_users': len(common_users),
            'train_items': len(train_items),
            'test_items': len(test_items),
            'common_items': len(common_items),
            'train_date_range': (train_df[self.date_column].min(), train_df[self.date_column].max()),
            'test_date_range': (test_df[self.date_column].min(), test_df[self.date_column].max())
        }
        
        return stats