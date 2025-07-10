from abc import ABC, abstractmethod
import polars as pl
from typing import Dict, List, Tuple, Any, Optional, Union

class BaseRecommender(ABC):
    """
    推薦システムの抽象基底クラス
    
    全ての推薦モデルはこのクラスを継承し、必要なメソッドを実装する必要があります。
    """
    
    def __init__(self, csv_path: str = 'data/user_artist_plays.csv', model_dir: str = 'weights/') -> None:
        """
        推薦システムの基本初期化
        
        Args:
            csv_path: ユーザー-アイテムインタラクションデータのCSVファイルパス
            model_dir: モデルを保存するディレクトリ
        """
        self.csv_path: str = csv_path
        self.model_dir: str = model_dir
        self.df: Optional[pl.DataFrame] = None
        self.is_trained: bool = False
    
    @abstractmethod
    def prepare_data(self, train_df: Optional[pl.DataFrame] = None) -> None:
        """
        データを前処理してモデル訓練用に準備
        
        Args:
            train_df: 訓練用データフレーム（Noneの場合はself.dfを使用）
        """
        pass
    
    @abstractmethod
    def train_model(self, **params: Any) -> None:
        """
        モデルを訓練
        
        Args:
            **params: モデル固有の訓練パラメータ
        """
        pass
    
    @abstractmethod
    def get_recommendations(
        self, 
        user_id: int, 
        n_recommendations: int = 5, 
        **kwargs: Any
    ) -> Union[List[str], List[Tuple[str, float]], str]:
        """
        指定したユーザーに対する推薦を生成
        
        Args:
            user_id: 推薦対象のユーザーID
            n_recommendations: 推薦アイテム数
            **kwargs: モデル固有の推薦パラメータ
            
        Returns:
            推薦アイテムのリスト（アイテム名のみ、またはスコア付きタプル）
            エラーの場合は文字列メッセージ
        """
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """
        モデルの種別を返す
        
        Returns:
            モデル種別文字列 ("matrix_factorization", "gnn", "deep_learning", etc.)
        """
        pass
    
    def load_data(self, csv_path: Optional[str] = None) -> None:
        """
        CSVファイルからデータを読み込み
        
        Args:
            csv_path: CSVファイルパス（省略時はinit時のパスを使用）
        """
        if csv_path is None:
            csv_path = self.csv_path
        self.df = pl.read_csv(csv_path)
        print(f"{csv_path}から{len(self.df)}件のインタラクション記録を読み込みました")
    
    def save_model(self, model_path: str) -> None:
        """
        モデルを保存（デフォルト実装）
        
        Args:
            model_path: 保存先パス
        """
        raise NotImplementedError("このモデルタイプではモデル保存がサポートされていません")
    
    def load_model(self, model_path: str) -> None:
        """
        モデルを読み込み（デフォルト実装）
        
        Args:
            model_path: 読み込み元パス
        """
        raise NotImplementedError("このモデルタイプではモデル読み込みがサポートされていません")
    
    def get_training_param_names(self) -> List[str]:
        """
        このモデルで使用される訓練パラメータ名のリストを返す
        
        Returns:
            パラメータ名のリスト
        """
        # デフォルトは空リスト（サブクラスでオーバーライド推奨）
        return []
    
    def get_inference_param_names(self) -> List[str]:
        """
        このモデルで使用される推論パラメータ名のリストを返す
        
        Returns:
            パラメータ名のリスト
        """
        # デフォルトは空リスト（サブクラスでオーバーライド推奨）
        return []
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータの検証とデフォルト値設定
        
        Args:
            params: 検証対象のパラメータ辞書
            
        Returns:
            検証済みパラメータ辞書
        """
        # デフォルト実装では何もしない（サブクラスでオーバーライド可能）
        return params
    
    def __str__(self) -> str:
        """オブジェクトの文字列表現"""
        return f"{self.__class__.__name__}(model_type={self.get_model_type()})"
    
    def __repr__(self) -> str:
        """オブジェクトの詳細表現"""
        return f"{self.__class__.__name__}(csv_path='{self.csv_path}', model_dir='{self.model_dir}')"