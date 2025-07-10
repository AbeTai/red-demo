from typing import Dict, List, Any, Set
from abc import ABC, abstractmethod

class ModelParamManager:
    """
    モデル種別に応じたパラメータ管理クラス
    """
    
    # モデル種別ごとの訓練パラメータマッピング
    TRAINING_PARAM_MAPPINGS = {
        "matrix_factorization": ["alpha", "factors", "regularization", "iterations"],
        "gnn": ["embedding_dim", "num_layers", "dropout", "lr", "epochs"],
        "deep_learning": ["hidden_dims", "dropout", "lr", "epochs", "batch_size"],
        "neural_collaborative_filtering": ["embedding_dim", "hidden_dims", "dropout", "lr", "epochs"],
        "variational_autoencoder": ["latent_dim", "hidden_dims", "beta", "lr", "epochs"]
    }
    
    # モデル種別ごとの推論パラメータマッピング
    INFERENCE_PARAM_MAPPINGS = {
        "matrix_factorization": ["use_mmr", "lambda_param", "candidate_pool_size"],
        "gnn": ["temperature", "top_k_sampling"],
        "deep_learning": ["temperature", "dropout_inference"],
        "neural_collaborative_filtering": ["temperature"],
        "variational_autoencoder": ["sampling_method", "temperature"]
    }
    
    @classmethod
    def get_training_params(cls, model_type: str) -> List[str]:
        """
        指定されたモデル種別の訓練パラメータ名リストを取得
        
        Args:
            model_type: モデル種別
            
        Returns:
            訓練パラメータ名のリスト
        """
        return cls.TRAINING_PARAM_MAPPINGS.get(model_type, [])
    
    @classmethod
    def get_inference_params(cls, model_type: str) -> List[str]:
        """
        指定されたモデル種別の推論パラメータ名リストを取得
        
        Args:
            model_type: モデル種別
            
        Returns:
            推論パラメータ名のリスト
        """
        return cls.INFERENCE_PARAM_MAPPINGS.get(model_type, [])
    
    @classmethod
    def extract_training_params(cls, model_params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        モデルパラメータから訓練用パラメータを抽出
        
        Args:
            model_params: 全てのモデルパラメータ
            model_type: モデル種別
            
        Returns:
            訓練用パラメータのみを含む辞書
        """
        training_param_names = cls.get_training_params(model_type)
        return {k: v for k, v in model_params.items() if k in training_param_names}
    
    @classmethod
    def extract_inference_params(cls, model_params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        モデルパラメータから推論用パラメータを抽出
        
        Args:
            model_params: 全てのモデルパラメータ
            model_type: モデル種別
            
        Returns:
            推論用パラメータのみを含む辞書
        """
        inference_param_names = cls.get_inference_params(model_type)
        return {k: v for k, v in model_params.items() if k in inference_param_names}
    
    @classmethod
    def validate_model_params(
        cls, 
        model_params: Dict[str, Any], 
        model_type: str,
        model_instance = None
    ) -> Dict[str, Any]:
        """
        モデルパラメータの検証
        
        Args:
            model_params: 検証対象のパラメータ
            model_type: モデル種別
            model_instance: モデルインスタンス（より詳細な検証のため）
            
        Returns:
            検証済みパラメータ
        """
        # モデルインスタンスが利用可能な場合はそれを使用
        if model_instance and hasattr(model_instance, 'validate_params'):
            return model_instance.validate_params(model_params)
        
        # デフォルトの検証
        validated = model_params.copy()
        
        # 未知のパラメータをチェック
        known_params = set(cls.get_training_params(model_type) + cls.get_inference_params(model_type))
        unknown_params = set(validated.keys()) - known_params
        
        if unknown_params:
            print(f"警告: モデル種別 '{model_type}' では未知のパラメータが指定されています: {unknown_params}")
        
        return validated
    
    @classmethod
    def get_supported_model_types(cls) -> List[str]:
        """
        サポートされているモデル種別のリストを取得
        
        Returns:
            サポートされているモデル種別のリスト
        """
        all_types = set(cls.TRAINING_PARAM_MAPPINGS.keys()) | set(cls.INFERENCE_PARAM_MAPPINGS.keys())
        return sorted(list(all_types))
    
    @classmethod
    def add_model_type(
        cls, 
        model_type: str, 
        training_params: List[str] = None,
        inference_params: List[str] = None
    ) -> None:
        """
        新しいモデル種別をサポートリストに追加
        
        Args:
            model_type: 新しいモデル種別
            training_params: 訓練パラメータ名のリスト
            inference_params: 推論パラメータ名のリスト
        """
        if training_params:
            cls.TRAINING_PARAM_MAPPINGS[model_type] = training_params
        if inference_params:
            cls.INFERENCE_PARAM_MAPPINGS[model_type] = inference_params
        
        print(f"モデル種別 '{model_type}' をサポートリストに追加しました")
    
    @classmethod
    def get_param_info(cls, model_type: str) -> Dict[str, List[str]]:
        """
        指定されたモデル種別のパラメータ情報を取得
        
        Args:
            model_type: モデル種別
            
        Returns:
            訓練パラメータと推論パラメータの情報を含む辞書
        """
        return {
            "training_params": cls.get_training_params(model_type),
            "inference_params": cls.get_inference_params(model_type)
        }
    
    @classmethod
    def display_supported_models(cls) -> None:
        """サポートされているモデル種別とそのパラメータを表示"""
        print("サポートされているモデル種別:")
        for model_type in cls.get_supported_model_types():
            info = cls.get_param_info(model_type)
            print(f"\n{model_type}:")
            print(f"  訓練パラメータ: {info['training_params']}")
            print(f"  推論パラメータ: {info['inference_params']}")

class DynamicParamExtractor:
    """
    モデルインスタンスから動的にパラメータ情報を抽出するクラス
    """
    
    @staticmethod
    def extract_from_model(model_instance) -> Dict[str, List[str]]:
        """
        モデルインスタンスからパラメータ情報を動的に抽出
        
        Args:
            model_instance: パラメータ情報を持つモデルインスタンス
            
        Returns:
            パラメータ情報を含む辞書
        """
        training_params = []
        inference_params = []
        
        # モデルインスタンスがパラメータ情報を提供する場合
        if hasattr(model_instance, 'get_training_param_names'):
            training_params = model_instance.get_training_param_names()
        
        if hasattr(model_instance, 'get_inference_param_names'):
            inference_params = model_instance.get_inference_param_names()
        
        return {
            "training_params": training_params,
            "inference_params": inference_params
        }
    
    @staticmethod
    def update_param_manager_from_model(model_instance) -> None:
        """
        モデルインスタンスの情報でModelParamManagerを更新
        
        Args:
            model_instance: パラメータ情報を持つモデルインスタンス
        """
        if not hasattr(model_instance, 'get_model_type'):
            print("警告: モデルインスタンスがget_model_type()メソッドを持っていません")
            return
        
        model_type = model_instance.get_model_type()
        param_info = DynamicParamExtractor.extract_from_model(model_instance)
        
        ModelParamManager.add_model_type(
            model_type,
            param_info["training_params"],
            param_info["inference_params"]
        )