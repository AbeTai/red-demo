import json
import importlib
import os
from typing import Dict, List, Any, Optional, Type
from pathlib import Path

class ModelConfigLoader:
    """
    モデル設定の読み込みと管理を行うクラス
    """
    
    def __init__(self, config_path: str = "configs/model_configs.json"):
        """
        設定ローダーの初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.config_data = None
        self.load_config()
    
    def load_config(self) -> None:
        """設定ファイルを読み込み"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config_data = json.load(f)
        
        print(f"設定ファイルを読み込みました: {self.config_path}")
    
    def get_all_model_configs(self) -> List[Dict[str, Any]]:
        """
        全てのモデル設定を取得
        
        Returns:
            モデル設定のリスト
        """
        if not self.config_data:
            raise ValueError("設定データが読み込まれていません")
        
        all_configs = []
        
        # 各モデルタイプの設定を統合
        for model_type_key in self.config_data:
            if model_type_key.endswith('_models'):
                model_configs = self.config_data[model_type_key]
                for config in model_configs:
                    # モデルクラスを動的に読み込み
                    try:
                        model_class = self._load_model_class(
                            config['model_module'],
                            config['model_class']
                        )
                        
                        processed_config = {
                            'model_name': config['model_name'],
                            'model_class': model_class,
                            'model_params': config['model_params'],
                            'description': config.get('description', ''),
                            'model_type': model_type_key.replace('_models', ''),
                            'module_path': config['model_module']
                        }
                        all_configs.append(processed_config)
                    except Exception as e:
                        print(f"警告: モデル {config['model_name']} の読み込みに失敗しました: {e}")
                        continue
        
        return all_configs
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        指定されたタイプのモデル設定を取得
        
        Args:
            model_type: モデルタイプ ("matrix_factorization", "gnn", "deep_learning", etc.)
            
        Returns:
            指定されたタイプのモデル設定のリスト
        """
        all_configs = self.get_all_model_configs()
        return [config for config in all_configs if config['model_type'] == model_type]
    
    def get_model_config_by_name(self, model_name: str, model_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        モデル名とパラメータでモデル設定を検索
        
        Args:
            model_name: モデル名
            model_params: 一致させるパラメータ（オプション）
            
        Returns:
            一致するモデル設定、見つからない場合はNone
        """
        all_configs = self.get_all_model_configs()
        
        for config in all_configs:
            if config['model_name'] == model_name:
                if model_params is None:
                    return config
                
                # パラメータが一致するかチェック
                if self._params_match(config['model_params'], model_params):
                    return config
        
        return None
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        デフォルト設定を取得
        
        Returns:
            デフォルト設定の辞書
        """
        return self.config_data.get('default_config', {})
    
    def get_available_model_types(self) -> List[str]:
        """
        利用可能なモデルタイプのリストを取得
        
        Returns:
            モデルタイプのリスト
        """
        if not self.config_data:
            return []
        
        return [
            key.replace('_models', '') 
            for key in self.config_data.keys() 
            if key.endswith('_models')
        ]
    
    def add_model_config(
        self, 
        model_type: str,
        model_name: str,
        model_module: str,
        model_class_name: str,
        model_params: Dict[str, Any],
        description: str = ""
    ) -> None:
        """
        新しいモデル設定を追加
        
        Args:
            model_type: モデルタイプ
            model_name: モデル名
            model_module: モデルが定義されているモジュールパス
            model_class_name: モデルクラス名
            model_params: モデルパラメータ
            description: モデルの説明
        """
        key = f"{model_type}_models"
        
        if key not in self.config_data:
            self.config_data[key] = []
        
        new_config = {
            "model_name": model_name,
            "model_module": model_module,
            "model_class": model_class_name,
            "model_params": model_params,
            "description": description
        }
        
        self.config_data[key].append(new_config)
        print(f"モデル設定を追加しました: {model_name}")
    
    def save_config(self, output_path: str = None) -> None:
        """
        設定ファイルを保存
        
        Args:
            output_path: 出力パス（未指定の場合は元のファイルに上書き）
        """
        if output_path is None:
            output_path = self.config_path
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, indent=2, ensure_ascii=False)
        
        print(f"設定ファイルを保存しました: {output_path}")
    
    def _load_model_class(self, module_path: str, class_name: str) -> Type:
        """
        モデルクラスを動的に読み込み
        
        Args:
            module_path: モジュールパス
            class_name: クラス名
            
        Returns:
            読み込まれたクラス
        """
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class
        except ImportError as e:
            raise ImportError(f"モジュール '{module_path}' の読み込みに失敗: {e}")
        except AttributeError as e:
            raise AttributeError(f"クラス '{class_name}' がモジュール '{module_path}' に見つかりません: {e}")
    
    def _params_match(self, config_params: Dict[str, Any], target_params: Dict[str, Any]) -> bool:
        """
        パラメータが一致するかチェック
        
        Args:
            config_params: 設定ファイルのパラメータ
            target_params: 比較対象のパラメータ
            
        Returns:
            一致する場合True
        """
        for key, value in target_params.items():
            if key not in config_params or config_params[key] != value:
                return False
        return True
    
    def display_available_models(self) -> None:
        """利用可能なモデルの一覧を表示"""
        print("利用可能なモデル:")
        
        for model_type in self.get_available_model_types():
            print(f"\n{model_type.upper()}:")
            models = self.get_models_by_type(model_type)
            
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model['model_name']}")
                print(f"     説明: {model['description']}")
                print(f"     パラメータ: {model['model_params']}")
                print()

class LegacyConfigAdapter:
    """
    既存のハードコードされた設定を新しい形式に変換するアダプター
    """
    
    @staticmethod
    def convert_legacy_configs(legacy_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        従来の設定形式を新しい形式に変換
        
        Args:
            legacy_configs: 従来の設定リスト
            
        Returns:
            新しい形式の設定リスト
        """
        converted = []
        
        for config in legacy_configs:
            # モジュールパスとクラス名を推測
            model_name = config['model_name']
            
            if 'MMR' in model_name:
                module_path = "models.matrix_factorization.music_recommender_mmr"
                class_name = "MusicRecommenderMMR"
            else:
                module_path = "models.matrix_factorization.music_recommender"
                class_name = "MusicRecommender"
            
            converted_config = {
                'model_name': model_name,
                'model_module': module_path,
                'model_class': class_name,
                'model_params': config['model_params'],
                'description': f"Legacy config for {model_name}"
            }
            
            converted.append(converted_config)
        
        return converted