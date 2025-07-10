import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class ResultsManager:
    """
    CSV形式で評価結果を管理するクラス（JSON指標対応版）
    """
    
    def __init__(self, results_csv_path: str = "results/evaluation_results.csv") -> None:
        """
        結果管理クラスの初期化
        
        Args:
            results_csv_path: 評価結果を保存するCSVファイルパス
        """
        self.results_csv_path: str = results_csv_path
        self.ensure_results_file()
    
    def ensure_results_file(self) -> None:
        """
        結果CSVファイルが適切なヘッダーで存在することを確認
        """
        if not os.path.exists(self.results_csv_path):
            # resultsディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(self.results_csv_path), exist_ok=True)
            # ヘッダーを含むファイルを作成
            headers = ['data_name', 'model_name', 'execute_date', 'param', 'metrics']
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.results_csv_path, index=False)
    
    def load_existing_results(self) -> pd.DataFrame:
        """
        CSVから既存の結果を読み込み
        
        Returns:
            既存結果を含むDataFrame
        """
        if os.path.exists(self.results_csv_path):
            return pd.read_csv(self.results_csv_path)
        else:
            return pd.DataFrame()
    
    def save_results(
        self,
        data_name: str,
        model_name: str,
        model_params: Dict[str, Any],
        metrics: Dict[str, float],
        execute_date: Optional[str] = None
    ) -> None:
        """
        評価結果をCSVに保存
        
        Args:
            data_name: データセット名（CSVファイル名から拡張子を除いたもの）
            model_name: モデルクラス名
            model_params: モデルパラメータの辞書
            metrics: 評価指標の辞書
            execute_date: 実行日付（YYYYMMDD形式）（デフォルトは今日）
        """
        if execute_date is None:
            execute_date = datetime.now().strftime('%Y%m%d')
        
        # モデルパラメータをJSON文字列に変換
        param_json: str = json.dumps(model_params, sort_keys=True)
        
        # 評価指標をJSON文字列に変換
        metrics_json: str = json.dumps(metrics, sort_keys=True)
        
        # 新しい行を作成
        new_row: Dict[str, str] = {
            'data_name': data_name,
            'model_name': model_name,
            'execute_date': execute_date,
            'param': param_json,
            'metrics': metrics_json
        }
        
        # 既存の同一設定がないかチェック
        existing_df = self.load_existing_results()
        if not existing_df.empty:
            # 重複エントリをチェック
            duplicate_mask = (
                (existing_df['data_name'] == data_name) &
                (existing_df['model_name'] == model_name) &
                (existing_df['execute_date'] == execute_date) &
                (existing_df['param'] == param_json)
            )
            
            if duplicate_mask.any():
                print(f"警告: 重複エントリが見つかりました {data_name}, {model_name}, {execute_date}")
                print("既存エントリを更新中...")
                # 既存行を更新
                existing_df.loc[duplicate_mask, list(new_row.keys())] = list(new_row.values())
                existing_df.to_csv(self.results_csv_path, index=False)
                return
        
        # 新しい行を追加
        new_df = pd.DataFrame([new_row])
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # CSVに保存
        combined_df.to_csv(self.results_csv_path, index=False)
        print(f"結果を{self.results_csv_path}に保存しました")
    
    def expand_metrics_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        メトリクスJSONカラムを個別カラムに展開
        
        Args:
            df: 元のDataFrame
            
        Returns:
            メトリクスが展開されたDataFrame
        """
        if df.empty or 'metrics' not in df.columns:
            return df
        
        expanded_rows = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            try:
                metrics = json.loads(row['metrics'])
                row_dict.update(metrics)
            except (json.JSONDecodeError, TypeError):
                pass
            expanded_rows.append(row_dict)
        
        return pd.DataFrame(expanded_rows)
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        全結果のサマリーを取得
        
        Returns:
            結果サマリーを含むDataFrame
        """
        df = self.load_existing_results()
        if df.empty:
            return df
        
        # メトリクスを展開
        expanded_df = self.expand_metrics_columns(df)
        
        # パラメータのサマリーを追加
        def summarize_params(param_json: str) -> str:
            try:
                params = json.loads(param_json)
                summary_parts = []
                for key, value in params.items():
                    if isinstance(value, float):
                        summary_parts.append(f"{key}={value:.2f}")
                    else:
                        summary_parts.append(f"{key}={value}")
                return ", ".join(summary_parts)
            except:
                return param_json
        
        expanded_df['param_summary'] = expanded_df['param'].apply(summarize_params)
        return expanded_df
    
    def get_best_results(self, metric: str = 'precision_at_5') -> pd.DataFrame:
        """
        各モデル/データの組み合わせで最良の結果を取得
        
        Args:
            metric: ランキングに使用する指標
            
        Returns:
            最良結果を含むDataFrame
        """
        df = self.load_existing_results()
        if df.empty:
            return df
        
        # メトリクスを展開
        expanded_df = self.expand_metrics_columns(df)
        
        # パラメータのサマリーを追加
        def summarize_params(param_json: str) -> str:
            try:
                params = json.loads(param_json)
                summary_parts = []
                for key, value in params.items():
                    if isinstance(value, float):
                        summary_parts.append(f"{key}={value:.2f}")
                    else:
                        summary_parts.append(f"{key}={value}")
                return ", ".join(summary_parts)
            except:
                return param_json
        
        expanded_df['param_summary'] = expanded_df['param'].apply(summarize_params)
        
        # データ名とモデル名でグループ化し、各グループの最良結果を取得
        if metric in expanded_df.columns:
            best_results = expanded_df.loc[expanded_df.groupby(['data_name', 'model_name'])[metric].idxmax()]
            return best_results.sort_values(metric, ascending=False)
        else:
            return expanded_df
    
    def compare_models(self, data_name: Optional[str] = None) -> pd.DataFrame:
        """
        特定のデータセットに対するモデル比較
        
        Args:
            data_name: フィルタリングするデータ名（オプション）
            
        Returns:
            モデル比較を含むDataFrame
        """
        df = self.load_existing_results()
        if df.empty:
            return df
        
        if data_name:
            df = df[df['data_name'] == data_name]
        
        # 各モデルの最新結果を取得
        latest_results = df.loc[df.groupby(['data_name', 'model_name'])['execute_date'].idxmax()]
        
        # メトリクスを展開
        expanded_df = self.expand_metrics_columns(latest_results)
        
        # パラメータサマリーを追加
        expanded_df['param_summary'] = expanded_df['param'].apply(
            lambda x: json.dumps(json.loads(x), sort_keys=True) if x else ""
        )
        
        # 基本カラム + 主要指標を選択
        base_cols = ['data_name', 'model_name', 'execute_date', 'param_summary']
        metric_cols = [col for col in expanded_df.columns if col not in base_cols + ['param', 'metrics']]
        
        selected_cols = base_cols + metric_cols
        available_cols = [col for col in selected_cols if col in expanded_df.columns]
        
        result_df = expanded_df[available_cols]
        
        # precision_at_5が存在する場合はそれでソート、なければ最初の指標でソート
        if 'precision_at_5' in result_df.columns:
            result_df = result_df.sort_values('precision_at_5', ascending=False)
        elif metric_cols:
            result_df = result_df.sort_values(metric_cols[0], ascending=False)
        
        return result_df
    
    def get_metric_names(self) -> List[str]:
        """
        データベース内で使用されている全ての指標名を取得
        
        Returns:
            指標名のリスト
        """
        df = self.load_existing_results()
        if df.empty or 'metrics' not in df.columns:
            return []
        
        all_metrics = set()
        for _, row in df.iterrows():
            try:
                metrics = json.loads(row['metrics'])
                all_metrics.update(metrics.keys())
            except (json.JSONDecodeError, TypeError):
                continue
        
        return sorted(list(all_metrics))
    
    def migrate_old_format(self, old_csv_path: str) -> None:
        """
        古いフォーマット（個別カラム）から新しいフォーマット（JSON）に移行
        
        Args:
            old_csv_path: 古いフォーマットのCSVファイルパス
        """
        if not os.path.exists(old_csv_path):
            print(f"旧フォーマットファイルが見つかりません: {old_csv_path}")
            return
        
        old_df = pd.read_csv(old_csv_path)
        
        # 古いフォーマットの指標カラムを特定
        metric_columns = [col for col in old_df.columns 
                         if col not in ['data_name', 'model_name', 'execute_date', 'param']]
        
        migrated_rows = []
        for _, row in old_df.iterrows():
            # 指標を辞書にまとめる
            metrics = {col: row[col] for col in metric_columns if pd.notna(row[col])}
            
            # パラメータを解析
            try:
                model_params = json.loads(row['param'])
            except:
                model_params = {}
            
            # 新しいフォーマットで保存
            self.save_results(
                data_name=row['data_name'],
                model_name=row['model_name'],
                model_params=model_params,
                metrics=metrics,
                execute_date=row['execute_date']
            )
        
        print(f"{old_csv_path}から{len(old_df)}件のレコードを移行しました")