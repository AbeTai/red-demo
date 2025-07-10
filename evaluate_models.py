#!/usr/bin/env python3
"""
音楽推薦システムのモデル評価CLIツール
"""

import argparse
import polars as pl
import pandas as pd
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from evaluation.evaluator import ModelEvaluator
from evaluation.data_splitter import TimeSeriesDataSplitter
from evaluation.results_manager import ResultsManager
from evaluation.config_loader import ModelConfigLoader, LegacyConfigAdapter
from models.matrix_factorization.music_recommender import MusicRecommender
from models.matrix_factorization.music_recommender_mmr import MusicRecommenderMMR

def get_model_configs(config_file: str = None, model_types: List[str] = None) -> List[Dict[str, Any]]:
    """
    評価対象のモデル設定を取得（新しい設定システム対応）
    
    Args:
        config_file: 設定ファイルのパス（デフォルト: configs/model_configs.json）
        model_types: 評価対象のモデルタイプリスト（デフォルト: matrix_factorization）
    
    Returns:
        モデル設定のリスト
    """
    try:
        # 新しい設定システムを使用
        if config_file is None:
            config_file = "configs/model_configs.json"
        
        config_loader = ModelConfigLoader(config_file)
        
        if model_types is None:
            # デフォルトは行列因子分解モデルのみ
            model_types = ["matrix_factorization"]
        
        all_configs = []
        for model_type in model_types:
            configs = config_loader.get_models_by_type(model_type)
            all_configs.extend(configs)
        
        if not all_configs:
            print("警告: 設定ファイルからモデルが見つかりませんでした。レガシー設定を使用します。")
            return get_legacy_model_configs()
        
        print(f"設定ファイルから{len(all_configs)}個のモデル設定を読み込みました")
        return all_configs
        
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        print("レガシー設定を使用します。")
        return get_legacy_model_configs()

def get_legacy_model_configs() -> List[Dict[str, Any]]:
    """
    レガシー（ハードコード）モデル設定を取得
    
    Returns:
        モデル設定のリスト
    """
    configs: List[Dict[str, Any]] = [
        {
            'model_name': 'MusicRecommender',
            'model_class': MusicRecommender,
            'model_params': {
                'alpha': 0.4,
                'factors': 50,
                'regularization': 0.1,
                'iterations': 20
            }
        },
        {
            'model_name': 'MusicRecommender',
            'model_class': MusicRecommender,
            'model_params': {
                'alpha': 0.8,
                'factors': 50,
                'regularization': 0.1,
                'iterations': 20
            }
        },
        {
            'model_name': 'MusicRecommenderMMR',
            'model_class': MusicRecommenderMMR,
            'model_params': {
                'alpha': 0.4,
                'factors': 50,
                'regularization': 0.1,
                'iterations': 20,
                'lambda_param': 0.5,
                'candidate_pool_size': 20
            }
        },
        {
            'model_name': 'MusicRecommenderMMR',
            'model_class': MusicRecommenderMMR,
            'model_params': {
                'alpha': 0.4,
                'factors': 50,
                'regularization': 0.1,
                'iterations': 20,
                'lambda_param': 0.3,
                'candidate_pool_size': 20
            }
        },
        {
            'model_name': 'MusicRecommenderMMR',
            'model_class': MusicRecommenderMMR,
            'model_params': {
                'alpha': 0.4,
                'factors': 50,
                'regularization': 0.1,
                'iterations': 20,
                'lambda_param': 0.7,
                'candidate_pool_size': 20
            }
        }
    ]
    
    return configs

def main() -> None:
    """
    メイン関数：コマンドライン引数を解析して評価を実行
    """
    parser = argparse.ArgumentParser(description='音楽推薦モデルの評価')
    parser.add_argument('--csv-path', default='data/user_artist_plays.csv',
                        help='ユーザー-アーティスト再生データを含むCSVファイルパス')
    parser.add_argument('--results-path', default='results/evaluation_results.csv',
                        help='評価結果を保存するパス')
    parser.add_argument('--split-date', type=int, default=None,
                        help='分割日付（YYYYMMDD形式）。未指定の場合はtrain-ratioを使用')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='訓練データの比率（デフォルト: 0.8）')
    parser.add_argument('--k', type=int, default=5,
                        help='評価対象の推薦数（デフォルト: 5）')
    parser.add_argument('--config-file', default=None,
                        help='カスタムモデル設定のJSONファイル (デフォルト: configs/model_configs.json)')
    parser.add_argument('--model-types', nargs='+', default=['matrix_factorization'],
                        help='評価対象のモデルタイプ (デフォルト: matrix_factorization)')
    parser.add_argument('--list-models', action='store_true',
                        help='利用可能なモデルを表示して終了')
    parser.add_argument('--execute-date', default=None,
                        help='実行日付（YYYYMMDD形式）（デフォルト: 今日）')
    parser.add_argument('--show-summary', action='store_true',
                        help='既存結果のサマリーを表示')
    parser.add_argument('--show-best', action='store_true',
                        help='各モデルの最適結果を表示')
    parser.add_argument('--compare-models', action='store_true',
                        help='全モデルの最新結果を比較')
    parser.add_argument('--include-serendipity', action='store_true',
                        help='セレンディピティ指標を含めて評価')
    
    args = parser.parse_args()
    
    # モデル一覧表示の処理
    if args.list_models:
        try:
            config_loader = ModelConfigLoader(args.config_file or "configs/model_configs.json")
            config_loader.display_available_models()
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            print("レガシー設定を表示します:")
            print("利用可能なモデル:")
            for config in get_legacy_model_configs():
                print(f"  - {config['model_name']}: {config['model_params']}")
        return
    
    # Initialize results manager
    results_manager = ResultsManager(args.results_path)
    
    # Show summaries if requested
    if args.show_summary:
        summary = results_manager.get_results_summary()
        if not summary.empty:
            print("=== 評価結果サマリー ===")
            print(summary.to_string(index=False))
        else:
            print("評価結果が見つかりません。")
        return
    
    if args.show_best:
        best = results_manager.get_best_results()
        if not best.empty:
            print("=== モデル別最適結果 ===")
            print(best[['data_name', 'model_name', 'param_summary', 'precision_at_5', 'recall_at_5', 'ndcg_at_5']].to_string(index=False))
        else:
            print("評価結果が見つかりません。")
        return
    
    if args.compare_models:
        data_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        comparison = results_manager.compare_models(data_name)
        if not comparison.empty:
            print(f"=== {data_name} のモデル比較 ===")
            print(comparison.to_string(index=False))
        else:
            print("比較用の評価結果が見つかりません。")
        return
    
    # CSVファイルの存在確認
    if not os.path.exists(args.csv_path):
        print(f"エラー: CSVファイルが見つかりません: {args.csv_path}")
        return
    
    # データ読み込み
    print(f"{args.csv_path}からデータを読み込み中...")
    df = pl.read_csv(args.csv_path)
    print(f"{len(df)}件のレコードを読み込みました")
    
    # 必須カラムの存在確認
    required_columns = ['user_id', 'artist', 'play_count', 'interaction_date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"エラー: 必須カラムが不足しています: {missing_columns}")
        print(f"利用可能なカラム: {df.columns}")
        return
    
    # Get model configurations
    model_configs = get_model_configs(args.config_file, args.model_types)
    
    # Set execute date
    execute_date = args.execute_date or datetime.now().strftime('%Y%m%d')
    
    # Initialize evaluator
    data_splitter = TimeSeriesDataSplitter(date_column='interaction_date')
    evaluator = ModelEvaluator(data_splitter)
    
    # Get data name from CSV path
    data_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    
    print(f"{len(model_configs)}個のモデル設定で評価を開始します...")
    print(f"データ名: {data_name}")
    print(f"実行日: {execute_date}")
    if args.include_serendipity:
        print("セレンディピティ指標を含めて評価します...")
    
    # 既存の古い形式ファイルがあれば移行
    if os.path.exists(args.results_path):
        try:
            old_df = pd.read_csv(args.results_path)
            # 古い形式かチェック（metricsカラムがなく、個別指標カラムがある）
            if 'metrics' not in old_df.columns and 'precision_at_5' in old_df.columns:
                print("古い形式のCSVファイルを検出しました。新しい形式に移行します...")
                backup_path = args.results_path.replace('.csv', '_backup.csv')
                old_df.to_csv(backup_path, index=False)
                results_manager.migrate_old_format(args.results_path)
                print(f"バックアップを作成しました: {backup_path}")
        except Exception as e:
            print(f"ファイル移行中にエラーが発生しました: {e}")

    # モデルを評価
    results = evaluator.compare_models(
        model_configs=model_configs,
        df=df,
        split_date=args.split_date,
        train_ratio=args.train_ratio,
        k=args.k,
        csv_path=args.csv_path,
        include_serendipity=args.include_serendipity
    )
    
    # 結果保存
    print(f"\n結果を{args.results_path}に保存中...")
    for result in results:
        results_manager.save_results(
            data_name=data_name,
            model_name=result['model_name'],
            model_params=result['model_params'],
            metrics=result['metrics'],
            execute_date=execute_date
        )
    
    # 最終サマリー表示
    print("\n=== 評価完了 ===")
    print(f"{len(results)}個のモデルを評価しました")
    print(f"結果の保存先: {args.results_path}")
    
    # この実行の結果を表示
    if results:
        print("\n=== この実行の結果 ===")
        for result in results:
            print(f"\n{result['model_name']} (params: {json.dumps(result['model_params'], sort_keys=True)}):")
            # 標準指標とセレンディピティ指標を分けて表示
            standard_metrics = ['precision_at_5', 'recall_at_5', 'ndcg_at_5', 'hit_rate_at_5', 'coverage']
            serendipity_metrics = [m for m in result['metrics'].keys() if m not in standard_metrics]
            
            print("  標準指標:")
            for metric in standard_metrics:
                if metric in result['metrics']:
                    print(f"    {metric}: {result['metrics'][metric]:.4f}")
            
            if serendipity_metrics and args.include_serendipity:
                print("  セレンディピティ指標:")
                for metric in serendipity_metrics:
                    print(f"    {metric}: {result['metrics'][metric]:.4f}")

if __name__ == "__main__":
    main()