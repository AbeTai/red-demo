# 音楽推薦システム

協調フィルタリングとMMR（Maximal Marginal Relevance）を使用した包括的な音楽推薦システムです。

## 概要

このプロジェクトは、以下の機能を提供する音楽推薦システムです：

- **協調フィルタリング**: Implicit ALSアルゴリズムを使用した推薦
- **MMR推薦**: 関連性と多様性のバランスを取った推薦
- **人口統計学フィルタリング**: 性別・年齢による推薦結果のフィルタリング
- **評価システム**: 複数の評価指標による推薦モデルの性能評価
- **Webインターフェース**: Streamlitを使用した直感的なUI

## ディレクトリ構成

```
.
├── README.md                    # このファイル
├── data_generator.py           # サンプルデータ生成スクリプト
├── app.py                      # メインのStreamlitアプリケーション
├── evaluate_models.py          # モデル評価CLIツール
├── evaluation_results.csv      # 評価結果保存ファイル
├── user_artist_plays.csv       # ユーザー-アーティスト再生データ
├── models/                     # 推薦モデル
│   ├── recommender.py         # 基本ALSモデル
│   └── recommender_mmr.py     # MMR拡張モデル
├── src/evaluation/             # 評価系モジュール
│   ├── metrics.py             # 評価指標
│   ├── data_splitter.py       # 時系列データ分割
│   ├── evaluator.py           # モデル評価
│   └── results_manager.py     # 評価結果管理
└── weights/                    # 訓練済みモデル保存場所
```

## 機能説明

### 1. 推薦アルゴリズム

#### 標準推薦（Implicit ALS）
- 再生回数を信頼度として使用
- 信頼度関数: `1 + α × 再生回数`
- αパラメータで重み付けを調整可能

#### MMR推薦
- 関連性と多様性のバランスを調整
- λパラメータ: `0 = 多様性重視, 1 = 関連性重視`
- アーティスト埋め込みのコサイン類似度を使用

### 2. データ構造

CSVファイルのスキーマ：
```csv
user_id,artist,play_count,gender,age,interaction_date,genre
1,Taylor Swift,41,Male,25-29,20210427,Pop
```

- `user_id`: ユーザーID
- `artist`: アーティスト名
- `play_count`: 再生回数
- `gender`: 性別（Male/Female/Other）
- `age`: 年齢カテゴリ（5歳刻み）
- `interaction_date`: インタラクション日付（YYYYMMDD）
- `genre`: 音楽ジャンル

### 3. 評価指標

- **Precision@K**: 上位K件の推薦中の関連アイテム割合
- **Recall@K**: 関連アイテム中の推薦できた割合
- **NDCG@K**: 正規化割引累積利得
- **Hit Rate@K**: 上位K件に関連アイテムが含まれる割合
- **Coverage**: 推薦された全アイテムの網羅率

## セットアップ

### 必要パッケージのインストール

```bash
# uvを使用する場合
uv add polars implicit scikit-learn streamlit

# pipを使用する場合
pip install polars implicit scikit-learn streamlit
```

### サンプルデータの生成

```bash
python data_generator.py
```

これにより、1000ユーザー、20アーティストの合成データが生成されます。

## 使用方法

### 1. Webアプリケーション

```bash
streamlit run app.py
```

機能：
- ユーザーID直接入力
- アーティスト選択による検索
- 人口統計学フィルタリング
- 標準推薦とMMR推薦の同時表示
- MMRパラメータのリアルタイム調整

### 2. モデル評価

```bash
# 基本評価
python evaluate_models.py --csv-path user_artist_plays.csv --k 5

# 結果表示
python evaluate_models.py --show-summary
python evaluate_models.py --compare-models
python evaluate_models.py --show-best
```

#### 評価オプション

- `--csv-path`: データファイルパス
- `--k`: 評価対象推薦数（デフォルト：5）
- `--split-date`: 分割日付（YYYYMMDD）
- `--train-ratio`: 訓練データ比率（デフォルト：0.8）
- `--results-path`: 結果保存パス

### 3. 単体モデル実行

```bash
# 標準モデル
python models/recommender.py --csv-path user_artist_plays.csv --user-id 1

# MMRモデル  
python models/recommender_mmr.py --csv-path user_artist_plays.csv --user-id 1 --lambda-param 0.5
```

## 評価結果の管理

評価結果は`evaluation_results.csv`に自動保存されます：

| data_name | model_name | execute_date | param | precision_at_5 | recall_at_5 | ... |
|-----------|------------|-------------|-------|----------------|-------------|-----|
| user_artist_plays | MusicRecommenderMMR | 20250711 | {"alpha": 0.4, "lambda_param": 0.7} | 0.2048 | 0.4389 | ... |

- 同一設定での重複実行を自動検出
- JSON形式でパラメータを記録
- 日付別の実行履歴を保持

## カスタマイズ

### 新しい評価指標の追加

`src/evaluation/metrics.py`の`EvaluationMetrics`クラスに静的メソッドを追加：

```python
@staticmethod
def new_metric(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    # 新しい指標の実装
    pass
```

### モデルの追加

1. `models/`に新しいクラスを作成
2. `evaluate_models.py`の`get_model_configs()`に設定を追加
3. 必要に応じて`src/evaluation/evaluator.py`を更新

### データソースの変更

1. `data_generator.py`を修正してカスタムデータを生成
2. または既存のCSVファイルを指定の形式に変換

## パフォーマンス

### 推薦結果例（user_artist_plays.csvでの評価）

| Model | λ | Precision@5 | Recall@5 | NDCG@5 | Hit Rate@5 |
|-------|---|-------------|----------|---------|-------------|
| Standard | - | 0.2018 | 0.4301 | 0.3244 | 0.6728 |
| MMR | 0.3 | 0.2044 | 0.4388 | 0.3307 | 0.6935 |
| MMR | 0.5 | 0.2028 | 0.4307 | 0.3267 | 0.6843 |
| **MMR** | **0.7** | **0.2048** | **0.4389** | **0.3289** | **0.6947** |

**最適設定**: MMR with λ=0.7 が最高性能を示しています。

## 開発とコントリビューション

### 型チェック

すべてのコードは型ヒント付きで記述されており、mypyでの型チェックが可能です：

```bash
mypy models/ src/ *.py
```

### テスト

新機能追加時は対応するテストを`tests/`ディレクトリに追加してください。

### コーディング規約

- 日本語コメントを使用
- 型ヒントを必須とする
- docstringでArgs/Returnsを明記

## トラブルシューティング

### よくある問題

1. **モジュールが見つからない**
   ```bash
   # パッケージが正しくインストールされているか確認
   pip list | grep implicit
   ```

2. **メモリエラー**
   ```bash
   # 大規模データセットの場合、chunk処理を検討
   # またはバッチサイズを小さくする
   ```

3. **評価結果が表示されない**
   ```bash
   # evaluation_results.csvが存在し、正しい形式か確認
   head evaluation_results.csv
   ```

### パフォーマンス最適化

- 大規模データセット（>100万レコード）の場合：
  - バッチ処理の実装を検討
  - スパースマトリックスの最適化
  - メモリ効率的なデータ分割

詳細な技術文書や追加機能については、各モジュールのdocstringを参照してください。