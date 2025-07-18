# 音楽推薦システム

協調フィルタリングとMMR（Maximal Marginal Relevance）を統合した包括的な音楽推薦システムです。ハッシュ化されたユーザーIDとインタラクティブな埋め込み可視化機能を備えています。

## 概要

このプロジェクトは、以下の機能を提供する音楽推薦システムです：

- **協調フィルタリング**: Implicit ALSアルゴリズムを使用した推薦
- **MMR推薦**: 関連性と多様性のバランスを取った推薦
- **ユーザーベース推薦**: ユーザー類似度に基づく協調フィルタリング
- **人口統計学フィルタリング**: 性別・年齢による推薦結果のフィルタリング
- **埋め込み可視化**: t-SNE/UMAPによるユーザー・アーティスト埋め込みの2D可視化
- **評価システム**: 複数の評価指標による推薦モデルの性能評価
- **Webインターフェース**: Streamlitを使用した直感的なUI

## ディレクトリ構成

```
.
├── README.md                         # このファイル
├── CLAUDE.md                         # Claude Code用プロジェクト指示書
├── data_generator.py                 # サンプルデータ生成（ハッシュID対応）
├── app.py                           # メイン推薦アプリケーション
├── app_rec-reason.py                # アイテムベース推薦（理由付き）
├── app_user-base_rec-reason.py      # ユーザーベース推薦（理由付き）
├── evaluate_models.py               # モデル評価CLIツール
├── data/
│   └── user_artist_plays.csv       # ユーザー-アーティスト再生データ（ハッシュID）
├── models/matrix_factorization/     # 推薦モデル
│   ├── base_recommender.py         # 基底クラス
│   ├── music_recommender.py        # 基本ALSモデル
│   ├── music_recommender_mmr.py    # MMR拡張モデル
│   └── user_based_recommender.py   # ユーザーベース推薦
├── sandbox/                         # 視覚化・実験ツール
│   ├── artist_embedding_visualizer.py  # アーティスト埋め込み可視化
│   └── user_embedding_visualizer.py    # ユーザー埋め込み可視化
├── src/evaluation/                  # 評価系モジュール
│   ├── evaluation_metrics.py       # 評価指標
│   ├── time_series_data_splitter.py  # 時系列データ分割
│   ├── model_evaluator.py          # モデル評価
│   └── results_manager.py          # 評価結果管理
├── results/                         # 評価結果
│   └── evaluation_results.csv      # 評価結果保存ファイル
└── weights/                         # 訓練済みモデル保存場所
```

## 機能説明

### 1. 推薦アルゴリズム

#### アイテムベース推薦（Implicit ALS）
- 再生回数を信頼度として使用
- 信頼度関数: `1 + α × 再生回数`
- αパラメータで重み付けを調整可能

#### MMR推薦（Maximal Marginal Relevance）
- 関連性と多様性のバランスを調整
- λパラメータ: `0 = 多様性重視, 1 = 関連性重視`
- アーティスト埋め込みのコサイン類似度を使用

#### ユーザーベース推薦
- ALS user_factorsを使用したユーザー類似度計算
- コサイン類似度によるトップK類似ユーザー取得
- 類似ユーザーの視聴履歴から重み付け推薦

### 2. データ構造

CSVファイルのスキーマ：
```csv
user_id,gender,age,artist,genre,interaction_date,play_count
79b0aa00,Male,65-70,BTS,K-Pop,2021-04-27,41
```

- `user_id`: ハッシュ化されたユーザーID（SHA256の8文字）
- `gender`: 性別（Male/Female/Other）
- `age`: 年齢カテゴリ（5歳刻み）
- `artist`: アーティスト名
- `genre`: 音楽ジャンル
- `interaction_date`: インタラクション日付（YYYYMMDD）
- `play_count`: 再生回数

### 3. 埋め込み可視化

#### アーティスト埋め込み可視化
- ALS item_factorsの2次元可視化
- ジャンル別色分け表示
- t-SNE/UMAP次元圧縮対応
- インタラクティブなホバー情報

#### ユーザー埋め込み可視化
- ALS user_factorsの2次元可視化
- クリック選択によるユーザーID取得
- ユーザー間類似度の視覚的検証
- 視聴履歴比較機能

### 4. 評価指標

- **Precision@K**: 上位K件の推薦中の関連アイテム割合
- **Recall@K**: 関連アイテム中の推薦できた割合
- **NDCG@K**: 正規化割引累積利得
- **Hit Rate@K**: 上位K件に関連アイテムが含まれる割合
- **Coverage**: 推薦された全アイテムの網羅率

## セットアップ

### 必要パッケージのインストール

```bash
# uvを使用する場合
uv add polars implicit scikit-learn streamlit plotly umap-learn

# pipを使用する場合
pip install polars implicit scikit-learn streamlit plotly umap-learn
```

### サンプルデータの生成

```bash
python data_generator.py
```

これにより、ハッシュ化されたユーザーIDを持つ1000ユーザー、20アーティストの合成データが生成されます。

## 使用方法

### 1. Webアプリケーション

#### メイン推薦アプリ
```bash
streamlit run app.py
```

#### アイテムベース推薦（理由付き）
```bash
streamlit run app_rec-reason.py
```

#### ユーザーベース推薦（理由付き）
```bash
streamlit run app_user-base_rec-reason.py
```

#### 埋め込み可視化ツール
```bash
# アーティスト埋め込み可視化
streamlit run sandbox/artist_embedding_visualizer.py

# ユーザー埋め込み可視化
streamlit run sandbox/user_embedding_visualizer.py
```

**主な機能：**
- ハッシュ化されたユーザーID対応
- アーティスト選択による検索
- 人口統計学フィルタリング
- 標準推薦とMMR推薦の同時表示
- MMRパラメータのリアルタイム調整
- 推薦理由の表示
- インタラクティブな埋め込み可視化

### 2. モデル評価

```bash
# 基本評価
python evaluate_models.py --csv-path data/user_artist_plays.csv --k 5

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
# MMRモデル
python models/matrix_factorization/music_recommender_mmr.py --csv-path data/user_artist_plays.csv --user-id 79b0aa00 --lambda-param 0.5

# ユーザーベースモデル
python models/matrix_factorization/user_based_recommender.py --csv-path data/user_artist_plays.csv --user-id 79b0aa00 --n-similar-users 10
```

## 評価結果の管理

評価結果は`results/evaluation_results.csv`に自動保存されます：

| data_name | model_name | execute_date | param | precision_at_5 | recall_at_5 | ... |
|-----------|------------|-------------|-------|----------------|-------------|-----|
| user_artist_plays | MusicRecommenderMMR | 20250718 | {"alpha": 0.4, "lambda_param": 0.7} | 0.2048 | 0.4389 | ... |

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

1. `models/matrix_factorization/`に新しいクラスを作成
2. `evaluate_models.py`の`get_model_configs()`に設定を追加
3. 必要に応じて`src/evaluation/model_evaluator.py`を更新

### データソースの変更

1. `data_generator.py`を修正してカスタムデータを生成（ハッシュID対応）
2. または既存のCSVファイルを指定の形式に変換
3. ハッシュIDが必要な場合は`generate_hash_user_id()`関数を使用

### 埋め込み可視化のカスタマイズ

1. `sandbox/`ディレクトリでt-SNE/UMAPパラメータを調整
2. クリック選択機能やホバー情報をカスタマイズ
3. 新しい次元圧縮手法の追加

## パフォーマンス

### 推薦結果例（ハッシュID対応データでの評価）

| Model | Parameters | Precision@5 | Recall@5 | NDCG@5 | Hit Rate@5 |
|-------|------------|-------------|----------|---------|-------------|
| MusicRecommenderMMR | α=0.4, λ=0.3 | 0.2044 | 0.4388 | 0.3307 | 0.6935 |
| MusicRecommenderMMR | α=0.4, λ=0.5 | 0.2028 | 0.4307 | 0.3267 | 0.6843 |
| **MusicRecommenderMMR** | **α=0.4, λ=0.7** | **0.2048** | **0.4389** | **0.3289** | **0.6947** |
| UserBasedRecommender | α=0.4, K=10 | 0.1985 | 0.4256 | 0.3201 | 0.6654 |

**最適設定**: MMR with α=0.4, λ=0.7 が最高性能を示しています。

### 埋め込み可視化の特徴

- **高速な次元圧縮**: UMAPによる効率的な可視化
- **インタラクティブ性**: クリック選択とリアルタイム更新
- **ユーザー類似度検証**: 視覚的距離と実際の類似度の対応確認
- **Windows対応**: セッション状態管理によるクロスプラットフォーム互換性

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
   # results/evaluation_results.csvが存在し、正しい形式か確認
   head results/evaluation_results.csv
   ```

4. **Windows環境でのUI問題**
   ```bash
   # セッション状態の問題の場合、ブラウザのキャッシュをクリア
   # または別のブラウザで試行
   ```

5. **ハッシュIDの不整合**
   ```bash
   # データを再生成して整合性を確保
   python data_generator.py
   # 古いモデルファイルを削除
   rm -rf weights/*
   ```

### パフォーマンス最適化

- 大規模データセット（>100万レコード）の場合：
  - バッチ処理の実装を検討
  - スパースマトリックスの最適化
  - メモリ効率的なデータ分割

- 埋め込み可視化の最適化：
  - t-SNEのperplexityを調整（5-50）
  - UMAPのn_neighbors, min_distanceパラメータ調整
  - キャッシュ機能によるリアルタイム操作

## 主な新機能（v2.0）

### ハッシュID対応
- SHA256ベースの8文字ハッシュID
- プライバシー保護とデータセキュリティの向上
- 既存システムとの互換性維持

### 埋め込み可視化システム
- アーティストとユーザーの2D可視化
- t-SNE/UMAP次元圧縮サポート
- インタラクティブなクリック選択
- リアルタイムユーザー比較機能

### ユーザーベース推薦
- ALS user_factorsベースの類似度計算
- コサイン類似度による高精度推薦
- 推薦理由の詳細表示

### Windows対応強化
- セッション状態管理の最適化
- クロスプラットフォーム互換性の確保
- UIウィジェットの安定性向上

詳細な技術文書や追加機能については、各モジュールのdocstringまたは`CLAUDE.md`を参照してください。