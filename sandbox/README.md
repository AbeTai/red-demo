# Sandbox - Artist Embedding Visualizer

このディレクトリには、ALSモデルのアイテムファクター（アーティスト埋め込み）を2次元で可視化するツールが含まれています。

## 概要

`artist_embedding_visualizer.py`は、音楽推薦システムで学習されたアーティストの潜在表現を可視化するStreamlitアプリケーションです。

## 機能

### 1. 次元圧縮手法
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
  - 非線形次元圧縮
  - 局所的な近傍関係を保持
  - クラスタ構造の可視化に優れる
  - パラメータ: Perplexity (5-50)

- **UMAP (Uniform Manifold Approximation and Projection)**
  - 高速な非線形次元圧縮
  - 大域的・局所的構造の両方を保持
  - より安定した結果
  - パラメータ: N Neighbors (5-50), Min Distance (0.01-1.0)

### 2. 可視化機能
- **ジャンル別色分け**: 各アーティストをジャンルに応じて色分け表示
- **インタラクティブプロット**: アーティスト名や詳細情報をホバーで表示
- **ジャンル分布**: アーティスト数の分布をバープロットで表示
- **並列比較**: t-SNEとUMAPを同時に表示可能

### 3. データ処理
- ALSモデルの`item_factors`を使用
- アーティストごとの主要ジャンルを自動決定
- 効率的なキャッシュ機能

## 使用方法

### 起動
```bash
cd sandbox
uv run streamlit run artist_embedding_visualizer.py
```

### 設定項目
- **CSVファイルパス**: データファイルの場所
- **Alpha値**: ALSモデルの信頼度パラメータ
- **次元圧縮手法**: t-SNE、UMAP、または両方
- **パラメータ調整**: 各手法の詳細パラメータ

## 技術詳細

### 依存関係
- streamlit
- numpy
- pandas
- polars
- plotly
- scikit-learn (t-SNE)
- umap-learn (UMAP)

### アルゴリズム
1. **ジャンル決定**: 各アーティストで最も多く登場するジャンルを主要ジャンルとして設定
2. **埋め込み抽出**: ALSモデルの`item_factors`から各アーティストの潜在表現を取得
3. **次元圧縮**: t-SNEまたはUMAPで2次元に圧縮
4. **可視化**: Plotlyでインタラクティブな散布図を生成

### パフォーマンス最適化
- `@st.cache_data`でt-SNE/UMAP計算結果をキャッシュ
- `@st.cache_resource`でモデル読み込みをキャッシュ
- 効率的なジャンル分析

## 出力例

可視化により以下のような洞察が得られます：
- 同じジャンルのアーティストが近くにクラスタリングされる
- アーティスト間の音楽的類似性が距離として表現される
- 異なるジャンル間の関係性が可視化される

## ファイル構成

```
sandbox/
├── artist_embedding_visualizer.py  # メインアプリケーション
└── README.md                       # このファイル
```

## 注意事項

- 初回実行時は、モデルが存在しない場合は自動で訓練が行われます
- 大量のアーティストがある場合、t-SNE/UMAPの計算に時間がかかることがあります
- ジャンル情報が不完全な場合、一部のアーティストが"Unknown"として表示されます