import streamlit as st
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import umap
import os
import sys
from typing import Dict, List, Tuple, Optional
import pickle

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from models.matrix_factorization.music_recommender_mmr import MusicRecommenderMMR

# ページ設定
st.set_page_config(
    page_title="Artist Embedding Visualizer",
    page_icon="🎨",
    layout="wide"
)

@st.cache_resource
def load_model_and_data(csv_path: str, alpha: float):
    """モデルとデータを読み込み"""
    recommender = MusicRecommenderMMR(csv_path=csv_path)
    
    # モデルファイルパスを生成
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    model_filename = f'{csv_basename}_mmr_alpha_{alpha:.1f}.pkl'
    model_path = os.path.join(parent_dir, 'weights', model_filename)
    
    # モデルが存在する場合は読み込み、そうでなければ訓練
    try:
        recommender.load_model(model_path)
    except FileNotFoundError:
        with st.spinner(f"モデルを訓練中です (α={alpha})..."):
            recommender.load_data()
            recommender.prepare_data()
            recommender.train_model(alpha=alpha)
            recommender.save_model(model_path)
    
    return recommender

@st.cache_data
def get_artist_genre_mapping(df: pl.DataFrame) -> Dict[str, str]:
    """アーティストとジャンルのマッピングを取得"""
    # アーティストとジャンルは一対一で紐づくので、各アーティストの最初のジャンルを取得
    # ソートして順序を保証
    artist_genre = df.group_by('artist').agg(
        pl.col('genre').first().alias('genre')
    ).sort('artist')
    
    return dict(zip(artist_genre['artist'].to_list(), artist_genre['genre'].to_list()))

@st.cache_data
def compute_tsne_embeddings(item_factors: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """t-SNEによる2次元埋め込みを計算"""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=0
    )
    return tsne.fit_transform(item_factors)

@st.cache_data
def compute_umap_embeddings(
    item_factors: np.ndarray, 
    n_neighbors: int = 15, 
    min_dist: float = 0.1, 
    random_state: int = 42
) -> np.ndarray:
    """UMAPによる2次元埋め込みを計算"""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False
    )
    return reducer.fit_transform(item_factors)

def create_interactive_plot(
    embeddings: np.ndarray,
    artist_names: List[str],
    genres: List[str],
    title: str,
    debug_mode: bool = False
) -> go.Figure:
    """インタラクティブな散布図を作成"""
    
    # データフレームを作成
    df_plot = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'artist': artist_names,
        'genre': genres,
        'index': range(len(artist_names))  # デバッグ用インデックス
    })
    
    # デバッグモード時の詳細情報出力
    if debug_mode:
        st.write("🔍 **プロット作成デバッグ情報**")
        st.write(f"Embeddings shape: {embeddings.shape}")
        st.write(f"Artist names length: {len(artist_names)}")
        st.write(f"Genres length: {len(genres)}")
        
        # 最初の5個の詳細情報
        st.write("**最初の5個の詳細情報:**")
        for i in range(min(5, len(artist_names))):
            st.write(f"Index {i}: {artist_names[i]} → {genres[i]} → Embedding({embeddings[i, 0]:.3f}, {embeddings[i, 1]:.3f})")
        
        # DataFrameの作成前後での順序確認
        st.write("**DataFrame作成前の順序確認:**")
        st.write("Artist names[:5]:", artist_names[:5])
        st.write("Genres[:5]:", genres[:5])
        
        # DataFrame作成後の順序確認
        st.write("**DataFrame作成後の順序確認:**")
        st.write("df_plot.head():")
        st.dataframe(df_plot.head(), use_container_width=True)
    
    # Plotlyの内部ソートを防ぐため、categoricalにして順序を固定
    df_plot['genre'] = pd.Categorical(df_plot['genre'], categories=df_plot['genre'].unique(), ordered=True)
    
    # ジャンルごとの色分け
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='genre',
        hover_data=['artist', 'index'],
        title=title,
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        width=800,
        height=600,
        category_orders={'genre': df_plot['genre'].cat.categories.tolist()}  # 順序を明示的に指定
    )
    
    # デバッグ情報: Plotly作成後の確認
    if debug_mode:
        st.write("**Plotly作成後のトレース確認:**")
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'name') and hasattr(trace, 'x') and len(trace.x) > 0:
                st.write(f"Trace {i} (Genre: {trace.name}): First point at ({trace.x[0]:.3f}, {trace.y[0]:.3f})")
                # 対応するDataFrameの行を探す
                matching_rows = df_plot[(df_plot['x'] == trace.x[0]) & (df_plot['y'] == trace.y[0])]
                if len(matching_rows) > 0:
                    row = matching_rows.iloc[0]
                    st.write(f"  → DataFrame: {row['artist']} (idx: {row['index']})")
                else:
                    st.write(f"  → DataFrame: No matching row found")
    
    # ホバー情報をカスタマイズ（インデックスも表示）
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b> (idx: %{customdata[1]})<br>' +
                      'Genre: %{marker.color}<br>' +
                      'X: %{x:.3f}<br>' +
                      'Y: %{y:.3f}<extra></extra>',
        customdata=df_plot[['artist', 'index']].values
    )
    
    # レイアウトを調整
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        margin=dict(l=0, r=150, t=50, b=0)
    )
    
    return fig

def main():
    st.title("🎨 Artist Embedding Visualizer")
    st.markdown("**ALSアイテムファクターの2次元可視化 - ジャンル別色分け**")
    
    # セッション状態の初期化（クリック選択機能）
    if 'plot_selected_artist' not in st.session_state:
        st.session_state.plot_selected_artist = None
    
    # サイドバー設定
    st.sidebar.header("設定")
    
    # データ設定
    default_csv_path = os.path.join(parent_dir, "data", "user_artist_plays.csv")
    csv_path = st.sidebar.text_input("CSVファイルパス", value=default_csv_path)
    alpha = st.sidebar.slider("Alpha値", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
    
    # デバッグモード設定
    debug_mode = st.sidebar.checkbox("プロット作成デバッグモード", value=False)
    
    # 次元圧縮手法の選択
    reduction_method = st.sidebar.selectbox(
        "次元圧縮手法",
        ["t-SNE", "UMAP", "両方"]
    )
    
    # t-SNEパラメータ
    if reduction_method in ["t-SNE", "両方"]:
        st.sidebar.subheader("t-SNEパラメータ")
        tsne_perplexity = st.sidebar.slider("Perplexity", min_value=5.0, max_value=50.0, value=30.0, step=5.0)
    
    # UMAPパラメータ
    if reduction_method in ["UMAP", "両方"]:
        st.sidebar.subheader("UMAPパラメータ")
        umap_n_neighbors = st.sidebar.slider("N Neighbors", min_value=5, max_value=50, value=15, step=5)
        umap_min_dist = st.sidebar.slider("Min Distance", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    # データ読み込み
    try:
        if not os.path.exists(csv_path):
            st.error(f"CSVファイルが見つかりません: {csv_path}")
            return
            
        # モデル読み込み
        with st.spinner("モデルを読み込み中..."):
            recommender = load_model_and_data(csv_path, alpha)
        
        if not recommender.is_trained:
            st.error("モデルが訓練されていません")
            return
            
        # データ情報表示
        st.subheader("📊 データ情報")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("アーティスト数", len(recommender.idx_to_artist))
        with col2:
            st.metric("潜在因子数", recommender.model.item_factors.shape[1])
        with col3:
            st.metric("Alpha値", alpha)
        
        # アーティスト-ジャンルマッピングを取得
        with st.spinner("ジャンル情報を処理中..."):
            artist_genre_mapping = get_artist_genre_mapping(recommender.df)
        
        # アーティスト名とジャンルのリストを作成
        artist_names = []
        genres = []
        for idx in range(len(recommender.idx_to_artist)):
            artist = recommender.idx_to_artist[idx]
            genre = artist_genre_mapping.get(artist, "Unknown")
            artist_names.append(artist)
            genres.append(genre)
        
        # デバッグ情報: 詳細な対応関係を検証
        if st.sidebar.checkbox("デバッグ情報を表示", value=False):
            st.sidebar.subheader("🔍 詳細デバッグ情報")
            
            # 基本情報
            st.sidebar.write(f"🔢 Item factors shape: {recommender.model.item_factors.shape}")
            st.sidebar.write(f"📊 Artist count: {len(artist_names)}")
            st.sidebar.write(f"🎵 Genre count: {len(set(genres))}")
            
            # アーティスト-ジャンル対応 (最初の10個)
            st.sidebar.subheader("📋 idx_to_artist → genre mapping")
            debug_data = []
            for i in range(min(10, len(artist_names))):
                debug_data.append({
                    "Index": i,
                    "Artist": artist_names[i],
                    "Genre": genres[i]
                })
            st.sidebar.dataframe(pd.DataFrame(debug_data), use_container_width=True, hide_index=True)
            
            # CSVから直接取得したアーティスト-ジャンルマッピング
            st.sidebar.subheader("📁 CSV直接マッピング (ソート済み)")
            csv_mapping_data = []
            sorted_mapping = sorted(artist_genre_mapping.items())[:10]
            for i, (artist, genre) in enumerate(sorted_mapping):
                csv_mapping_data.append({
                    "CSV_Order": i,
                    "Artist": artist,
                    "Genre": genre
                })
            st.sidebar.dataframe(pd.DataFrame(csv_mapping_data), use_container_width=True, hide_index=True)
            
            # 実際のCSVでの最初の出現順序
            st.sidebar.subheader("📄 CSV最初出現順序")
            csv_first_appearance = recommender.df.select(['artist', 'genre']).unique(maintain_order=True)
            csv_appear_data = []
            for i in range(min(10, len(csv_first_appearance))):
                row = csv_first_appearance.row(i)
                csv_appear_data.append({
                    "CSV_Index": i,
                    "Artist": row[0],
                    "Genre": row[1]
                })
            st.sidebar.dataframe(pd.DataFrame(csv_appear_data), use_container_width=True, hide_index=True)
            
            # artist_to_idxの検証
            st.sidebar.subheader("🔗 artist_to_idx検証")
            artist_to_idx_data = []
            for artist in list(recommender.artist_to_idx.keys())[:10]:
                idx = recommender.artist_to_idx[artist]
                reverse_artist = recommender.idx_to_artist[idx]
                artist_to_idx_data.append({
                    "Artist": artist,
                    "→ Index": idx,
                    "Index → Artist": reverse_artist,
                    "Match": "✓" if artist == reverse_artist else "✗"
                })
            st.sidebar.dataframe(pd.DataFrame(artist_to_idx_data), use_container_width=True, hide_index=True)
        
        # ジャンル統計表示
        st.subheader("🎵 ジャンル分布")
        genre_counts = pd.Series(genres).value_counts()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_genre = px.bar(
                x=genre_counts.index,
                y=genre_counts.values,
                labels={'x': 'ジャンル', 'y': 'アーティスト数'},
                title="ジャンル別アーティスト数"
            )
            fig_genre.update_xaxes(tickangle=45)
            st.plotly_chart(fig_genre, use_container_width=True)
        
        with col2:
            st.dataframe(
                pd.DataFrame({
                    'ジャンル': genre_counts.index,
                    'アーティスト数': genre_counts.values
                }),
                use_container_width=True,
                hide_index=True
            )
        
        # 次元圧縮と可視化
        item_factors = recommender.model.item_factors
        
        if reduction_method == "t-SNE":
            st.subheader("🔬 t-SNE可視化")
            with st.spinner("t-SNE計算中..."):
                tsne_embeddings = compute_tsne_embeddings(
                    item_factors, 
                    perplexity=tsne_perplexity
                )
            
            fig_tsne = create_interactive_plot(
                tsne_embeddings,
                artist_names,
                genres,
                f"t-SNE Artist Embeddings (perplexity={tsne_perplexity})",
                debug_mode=debug_mode
            )
            st.plotly_chart(fig_tsne, use_container_width=True)
            
        elif reduction_method == "UMAP":
            st.subheader("🗺️ UMAP可視化")
            with st.spinner("UMAP計算中..."):
                umap_embeddings = compute_umap_embeddings(
                    item_factors,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist
                )
            
            fig_umap = create_interactive_plot(
                umap_embeddings,
                artist_names,
                genres,
                f"UMAP Artist Embeddings (neighbors={umap_n_neighbors}, min_dist={umap_min_dist})",
                debug_mode=debug_mode
            )
            st.plotly_chart(fig_umap, use_container_width=True)
            
        else:  # 両方
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔬 t-SNE可視化")
                with st.spinner("t-SNE計算中..."):
                    tsne_embeddings = compute_tsne_embeddings(
                        item_factors, 
                        perplexity=tsne_perplexity
                    )
                
                fig_tsne = create_interactive_plot(
                    tsne_embeddings,
                    artist_names,
                    genres,
                    f"t-SNE (perplexity={tsne_perplexity})",
                    debug_mode=debug_mode
                )
                st.plotly_chart(fig_tsne, use_container_width=True)
            
            with col2:
                st.subheader("🗺️ UMAP可視化")
                with st.spinner("UMAP計算中..."):
                    umap_embeddings = compute_umap_embeddings(
                        item_factors,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist
                    )
                
                fig_umap = create_interactive_plot(
                    umap_embeddings,
                    artist_names,
                    genres,
                    f"UMAP (neighbors={umap_n_neighbors})",
                    debug_mode=debug_mode
                )
                st.plotly_chart(fig_umap, use_container_width=True)
        
        # 手法の説明
        st.subheader("📚 手法について")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
            - 非線形次元圧縮手法
            - 局所的な近傍関係を保持
            - クラスタ構造の可視化に優れる
            - Perplexity: 近傍点数の調整パラメータ
            """)
        
        with col2:
            st.markdown("""
            **UMAP (Uniform Manifold Approximation and Projection)**
            - 高速な非線形次元圧縮手法
            - 大域的・局所的構造の両方を保持
            - より安定した結果
            - N Neighbors: 近傍数、Min Distance: 点間最小距離
            """)
        
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()