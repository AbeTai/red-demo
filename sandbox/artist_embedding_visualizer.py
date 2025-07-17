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
    artist_genre = df.group_by('artist').agg(
        pl.col('genre').first().alias('genre')
    )
    
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
    title: str
) -> go.Figure:
    """インタラクティブな散布図を作成"""
    
    # データフレームを作成
    df_plot = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'artist': artist_names,
        'genre': genres
    })
    
    # ジャンルごとの色分け
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='genre',
        hover_data=['artist'],
        title=title,
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        width=800,
        height=600
    )
    
    # ホバー情報をカスタマイズ
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      'Genre: %{marker.color}<br>' +
                      'X: %{x:.3f}<br>' +
                      'Y: %{y:.3f}<extra></extra>',
        customdata=df_plot[['artist']].values
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
    
    # サイドバー設定
    st.sidebar.header("設定")
    
    # データ設定
    default_csv_path = os.path.join(parent_dir, "data", "user_artist_plays.csv")
    csv_path = st.sidebar.text_input("CSVファイルパス", value=default_csv_path)
    alpha = st.sidebar.slider("Alpha値", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
    
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
                f"t-SNE Artist Embeddings (perplexity={tsne_perplexity})"
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
                f"UMAP Artist Embeddings (neighbors={umap_n_neighbors}, min_dist={umap_min_dist})"
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
                    f"t-SNE (perplexity={tsne_perplexity})"
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
                    f"UMAP (neighbors={umap_n_neighbors})"
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