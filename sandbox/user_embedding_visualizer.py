import streamlit as st
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
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
    page_title="User Embedding Visualizer",
    page_icon="👥",
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
def compute_tsne_embeddings(user_factors: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """t-SNEによる2次元埋め込みを計算"""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=0
    )
    return tsne.fit_transform(user_factors)

@st.cache_data
def compute_umap_embeddings(
    user_factors: np.ndarray, 
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
    return reducer.fit_transform(user_factors)

def create_user_interactive_plot(
    embeddings: np.ndarray,
    user_ids: List,
    title: str,
    selected_user_id: Optional[str] = None
) -> go.Figure:
    """ユーザー用インタラクティブな散布図を作成（クリック選択機能付き）"""
    
    # データフレームを作成
    df_plot = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'user_id': user_ids
    })
    
    # 選択されたユーザーとその他を分離
    if selected_user_id and selected_user_id in user_ids:
        df_selected = df_plot[df_plot['user_id'] == selected_user_id]
        df_others = df_plot[df_plot['user_id'] != selected_user_id]
    else:
        df_selected = pd.DataFrame(columns=['x', 'y', 'user_id'])
        df_others = df_plot
    
    fig = go.Figure()
    
    # 通常の点を追加
    if len(df_others) > 0:
        fig.add_trace(go.Scatter(
            x=df_others['x'],
            y=df_others['y'],
            mode='markers',
            marker=dict(
                size=6,
                color='lightblue',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            customdata=df_others[['user_id']].values,
            hovertemplate='<b>User ID: %{customdata[0]}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         '<i>クリックして選択</i><extra></extra>',
            name='Users',
            showlegend=False
        ))
    
    # 選択された点を追加（ハイライト表示）
    if len(df_selected) > 0:
        fig.add_trace(go.Scatter(
            x=df_selected['x'],
            y=df_selected['y'],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                opacity=1.0,
                line=dict(width=3, color='darkred'),
                symbol='circle'
            ),
            customdata=df_selected[['user_id']].values,
            hovertemplate='<b>🎯 選択中: %{customdata[0]}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         '<i>選択されています</i><extra></extra>',
            name='Selected User',
            showlegend=False
        ))
    
    # レイアウトを調整
    fig.update_layout(
        title=title,
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        width=800,
        height=600,
        showlegend=False,
        margin=dict(l=0, r=50, t=50, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # グリッド線を追加
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def get_user_listening_history(recommender: MusicRecommenderMMR, user_id) -> pd.DataFrame:
    """ユーザーの視聴履歴を取得"""
    try:
        user_data = recommender.df.filter(pl.col('user_id') == user_id)
        if len(user_data) == 0:
            return pd.DataFrame()
        
        # 再生回数順にソート
        user_data = user_data.sort('play_count', descending=True)
        
        return user_data.select(['artist', 'genre', 'play_count']).to_pandas()
    except:
        return pd.DataFrame()

def display_user_comparison(recommender: MusicRecommenderMMR, user_id1, user_id2):
    """2人のユーザーの視聴履歴を比較表示"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"👤 User {user_id1} の視聴履歴")
        history1 = get_user_listening_history(recommender, user_id1)
        
        if len(history1) > 0:
            # 統計情報
            total_plays1 = history1['play_count'].sum()
            unique_artists1 = len(history1)
            unique_genres1 = history1['genre'].nunique()
            
            st.metric("総再生回数", f"{total_plays1:,}")
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("アーティスト数", unique_artists1)
            with col1b:
                st.metric("ジャンル数", unique_genres1)
            
            # 視聴履歴テーブル
            st.dataframe(
                history1.head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # ジャンル分布
            genre_counts1 = history1.groupby('genre')['play_count'].sum().sort_values(ascending=False)
            st.bar_chart(genre_counts1.head(10))
            
        else:
            st.error(f"User {user_id1} が見つかりません")
    
    with col2:
        st.subheader(f"👤 User {user_id2} の視聴履歴")
        history2 = get_user_listening_history(recommender, user_id2)
        
        if len(history2) > 0:
            # 統計情報
            total_plays2 = history2['play_count'].sum()
            unique_artists2 = len(history2)
            unique_genres2 = history2['genre'].nunique()
            
            st.metric("総再生回数", f"{total_plays2:,}")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("アーティスト数", unique_artists2)
            with col2b:
                st.metric("ジャンル数", unique_genres2)
            
            # 視聴履歴テーブル
            st.dataframe(
                history2.head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # ジャンル分布
            genre_counts2 = history2.groupby('genre')['play_count'].sum().sort_values(ascending=False)
            st.bar_chart(genre_counts2.head(10))
            
        else:
            st.error(f"User {user_id2} が見つかりません")
    
    # 共通性分析
    if len(history1) > 0 and len(history2) > 0:
        st.subheader("🔍 共通性分析")
        
        # 共通アーティスト
        common_artists = set(history1['artist']) & set(history2['artist'])
        common_genres = set(history1['genre']) & set(history2['genre'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("共通アーティスト数", len(common_artists))
        with col2:
            st.metric("共通ジャンル数", len(common_genres))
        with col3:
            # ユーザーベクトルのコサイン類似度計算
            try:
                user_idx1 = recommender.user_to_idx[user_id1]
                user_idx2 = recommender.user_to_idx[user_id2]
                
                user_vector1 = recommender.model.user_factors[user_idx1].reshape(1, -1)
                user_vector2 = recommender.model.user_factors[user_idx2].reshape(1, -1)
                
                cosine_sim = cosine_similarity(user_vector1, user_vector2)[0][0]
                st.metric("ユーザー類似度 (コサイン)", f"{cosine_sim:.3f}")
            except Exception as e:
                st.metric("ユーザー類似度 (コサイン)", "計算エラー")
        
        if common_artists:
            st.markdown("**共通アーティスト:**")
            common_list = list(common_artists)[:10]  # 上位10個
            st.write(", ".join(common_list))

# コールバック関数の定義
def on_user_id1_change():
    """ユーザーID1選択変更時のコールバック関数"""
    if 'user_id1_input' in st.session_state:
        st.session_state.selected_user_id1 = st.session_state.user_id1_input

def on_user_id2_change():
    """ユーザーID2選択変更時のコールバック関数"""
    if 'user_id2_input' in st.session_state:
        st.session_state.selected_user_id2 = st.session_state.user_id2_input

def main():
    st.title("👥 User Embedding Visualizer")
    st.markdown("**ALSユーザーファクターの2次元可視化**")
    
    # セッション状態の初期化（Windows環境対応・文字列ID対応・クリック選択機能）
    session_defaults = {
        'selected_user_id1': None,
        'selected_user_id2': None,
        'plot_selected_user_id': None
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
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
            st.metric("ユーザー数", len(recommender.idx_to_user))
        with col2:
            st.metric("潜在因子数", recommender.model.user_factors.shape[1])
        with col3:
            st.metric("Alpha値", alpha)
        
        # ユーザーIDのリストを作成
        user_ids = []
        for idx in range(len(recommender.idx_to_user)):
            user_id = recommender.idx_to_user[idx]
            user_ids.append(user_id)
        
        # 次元圧縮と可視化（クリック選択機能付き）
        user_factors = recommender.model.user_factors
        
        # 選択されたユーザーID表示エリア
        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
        with col_info1:
            if st.session_state.plot_selected_user_id:
                st.success(f"🎯 **選択中のユーザー**: `{st.session_state.plot_selected_user_id}`")
                st.info("💡 このIDをコピーして比較分析で使用できます")
            else:
                st.info("📌 **散布図の点をクリックまたは手動入力でユーザーを選択**")
        
        with col_info2:
            # 手動入力によるユーザー選択
            manual_user_id = st.selectbox(
                "手動選択",
                ["選択なし"] + user_ids[:50],  # 最初の50ユーザーのみ表示
                key="manual_user_select",
                help="手動でユーザーを選択"
            )
            if manual_user_id != "選択なし" and manual_user_id != st.session_state.plot_selected_user_id:
                if st.button("✅ この選択を使用", key="apply_manual_selection"):
                    st.session_state.plot_selected_user_id = manual_user_id
                    st.rerun()
        
        with col_info3:
            if st.session_state.plot_selected_user_id:
                # コピー用のテキストボックス
                st.text_input(
                    "コピー用",
                    value=st.session_state.plot_selected_user_id,
                    key="copy_user_id",
                    help="このIDをコピーして他の機能で使用できます"
                )
                if st.button("🗑️ 選択をクリア", key="clear_selection"):
                    st.session_state.plot_selected_user_id = None
                    st.rerun()
        
        st.markdown("---")
        
        if reduction_method == "t-SNE":
            st.subheader("🔬 t-SNE可視化")
            with st.spinner("t-SNE計算中..."):
                tsne_embeddings = compute_tsne_embeddings(
                    user_factors, 
                    perplexity=tsne_perplexity
                )
            
            fig_tsne = create_user_interactive_plot(
                tsne_embeddings,
                user_ids,
                f"t-SNE User Embeddings (perplexity={tsne_perplexity})",
                selected_user_id=st.session_state.plot_selected_user_id
            )
            
            # Streamlitのクリックイベント処理（修正版）
            st.plotly_chart(fig_tsne, use_container_width=True, key="tsne_plot", on_select="rerun")
            
            # プロット下にクリック情報を表示
            st.markdown("**💡 ヒント**: 散布図の点をクリックしてユーザーを選択できます")
            
            # クリック情報の取得とデバッグ表示
            if st.checkbox("クリック情報を表示", key="show_click_info"):
                # セッション状態からクリック情報を取得
                tsne_selection = st.session_state.get("tsne_plot", {})
                
                # 選択情報がある場合の処理
                if tsne_selection and isinstance(tsne_selection, dict):
                    if 'selection' in tsne_selection:
                        selection = tsne_selection['selection']
                        
                        if selection and 'points' in selection and selection['points']:
                            points = selection['points']
                            st.write(f"選択された点の数: {len(points)}")
                            
                            for i, point in enumerate(points):
                                if 'customdata' in point and point['customdata']:
                                    clicked_id = point['customdata'][0]
                                    st.write(f"📌 クリックされたユーザーID: **{clicked_id}**")
                                    if st.button(f"🎯 {clicked_id} を選択", key=f"select_from_click_{i}"):
                                        st.session_state.plot_selected_user_id = clicked_id
                                        st.rerun()
                                else:
                                    st.write(f"点 {i+1}: customdata情報なし")
                    else:
                        st.info("まだ点がクリックされていません")
                else:
                    st.info("クリック情報がありません")
            
        elif reduction_method == "UMAP":
            st.subheader("🗺️ UMAP可視化")
            with st.spinner("UMAP計算中..."):
                umap_embeddings = compute_umap_embeddings(
                    user_factors,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist
                )
            
            fig_umap = create_user_interactive_plot(
                umap_embeddings,
                user_ids,
                f"UMAP User Embeddings (neighbors={umap_n_neighbors}, min_dist={umap_min_dist})",
                selected_user_id=st.session_state.plot_selected_user_id
            )
            
            # Streamlitのクリックイベント処理（修正版）
            st.plotly_chart(fig_umap, use_container_width=True, key="umap_plot", on_select="rerun")
            
            # プロット下にクリック情報を表示
            st.markdown("**💡 ヒント**: 散布図の点をクリックしてユーザーを選択できます")
            
            # クリック情報の取得とデバッグ表示
            if st.checkbox("クリック情報を表示", key="show_click_info_umap"):
                # セッション状態からクリック情報を取得
                umap_selection = st.session_state.get("umap_plot", {})
                
                # 選択情報がある場合の処理
                if umap_selection and isinstance(umap_selection, dict):
                    if 'selection' in umap_selection:
                        selection = umap_selection['selection']
                        
                        if selection and 'points' in selection and selection['points']:
                            points = selection['points']
                            st.write(f"選択された点の数: {len(points)}")
                            
                            for i, point in enumerate(points):
                                if 'customdata' in point and point['customdata']:
                                    clicked_id = point['customdata'][0]
                                    st.write(f"📌 クリックされたユーザーID: **{clicked_id}**")
                                    if st.button(f"🎯 {clicked_id} を選択", key=f"select_from_click_umap_{i}"):
                                        st.session_state.plot_selected_user_id = clicked_id
                                        st.rerun()
                                else:
                                    st.write(f"点 {i+1}: customdata情報なし")
                    else:
                        st.info("まだ点がクリックされていません")
                else:
                    st.info("クリック情報がありません")
            
        else:  # 両方
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔬 t-SNE可視化")
                with st.spinner("t-SNE計算中..."):
                    tsne_embeddings = compute_tsne_embeddings(
                        user_factors, 
                        perplexity=tsne_perplexity
                    )
                
                fig_tsne = create_user_interactive_plot(
                    tsne_embeddings,
                    user_ids,
                    "t-SNE User Embeddings",
                    selected_user_id=st.session_state.plot_selected_user_id
                )
                
                st.plotly_chart(fig_tsne, use_container_width=True, key="tsne_plot_dual")
            
            with col2:
                st.subheader("🗺️ UMAP可視化")
                with st.spinner("UMAP計算中..."):
                    umap_embeddings = compute_umap_embeddings(
                        user_factors,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist
                    )
                
                fig_umap = create_user_interactive_plot(
                    umap_embeddings,
                    user_ids,
                    "UMAP User Embeddings",
                    selected_user_id=st.session_state.plot_selected_user_id
                )
                
                st.plotly_chart(fig_umap, use_container_width=True, key="umap_plot_dual")
            
            st.markdown("**💡 ヒント**: 手動選択またはドロップダウンメニューでユーザーを選択してください")
        
        # ユーザー比較セクション
        st.markdown("---")
        st.subheader("🔍 ユーザー比較分析")
        st.markdown("**グラフ上で近くにいるユーザーの視聴履歴が本当に似ているか検証してみましょう**")
        
        # 選択したユーザーを比較に使用するボタン
        if st.session_state.plot_selected_user_id:
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button(f"📍 選択ユーザーをユーザー1に設定", key="use_selected_as_user1"):
                    st.session_state.selected_user_id1 = st.session_state.plot_selected_user_id
                    st.rerun()
            with col_btn2:
                if st.button(f"📍 選択ユーザーをユーザー2に設定", key="use_selected_as_user2"):
                    st.session_state.selected_user_id2 = st.session_state.plot_selected_user_id
                    st.rerun()
            with col_btn3:
                st.write(f"選択中: `{st.session_state.plot_selected_user_id}`")
        
        # ユーザーID入力（Windows環境対応のコールバック使用・文字列ID対応）
        col1, col2 = st.columns(2)
        
        with col1:
            # セッション状態の初期化・デフォルト値の設定
            if st.session_state.selected_user_id1 is None and user_ids:
                st.session_state.selected_user_id1 = user_ids[0]
            elif st.session_state.selected_user_id1 not in user_ids and user_ids:
                st.session_state.selected_user_id1 = user_ids[0]
            
            # デフォルトインデックスの計算
            default_index1 = 0
            if st.session_state.selected_user_id1 in user_ids:
                default_index1 = user_ids.index(st.session_state.selected_user_id1)
            
            user_id1 = st.selectbox(
                "ユーザーID 1",
                user_ids,
                index=default_index1,
                key="user_id1_input",
                on_change=on_user_id1_change
            )
        
        with col2:
            # セッション状態の初期化・デフォルト値の設定
            if st.session_state.selected_user_id2 is None and len(user_ids) > 1:
                st.session_state.selected_user_id2 = user_ids[1]
            elif st.session_state.selected_user_id2 not in user_ids and len(user_ids) > 1:
                st.session_state.selected_user_id2 = user_ids[1]
            elif st.session_state.selected_user_id2 not in user_ids and user_ids:
                st.session_state.selected_user_id2 = user_ids[0]
            
            # デフォルトインデックスの計算
            default_index2 = min(1, len(user_ids) - 1) if len(user_ids) > 1 else 0
            if st.session_state.selected_user_id2 in user_ids:
                default_index2 = user_ids.index(st.session_state.selected_user_id2)
            
            user_id2 = st.selectbox(
                "ユーザーID 2", 
                user_ids,
                index=default_index2,
                key="user_id2_input",
                on_change=on_user_id2_change
            )
        
        # 比較実行ボタン
        if st.button("📊 ユーザー比較を実行", type="primary"):
            # セッション状態の値を使用
            final_user_id1 = st.session_state.selected_user_id1
            final_user_id2 = st.session_state.selected_user_id2
            
            if final_user_id1 != final_user_id2:
                display_user_comparison(recommender, final_user_id1, final_user_id2)
            else:
                st.warning("異なるユーザーIDを入力してください")
        
        # 手法の説明
        st.subheader("📚 手法について")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **ユーザー埋め込み可視化**
            - ALSのuser_factorsを2次元に圧縮
            - 近くにいるユーザーは音楽的嗜好が類似
            - ユーザークラスタの形成パターンを分析
            - 埋め込み空間の構造を直感的に理解
            
            **🎯 クリック選択機能**
            - 散布図の点をクリックしてユーザーを選択
            - 選択されたユーザーは赤色でハイライト
            - IDをコピーして他の機能で利用可能
            """)
        
        with col2:
            st.markdown("""
            **類似性検証**
            - グラフ上の距離と実際の嗜好類似性を比較
            - 共通アーティスト・ジャンル分析
            - ジャカード係数による類似度計算
            - 推薦システムの妥当性検証
            
            **📋 便利機能**
            - 選択したユーザーを比較分析に自動セット
            - ユーザーIDコピー用テキストボックス
            - 選択クリアボタンで簡単リセット
            """)
        
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()