import streamlit as st
import polars as pl
from recommender_mmr import MusicRecommenderMMR

# ページ設定
st.set_page_config(
    page_title="Music Recommender Demo with MMR",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_recommender(alpha):
    """レコメンダーを読み込み（キャッシュ）"""
    recommender = MusicRecommenderMMR()
    
    # モデルが存在しない場合は訓練
    if not recommender.load_model(alpha=alpha):
        with st.spinner(f"モデルを訓練中です (α={alpha})..."):
            recommender.load_data()
            recommender.prepare_data()
            recommender.train_model(alpha=alpha)
            recommender.save_model(alpha=alpha)
    
    return recommender

def get_unique_artists():
    """CSVファイルからユニークなアーティスト一覧を取得"""
    df = pl.read_csv('user_artist_plays.csv')
    return sorted(df['artist'].unique().to_list())

def get_users_by_artists(selected_artists):
    """選択されたアーティスト全てを聴いているユーザーIDを取得"""
    if not selected_artists:
        return []
    
    df = pl.read_csv('user_artist_plays.csv')
    
    # 選択されたアーティストを聴いているユーザーを取得
    filtered_df = df.filter(pl.col('artist').is_in(selected_artists))
    
    # ユーザーIDでグループ化して、選択したアーティスト数と一致するユーザーを抽出
    user_artist_counts = filtered_df.group_by('user_id').agg(pl.col('artist').n_unique().alias('artist_count'))
    users_with_all_artists = user_artist_counts.filter(pl.col('artist_count') == len(selected_artists))['user_id'].to_list()
    
    return sorted(users_with_all_artists)

def display_recommendations(recommendations, title, description=""):
    """推薦結果を表示"""
    st.subheader(title)
    if description:
        st.markdown(description)
    
    if isinstance(recommendations, str):
        st.error(recommendations)
    elif recommendations:
        for i, (artist, score) in enumerate(recommendations, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {artist}**")
                with col2:
                    st.write(f"スコア: {score:.3f}")
                st.divider()
    else:
        st.info("レコメンドできるアーティストがありません。")

def main():
    st.title("🎵 Music Recommender Demo with MMR")
    st.markdown("**ユーザーIDを入力して、おすすめのアーティストを見つけよう！**")
    
    # サイドバー
    st.sidebar.header("設定")
    n_recommendations = st.sidebar.slider("レコメンド数", 1, 10, 5)
    alpha = st.sidebar.slider(
        "Alpha値 (信頼度パラメータ)", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.4, 
        step=0.1,
        help="再生回数に対する重み付け。大きいほど再生回数の多いアイテムを重視"
    )
    
    # MMRパラメータ
    st.sidebar.header("MMR設定")
    lambda_param = st.sidebar.slider(
        "Lambda値 (関連性と多様性のバランス)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0=多様性重視、1=関連性重視"
    )
    candidate_pool_size = st.sidebar.slider(
        "候補プールサイズ",
        min_value=10,
        max_value=50,
        value=20,
        help="MMR前の候補数（推薦数より多く設定）"
    )
    
    # レコメンダーを読み込み
    try:
        recommender = load_recommender(alpha)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        return
    
    # ユーザー検索方法の選択
    st.subheader("🔍 ユーザー検索")
    search_method = st.radio(
        "検索方法を選択してください:",
        ["ID直接入力", "アーティスト指定検索"],
        horizontal=True
    )
    
    user_id = None
    
    if search_method == "ID直接入力":
        # 従来のID直接入力
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.number_input(
                "ユーザーID", 
                min_value=1, 
                max_value=1000, 
                value=1,
                help="1から1000までのユーザーIDを入力してください"
            )
        
        with col2:
            get_recommendations = st.button("レコメンドを取得", type="primary")
            
    else:
        # アーティスト指定による検索
        st.markdown("**アーティストを選択して、そのアーティスト全てを聴いているユーザーから選択してください（最大10つまで）**")
        
        # アーティスト選択
        artists = get_unique_artists()
        selected_artists = st.multiselect(
            "アーティストを選択（最大10アーティスト）:",
            artists,
            max_selections=10,
            help="選択したアーティスト全てを聴いているユーザーが表示されます"
        )
        
        if selected_artists:
            # 該当ユーザーを取得
            matching_users = get_users_by_artists(selected_artists)
            
            if matching_users:
                st.info(f"選択したアーティスト全てを聴いている{len(matching_users)}人のユーザーが見つかりました")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    user_id = st.selectbox(
                        "ユーザーIDを選択:",
                        matching_users,
                        help="選択したアーティスト全てを聴いているユーザーから選択してください"
                    )
                
                with col2:
                    get_recommendations = st.button("レコメンドを取得", type="primary")
            else:
                st.warning("選択したアーティスト全てを聴いているユーザーが見つかりませんでした")
                get_recommendations = False
        else:
            st.info("アーティストを選択してください")
            get_recommendations = False
    
    # データセット情報
    with st.expander("データセット情報"):
        st.write("- **ユーザー数**: 1,000人")
        st.write("- **アーティスト数**: 20組")
        st.write("- **再生記録数**: 9,913件")
        st.write("- **再生回数範囲**: 1-500回")
        st.write("- **アルゴリズム**: Implicit ALS (Alternating Least Squares)")
        st.write(f"- **信頼度関数**: 1 + {alpha} × 再生回数")
        st.write("- **MMR**: Maximal Marginal Relevance による多様性考慮リランキング")
    
    if get_recommendations or user_id:
        # ユーザーの履歴を表示
        st.subheader(f"👤 User {user_id} の再生履歴")
        
        history = recommender.get_user_history(user_id)
        if isinstance(history, str):
            st.error(history)
            return
        
        if history:
            history_df = pl.DataFrame(history, schema=["アーティスト", "再生回数"], orient="row")
            
            # 再生履歴をテーブルで表示
            st.dataframe(
                history_df.to_pandas(),
                use_container_width=True,
                hide_index=True
            )
            
            # 再生の統計
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("聴いたアーティスト数", len(history_df))
            with col2:
                st.metric("平均再生回数", f"{history_df['再生回数'].mean():.1f}")
            with col3:
                st.metric("総再生回数", f"{history_df['再生回数'].sum():,}")
        else:
            st.info("このユーザーには再生履歴がありません。")
            return
        
        # 推薦結果を並列で表示
        col1, col2 = st.columns(2)
        
        with col1:
            # 標準推薦
            with st.spinner("標準レコメンド生成中..."):
                standard_recommendations = recommender.get_recommendations(
                    user_id, 
                    n_recommendations, 
                    use_mmr=False
                )
            
            display_recommendations(
                standard_recommendations,
                "🎯 標準おすすめアーティスト",
                "*関連性スコア順*"
            )
        
        with col2:
            # MMR推薦
            with st.spinner("MMRレコメンド生成中..."):
                mmr_recommendations = recommender.get_recommendations(
                    user_id, 
                    n_recommendations, 
                    use_mmr=True,
                    lambda_param=lambda_param,
                    candidate_pool_size=candidate_pool_size
                )
            
            display_recommendations(
                mmr_recommendations,
                "🌟 MMRおすすめアーティスト",
                f"*関連性と多様性のバランス (λ={lambda_param})*"
            )
        
        # MMR設定の説明
        st.info(
            f"**MMR設定**: λ={lambda_param} (候補プール: {candidate_pool_size}件)\n\n"
            f"- λ=0.0: 完全に多様性重視\n"
            f"- λ=0.5: 関連性と多様性のバランス\n"
            f"- λ=1.0: 完全に関連性重視"
        )
    
    # フッター
    st.markdown("---")
    st.markdown("Built with Streamlit and Implicit ALS - Enhanced with MMR (Maximal Marginal Relevance)")

if __name__ == "__main__":
    main()