import streamlit as st
import pandas as pd
from recommender import MusicRecommender

# ページ設定
st.set_page_config(
    page_title="Music Recommender Demo",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    """レコメンダーを読み込み（キャッシュ）"""
    recommender = MusicRecommender()
    
    # モデルが存在しない場合は訓練
    if not recommender.load_model():
        with st.spinner("モデルを訓練中です..."):
            recommender.load_data()
            recommender.prepare_data()
            recommender.train_model()
            recommender.save_model()
    
    return recommender

def main():
    st.title("🎵 Music Recommender Demo")
    st.markdown("**ユーザーIDを入力して、おすすめのアーティストを見つけよう！**")
    
    # サイドバー
    st.sidebar.header("設定")
    n_recommendations = st.sidebar.slider("レコメンド数", 1, 10, 5)
    
    # レコメンダーを読み込み
    try:
        recommender = load_recommender()
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        return
    
    # ユーザー入力
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
    
    # データセット情報
    with st.expander("データセット情報"):
        st.write("- **ユーザー数**: 1,000人")
        st.write("- **アーティスト数**: 20組")
        st.write("- **再生記録数**: 9,913件")
        st.write("- **再生回数範囲**: 1-500回")
        st.write("- **アルゴリズム**: Implicit ALS (Alternating Least Squares)")
        st.write("- **信頼度関数**: 1 + 0.4 × 再生回数")
    
    if get_recommendations or user_id:
        # ユーザーの履歴を表示
        st.subheader(f"👤 User {user_id} の再生履歴")
        
        history = recommender.get_user_history(user_id)
        if isinstance(history, str):
            st.error(history)
            return
        
        if history:
            history_df = pd.DataFrame(history, columns=["アーティスト", "再生回数"])
            
            # 再生履歴をテーブルで表示
            st.dataframe(
                history_df,
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
        
        # レコメンドを取得して表示
        st.subheader("🎯 おすすめアーティスト")
        
        with st.spinner("レコメンド生成中..."):
            recommendations = recommender.get_recommendations(user_id, n_recommendations)
        
        if isinstance(recommendations, str):
            st.error(recommendations)
        elif recommendations:
            # レコメンドをカードスタイルで表示
            for i, (artist, score) in enumerate(recommendations, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{i}. {artist}**")
                    with col2:
                        st.write(f"スコア: {score:.3f}")
                    st.divider()
        else:
            st.info("このユーザーにはレコメンドできるアーティストがありません。")
    
    # フッター
    st.markdown("---")
    st.markdown("Built with Streamlit and Implicit ALS")

if __name__ == "__main__":
    main()