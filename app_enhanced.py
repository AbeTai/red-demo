import streamlit as st
import pandas as pd
from recommender import MusicRecommender

# ページ設定
st.set_page_config(
    page_title="Music Recommender Demo (Enhanced)",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_recommender(alpha):
    """レコメンダーを読み込み（キャッシュ）"""
    recommender = MusicRecommender()
    
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
    df = pd.read_csv('user_artist_plays.csv')
    return sorted(df['artist'].unique())

def get_users_by_artists_and_demographics(selected_artists, selected_gender=None, age_range=None):
    """選択されたアーティスト全てを聴いているユーザーIDを取得（性別・年齢フィルタ付き）"""
    if not selected_artists:
        return []
    
    df = pd.read_csv('user_artist_plays.csv')
    
    # 選択されたアーティストを聴いているユーザーを取得
    filtered_df = df[df['artist'].isin(selected_artists)]
    
    # ユーザーIDでグループ化して、選択したアーティスト数と一致するユーザーを抽出
    user_artist_counts = filtered_df.groupby('user_id')['artist'].nunique()
    users_with_all_artists = user_artist_counts[user_artist_counts == len(selected_artists)].index.tolist()
    
    # 性別・年齢フィルタが指定されている場合は適用
    if selected_gender or age_range:
        # ユーザーごとの性別・年齢情報を取得（最初の行を使用）
        user_demographics = df.groupby('user_id')[['gender', 'age']].first()
        
        # 性別フィルタを適用
        if selected_gender:
            gender_filtered_users = user_demographics[user_demographics['gender'] == selected_gender].index.tolist()
            users_with_all_artists = [user for user in users_with_all_artists if user in gender_filtered_users]
        
        # 年齢フィルタを適用
        if age_range:
            min_age, max_age = age_range
            age_filtered_users = user_demographics[
                (user_demographics['age'] >= min_age) & 
                (user_demographics['age'] <= max_age)
            ].index.tolist()
            users_with_all_artists = [user for user in users_with_all_artists if user in age_filtered_users]
    
    return sorted(users_with_all_artists)

def get_demographics_info():
    """性別・年齢の情報を取得"""
    df = pd.read_csv('user_artist_plays.csv')
    user_demographics = df.groupby('user_id')[['gender', 'age']].first()
    
    unique_genders = sorted(user_demographics['gender'].unique())
    min_age = int(user_demographics['age'].min())
    max_age = int(user_demographics['age'].max())
    
    return unique_genders, min_age, max_age

def main():
    st.title("🎵 Music Recommender Demo (Enhanced)")
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
        
        # 性別・年齢フィルタ（任意）
        st.markdown("**追加フィルタ（任意）**")
        col1, col2 = st.columns(2)
        
        with col1:
            # 性別フィルタ
            try:
                unique_genders, min_age, max_age = get_demographics_info()
                selected_gender = st.selectbox(
                    "性別で絞り込み（任意）:",
                    ["すべて"] + unique_genders,
                    help="特定の性別のユーザーのみに絞り込みます"
                )
                if selected_gender == "すべて":
                    selected_gender = None
            except Exception as e:
                st.warning("性別情報を取得できませんでした。性別フィルタは利用できません。")
                selected_gender = None
        
        with col2:
            # 年齢フィルタ
            try:
                age_filter_enabled = st.checkbox("年齢で絞り込み")
                if age_filter_enabled:
                    age_range = st.slider(
                        "年齢範囲:",
                        min_value=min_age,
                        max_value=max_age,
                        value=(min_age, max_age),
                        help="指定した年齢範囲のユーザーのみに絞り込みます"
                    )
                else:
                    age_range = None
            except Exception as e:
                st.warning("年齢情報を取得できませんでした。年齢フィルタは利用できません。")
                age_range = None
        
        if selected_artists:
            # 該当ユーザーを取得
            matching_users = get_users_by_artists_and_demographics(
                selected_artists, 
                selected_gender, 
                age_range
            )
            
            if matching_users:
                # フィルタ情報を表示
                filter_info = []
                if selected_gender:
                    filter_info.append(f"性別: {selected_gender}")
                if age_range:
                    filter_info.append(f"年齢: {age_range[0]}-{age_range[1]}歳")
                
                if filter_info:
                    filter_text = "（" + ", ".join(filter_info) + "）"
                else:
                    filter_text = ""
                
                st.info(f"選択したアーティスト全てを聴いている{len(matching_users)}人のユーザーが見つかりました{filter_text}")
                
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
                st.warning("指定した条件に該当するユーザーが見つかりませんでした")
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
        st.write("- **新機能**: 性別・年齢による絞り込み検索")
    
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
            
            # ユーザーの性別・年齢情報を表示（データにある場合）
            try:
                df = pd.read_csv('user_artist_plays.csv')
                user_info = df[df['user_id'] == user_id][['gender', 'age']].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("性別", user_info['gender'])
                with col2:
                    st.metric("年齢", f"{user_info['age']}歳")
            except:
                pass  # 性別・年齢情報がない場合は表示しない
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
    st.markdown("Built with Streamlit and Implicit ALS - Enhanced with Demographics Filtering")

if __name__ == "__main__":
    main()