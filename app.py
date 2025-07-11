import streamlit as st
import polars as pl
import os
from models.matrix_factorization.music_recommender_mmr import MusicRecommenderMMR

# コールバック関数の定義
def on_artist_selection_change():
    """アーティスト選択変更時のコールバック関数"""
    # ウィジェットから現在の選択を取得
    if 'artist_multiselect' in st.session_state:
        new_selection = st.session_state.artist_multiselect
        # セッション状態を更新
        st.session_state.selected_artists = list(new_selection) if new_selection else []
        # 関連する状態をリセット
        st.session_state.selected_user_id = None
        st.session_state.matching_users = []
        st.session_state.show_user_selection = False

def on_search_button_click():
    """検索ボタンクリック時のコールバック関数"""
    st.session_state.search_triggered = True

# ページ設定
st.set_page_config(
    page_title="Music Recommender Demo (Integrated)",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_recommender(alpha, csv_path):
    """レコメンダーを読み込み（キャッシュ）"""
    recommender = MusicRecommenderMMR(csv_path=csv_path)
    
    # モデルファイルパスを生成
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    model_filename = f'{csv_basename}_mmr_alpha_{alpha:.1f}.pkl'
    model_path = os.path.join('weights', model_filename)
    
    # モデルが存在する場合は読み込み
    try:
        recommender.load_model(model_path)
    except FileNotFoundError:
        # モデルが存在しない場合は訓練
        with st.spinner(f"モデルを訓練中です (α={alpha})..."):
            recommender.load_data()
            recommender.prepare_data()
            recommender.train_model(alpha=alpha)
            recommender.save_model(model_path)
    
    return recommender

def get_user_ids(df):
    """DataFrameから実際のユーザーID一覧を取得"""
    return df['user_id'].unique().to_list()

def get_unique_artists(df):
    """DataFrameからユニークなアーティスト一覧を取得"""
    return df['artist'].unique().to_list()

def get_users_by_artists(df, selected_artists):
    """選択されたアーティスト全てを聴いているユーザーIDを取得"""
    if not selected_artists:
        return []
    
    # アーティスト名の正規化（前後の空白を除去）
    normalized_selected = [artist.strip() for artist in selected_artists]
    
    # 選択されたアーティストを聴いているユーザーを取得
    filtered_df = df.filter(pl.col('artist').is_in(normalized_selected))
    
    # ユーザーIDでグループ化して、選択したアーティスト数と一致するユーザーを抽出
    user_artist_counts = filtered_df.group_by('user_id').agg(pl.col('artist').n_unique().alias('artist_count'))
    users_with_all_artists = user_artist_counts.filter(pl.col('artist_count') == len(selected_artists))['user_id'].to_list()
    
    return users_with_all_artists

def get_users_by_artists_and_demographics(df, selected_artists, selected_gender=None, age_range=None):
    """選択されたアーティスト全てを聴いているユーザーIDを取得（性別・年齢フィルタ付き）"""
    if not selected_artists:
        return []
    
    # アーティスト名の正規化（前後の空白を除去）
    normalized_selected = [artist.strip() for artist in selected_artists]
    
    # 選択されたアーティストを聴いているユーザーを取得
    filtered_df = df.filter(pl.col('artist').is_in(normalized_selected))
    
    # ユーザーIDでグループ化して、選択したアーティスト数と一致するユーザーを抽出
    user_artist_counts = filtered_df.group_by('user_id').agg(pl.col('artist').n_unique().alias('artist_count'))
    users_with_all_artists = user_artist_counts.filter(pl.col('artist_count') == len(selected_artists))['user_id'].to_list()
    
    # 性別・年齢フィルタが指定されている場合は適用
    if selected_gender or age_range:
        # ユーザーごとの性別・年齢情報を取得（最初の行を使用）
        user_demographics = df.group_by('user_id').agg([
            pl.col('gender').first().alias('gender'),
            pl.col('age').first().alias('age')
        ])
        
        # 性別フィルタを適用
        if selected_gender:
            gender_filtered_users = user_demographics.filter(pl.col('gender') == selected_gender)['user_id'].to_list()
            users_with_all_artists = [user for user in users_with_all_artists if user in gender_filtered_users]
        
        # 年齢カテゴリフィルタを適用
        if age_range:
            age_filtered_users = user_demographics.filter(
                pl.col('age') == age_range
            )['user_id'].to_list()
            users_with_all_artists = [user for user in users_with_all_artists if user in age_filtered_users]
    
    return users_with_all_artists

def get_demographics_info(df):
    """性別・年齢カテゴリの情報を取得"""
    user_demographics = df.group_by('user_id').agg([
        pl.col('gender').first().alias('gender'),
        pl.col('age').first().alias('age')
    ])
    
    unique_genders = user_demographics['gender'].unique().to_list()
    unique_age_categories = user_demographics['age'].unique().to_list()
    
    return unique_genders, unique_age_categories

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
    st.title("🎵 Music Recommender Demo (Integrated)")
    st.markdown("**統合版音楽推薦システム - MMR、人口統計学フィルタリング対応**")
    
    # セッション状態の初期化（防御的な処理・Windows/Mac環境対応）
    session_defaults = {
        'selected_artists': [],
        'selected_user_id': None,
        'get_recommendations': False,
        'matching_users': [],
        'show_user_selection': False,
        'search_triggered': False
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state or st.session_state[key] is None:
            st.session_state[key] = default_value
    
    # リストの型チェック（Windows環境での安定性向上）
    if not isinstance(st.session_state.selected_artists, list):
        st.session_state.selected_artists = []
    if not isinstance(st.session_state.matching_users, list):
        st.session_state.matching_users = []
    
    # CSVファイル選択
    csv_path = st.sidebar.text_input("CSVファイルパス", value="data/user_artist_plays.csv")
    
    
    # CSVファイルを読み込み
    try:
        if not os.path.exists(csv_path):
            st.sidebar.error(f"CSVファイルが見つかりません: {csv_path}")
            return
        df = pl.read_csv(csv_path)
        st.sidebar.success(f"データ読み込み成功: {len(df)}件")
    except Exception as e:
        st.sidebar.error(f"CSVファイルエラー: {e}")
        return
    
    # ユーザーID一覧を取得
    user_ids = get_user_ids(df)
    
    # サイドバー設定
    st.sidebar.header("アルゴリズム設定")
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
        recommender = load_recommender(alpha, csv_path)
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return
    
    # ユーザー検索方法の選択
    st.subheader("🔍 ユーザー検索")
    search_method = st.radio(
        "検索方法を選択してください:",
        ["ID直接入力", "アーティスト指定検索"],
        horizontal=True,
        key="search_method_radio"
    )
    
    user_id = None
    get_recommendations = False
    
    # 人口統計学データの存在確認（共通）
    has_demographics = 'gender' in df.columns and 'age' in df.columns
    selected_gender = None
    age_range = None
    
    if search_method == "ID直接入力":
        # ID直接入力
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.selectbox(
                "ユーザーID", 
                user_ids,
                help="レコメンドを取得したいユーザーIDを選択してください",
                key="direct_user_selector"
            )
        
        with col2:
            get_recommendations = st.button("レコメンドを取得", 
                                           type="primary", 
                                           key="get_recommendations_button")
            
    else:
        # アーティスト指定による検索
        st.markdown("**アーティストを選択して、そのアーティスト全てを聴いているユーザーから選択してください**")
        
        # アーティスト選択（コールバック関数使用）
        artists = get_unique_artists(df)
        
        # アーティスト選択の状態を表示
        if st.session_state.selected_artists:
            st.info(f"📝 現在選択中: {', '.join(st.session_state.selected_artists[:3])}{'...' if len(st.session_state.selected_artists) > 3 else ''} ({len(st.session_state.selected_artists)}個)")
        
        # アーティスト選択ウィジェット（コールバック付き）
        selected_artists = st.multiselect(
            "アーティストを選択:",
            artists,
            default=st.session_state.selected_artists,
            max_selections=10,
            help="選択したアーティスト全てを聴いているユーザーが表示されます",
            key="artist_multiselect",
            on_change=on_artist_selection_change
        )
        
        # 性別・年齢フィルタ（任意）
        selected_gender = None
        age_range = None
        
        if has_demographics:
            st.markdown("**追加フィルタ（任意）**")
            col1, col2 = st.columns(2)
            
            with col1:
                # 性別フィルタ
                try:
                    unique_genders, unique_age_categories = get_demographics_info(df)
                    selected_gender = st.selectbox(
                        "性別で絞り込み（任意）:",
                        ["すべて"] + unique_genders,
                        help="特定の性別のユーザーのみに絞り込みます",
                        key="gender_filter"
                    )
                    if selected_gender == "すべて":
                        selected_gender = None
                except Exception as e:
                    st.warning("性別情報を取得できませんでした。")
                    selected_gender = None
            
            with col2:
                # 年齢カテゴリフィルタ
                try:
                    selected_age_category = st.selectbox(
                        "年齢で絞り込み（任意）:",
                        ["すべて"] + unique_age_categories,
                        help="指定した年齢カテゴリのユーザーのみに絞り込みます",
                        key="age_filter"
                    )
                    if selected_age_category == "すべて":
                        age_range = None
                    else:
                        age_range = selected_age_category
                except Exception as e:
                    st.warning("年齢情報を取得できませんでした。")
                    age_range = None
        
        # 検索ボタンとリセットボタン
        col1, col2 = st.columns([3, 1])
        with col1:
            search_button = st.button("🔍 ユーザーを検索", 
                                    type="primary", 
                                    key="search_users_button",
                                    on_click=on_search_button_click,
                                    disabled=len(st.session_state.selected_artists) == 0)
        with col2:
            if st.button("🔄 リセット", key="reset_button"):
                st.session_state.selected_artists = []
                st.session_state.selected_user_id = None
                st.session_state.matching_users = []
                st.session_state.show_user_selection = False
                st.session_state.search_triggered = False
                st.rerun()
        
        # 検索が実行された場合
        if st.session_state.search_triggered:
            # 検索フラグをリセット
            st.session_state.search_triggered = False
            
            # 現在のアーティスト選択を取得
            current_artists = st.session_state.selected_artists
            
            # 検索実行の表示
            st.success(f"🔍 検索実行: {len(current_artists)}個のアーティストを選択")
            if current_artists:
                with st.expander("選択されたアーティスト", expanded=False):
                    for i, artist in enumerate(current_artists, 1):
                        st.write(f"{i}. {artist}")
            
            if current_artists and len(current_artists) > 0:
                try:
                    # 該当ユーザーを取得
                    if has_demographics:
                        matching_users = get_users_by_artists_and_demographics(
                            df, current_artists, selected_gender, age_range
                        )
                    else:
                        matching_users = get_users_by_artists(df, current_artists)
                    
                    # 防御的な処理: matching_usersが有効かチェック
                    if matching_users is None:
                        matching_users = []
                    
                    # マッチングユーザーをセッション状態に保存
                    st.session_state.matching_users = matching_users
                    st.session_state.show_user_selection = len(matching_users) > 0
                    
                    if len(matching_users) > 0:
                        st.session_state.selected_user_id = matching_users[0]  # デフォルト選択
                        st.success(f"✅ {len(matching_users)}人のユーザーが見つかりました")
                    else:
                        st.session_state.selected_user_id = None
                        st.warning("⚠️ 指定した条件に該当するユーザーが見つかりませんでした")
                        
                except Exception as e:
                    st.error(f"❌ ユーザー検索中にエラーが発生しました: {str(e)}")
                    st.session_state.matching_users = []
                    st.session_state.show_user_selection = False
                    st.session_state.selected_user_id = None
            else:
                st.session_state.matching_users = []
                st.session_state.show_user_selection = False
                st.info("ℹ️ アーティストを選択してください")
        
        # セッション状態に基づいてユーザー選択を表示（防御的な処理）
        if (st.session_state.show_user_selection and 
            st.session_state.matching_users and 
            len(st.session_state.matching_users) > 0):
            
            st.markdown("---")
            st.markdown("### 👤 ユーザー選択")
            
            # デバッグ情報（Windows環境確認用）
            st.write(f"🔍 表示中: {len(st.session_state.matching_users)}人のユーザー")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                try:
                    # デフォルト値の設定（防御的な処理）
                    default_index = 0
                    if (st.session_state.selected_user_id and 
                        st.session_state.selected_user_id in st.session_state.matching_users):
                        try:
                            default_index = st.session_state.matching_users.index(st.session_state.selected_user_id)
                        except ValueError:
                            default_index = 0
                    
                    user_id = st.selectbox(
                        "ユーザーIDを選択:",
                        st.session_state.matching_users,
                        index=default_index,
                        help="選択したアーティスト全てを聴いているユーザーから選択してください",
                        key="user_selector"
                    )
                    
                    # 選択されたユーザーIDをセッション状態に保存
                    if user_id != st.session_state.selected_user_id:
                        st.session_state.selected_user_id = user_id
                        st.write(f"✅ ユーザー {user_id} を選択しました")
                
                except Exception as e:
                    st.error(f"❌ ユーザー選択でエラーが発生しました: {str(e)}")
                    get_recommendations = False
            
            with col2:
                get_recommendations = st.button("レコメンドを取得", 
                                               type="primary", 
                                               key="get_recommendations_from_search")
        else:
            get_recommendations = False
    
    # データセット情報
    with st.expander("データセット情報"):
        st.write(f"- **CSVファイル**: {csv_path}")
        st.write(f"- **ユーザー数**: {len(user_ids):,}人")
        st.write(f"- **アーティスト数**: {len(get_unique_artists(df))}組")
        st.write(f"- **再生記録数**: {len(df):,}件")
        if has_demographics:
            st.write("- **人口統計学データ**: 性別・年齢カテゴリ利用可能")
        st.write("- **アルゴリズム**: Implicit ALS (Alternating Least Squares)")
        st.write(f"- **信頼度関数**: 1 + {alpha} × 再生回数")
        st.write("- **MMR**: Maximal Marginal Relevance による多様性考慮リランキング")
    
    if get_recommendations:
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
            
            # ユーザーの性別・年齢情報を表示（データにある場合）
            if has_demographics:
                try:
                    user_info = df.filter(pl.col('user_id') == user_id).select(['gender', 'age']).row(0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("性別", user_info[0])
                    with col2:
                        st.metric("年齢", user_info[1])
                except:
                    pass
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
    st.markdown("Built with Streamlit and Implicit ALS - Integrated Version with Full Features")

if __name__ == "__main__":
    main()