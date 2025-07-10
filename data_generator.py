import pandas as pd
import numpy as np
import datetime
from typing import Dict, List

def generate_sample_data() -> pd.DataFrame:
    """
    音楽推薦システム用のサンプルデータを生成
    
    Returns:
        pd.DataFrame: ユーザー-アーティスト再生データ
    """
    np.random.seed(42)
    
    # アーティストリストとジャンルのマッピング
    artists_genres: Dict[str, str] = {
        "Taylor Swift": "Pop",
        "Ed Sheeran": "Pop",
        "Ariana Grande": "Pop",
        "Drake": "Hip-Hop",
        "Billie Eilish": "Pop",
        "The Weeknd": "R&B",
        "Olivia Rodrigo": "Pop",
        "Harry Styles": "Pop",
        "Dua Lipa": "Pop",
        "Post Malone": "Hip-Hop",
        "BTS": "K-Pop",
        "Justin Bieber": "Pop",
        "Adele": "Pop",
        "Bad Bunny": "Reggaeton",
        "Doja Cat": "Hip-Hop",
        "Bruno Mars": "Pop",
        "The Beatles": "Rock",
        "Queen": "Rock",
        "Eminem": "Hip-Hop",
        "Kendrick Lamar": "Hip-Hop"
    }
    
    artists: List[str] = list(artists_genres.keys())
    
    # データ生成パラメータ
    n_users: int = 1000
    n_artists: int = len(artists)
    
    # 年齢カテゴリを定義（5歳刻み）
    age_categories: List[str] = [
        "15-19", "20-24", "25-29", "30-34", "35-39", 
        "40-44", "45-49", "50-54", "55-59", "60-64", "65-70"
    ]
    
    # ユーザーごとの人口統計学的属性を生成
    user_demographics: Dict[int, Dict[str, str]] = {}
    for user_id in range(1, n_users + 1):
        # 性別を生成（男性、女性、その他の分布）
        gender: str = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04])
        
        # 年齢カテゴリを生成（各カテゴリに均等に近い分布）
        age_category: str = np.random.choice(age_categories)
        
        user_demographics[user_id] = {'gender': gender, 'age': age_category}
    
    # 再生記録データのリストを初期化
    user_ids: List[int] = []
    artist_names: List[str] = []
    play_counts: List[int] = []
    genders: List[str] = []
    ages: List[str] = []
    interaction_dates: List[int] = []
    genres: List[str] = []
    
    # インタラクション日付の範囲を設定（2020年1月1日から2024年12月31日）
    start_date: datetime.date = datetime.date(2020, 1, 1)
    end_date: datetime.date = datetime.date(2024, 12, 31)
    
    # 各ユーザーが聴くアーティスト数とその再生記録を生成
    for user_id in range(1, n_users + 1):
        # 各ユーザーが聴くアーティスト数（5-15個）
        n_artists_listened: int = np.random.randint(5, 16)
        selected_artists: np.ndarray = np.random.choice(artists, size=n_artists_listened, replace=False)
        
        # ユーザーの人口統計学的属性を取得
        user_gender: str = user_demographics[user_id]['gender']
        user_age_category: str = user_demographics[user_id]['age']
        
        for artist in selected_artists:
            # 再生回数を対数正規分布で生成（現実的な分布を模擬）
            # 平均10回程度、最大500回の範囲
            log_plays: float = np.random.lognormal(mean=2.0, sigma=1.0)
            play_count: int = int(np.clip(log_plays, 1, 500))
            
            # インタラクション日付を生成（YYYYMMDD形式）
            random_days: int = np.random.randint(0, (end_date - start_date).days)
            interaction_date_obj: datetime.date = start_date + datetime.timedelta(days=random_days)
            interaction_date: int = int(interaction_date_obj.strftime('%Y%m%d'))
            
            # データをリストに追加
            user_ids.append(user_id)
            artist_names.append(artist)
            play_counts.append(play_count)
            genders.append(user_gender)
            ages.append(user_age_category)
            interaction_dates.append(interaction_date)
            genres.append(artists_genres[artist])
    
    # データフレームを作成
    df: pd.DataFrame = pd.DataFrame({
        'user_id': user_ids,
        'artist': artist_names,
        'play_count': play_counts,
        'gender': genders,
        'age': ages,
        'interaction_date': interaction_dates,
        'genre': genres
    })
    
    # CSVファイルに保存
    df.to_csv('user_artist_plays.csv', index=False)
    print(f"{n_users}ユーザーと{n_artists}アーティストで{len(df)}件の再生記録を生成しました")
    print(f"ユーザー-アーティストあたりの平均再生回数: {df['play_count'].mean():.1f}")
    print(f"総再生回数: {df['play_count'].sum():,}")
    
    # 人口統計学的属性の統計を表示
    print("\n人口統計学データ概要:")
    print(f"性別分布: {df['gender'].value_counts().to_dict()}")
    print("年齢カテゴリ分布:")
    for category, count in df['age'].value_counts().items():
        print(f"  {category}: {count} records")
    
    print("\nジャンル分布:")
    for genre, count in df['genre'].value_counts().items():
        print(f"  {genre}: {count} records")
    
    return df

if __name__ == "__main__":
    generate_sample_data()