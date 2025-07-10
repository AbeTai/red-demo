import pandas as pd
import numpy as np

def generate_sample_data():
    np.random.seed(42)
    
    # アーティストリスト
    artists = [
        "Taylor Swift", "Ed Sheeran", "Ariana Grande", "Drake", "Billie Eilish",
        "The Weeknd", "Olivia Rodrigo", "Harry Styles", "Dua Lipa", "Post Malone",
        "BTS", "Justin Bieber", "Adele", "Bad Bunny", "Doja Cat",
        "Bruno Mars", "The Beatles", "Queen", "Eminem", "Kendrick Lamar"
    ]
    
    # ユーザー数とアーティスト数
    n_users = 1000
    n_artists = len(artists)
    
    # 年齢カテゴリを定義（5歳刻み）
    age_categories = [
        "15-19", "20-24", "25-29", "30-34", "35-39", 
        "40-44", "45-49", "50-54", "55-59", "60-64", "65-70"
    ]
    
    # ユーザーごとの属性情報を生成
    user_demographics = {}
    for user_id in range(1, n_users + 1):
        # 性別を生成（Male, Female, Other）
        gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04])
        
        # 年齢カテゴリを生成（各カテゴリに均等に近い分布）
        age_category = np.random.choice(age_categories)
        
        user_demographics[user_id] = {'gender': gender, 'age': age_category}
    
    # ユーザーIDとアーティストIDのペアを生成
    user_ids = []
    artist_names = []
    play_counts = []
    genders = []
    ages = []
    
    # 各ユーザーが5-15個のアーティストを聴く
    for user_id in range(1, n_users + 1):
        n_artists_listened = np.random.randint(5, 16)  # 5-15個のアーティストを聴く
        selected_artists = np.random.choice(artists, size=n_artists_listened, replace=False)
        
        # ユーザーの属性情報を取得
        user_gender = user_demographics[user_id]['gender']
        user_age_category = user_demographics[user_id]['age']
        
        for artist in selected_artists:
            # 再生回数は対数正規分布を使って現実的な分布を作成
            # 平均再生回数は10回程度、最大500回程度
            log_plays = np.random.lognormal(mean=2.0, sigma=1.0)  # 対数正規分布
            play_count = int(np.clip(log_plays, 1, 500))  # 1-500回の範囲
            
            user_ids.append(user_id)
            artist_names.append(artist)
            play_counts.append(play_count)
            genders.append(user_gender)
            ages.append(user_age_category)
    
    # データフレーム作成
    df = pd.DataFrame({
        'user_id': user_ids,
        'artist': artist_names,
        'play_count': play_counts,
        'gender': genders,
        'age': ages
    })
    
    # CSVで保存
    df.to_csv('user_artist_plays.csv', index=False)
    print(f"Generated {len(df)} play records for {n_users} users and {n_artists} artists")
    print(f"Average plays per user-artist: {df['play_count'].mean():.1f}")
    print(f"Total plays: {df['play_count'].sum():,}")
    
    # 属性情報の統計を表示
    print("\nDemographics Summary:")
    print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
    print("Age category distribution:")
    for category, count in df['age'].value_counts().items():
        print(f"  {category}: {count} records")
    
    return df

if __name__ == "__main__":
    generate_sample_data()