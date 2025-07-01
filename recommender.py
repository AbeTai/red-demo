import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle
import os

class MusicRecommender:
    def __init__(self):
        self.model = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.artist_to_idx = {}
        self.idx_to_artist = {}
        self.user_item_matrix = None
        
    def load_data(self, csv_path='user_artist_plays.csv'):
        """CSVファイルからデータを読み込み"""
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} play records")
        
    def prepare_data(self):
        """データを前処理してスパース行列を作成"""
        # ユーザーとアーティストのマッピングを作成
        unique_users = self.df['user_id'].unique()
        unique_artists = self.df['artist'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
        self.idx_to_artist = {idx: artist for artist, idx in self.artist_to_idx.items()}
        
        # スパース行列用のデータを準備
        user_indices = [self.user_to_idx[user] for user in self.df['user_id']]
        artist_indices = [self.artist_to_idx[artist] for artist in self.df['artist']]
        play_counts = self.df['play_count'].values
        
        # ユーザー-アイテム行列を作成
        self.user_item_matrix = csr_matrix(
            (play_counts, (user_indices, artist_indices)),
            shape=(len(unique_users), len(unique_artists))
        )
        
        print(f"Created matrix with {len(unique_users)} users and {len(unique_artists)} artists")
        
    def train_model(self, factors=50, regularization=0.1, iterations=20):
        """ALSモデルを訓練"""
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        
        # implicitは confidence matrix を期待する
        # 再生回数をそのまま信頼度として使用（1 + alpha * play_count の形式）
        confidence_matrix = self.user_item_matrix.copy()
        confidence_matrix.data = 1 + confidence_matrix.data * 0.4  # alpha=0.4で信頼度を計算
        
        print("Training ALS model...")
        self.model.fit(confidence_matrix.T)  # アイテム×ユーザー行列として渡す
        print("Training completed!")
        
    def get_recommendations(self, user_id, n_recommendations=5):
        """指定したユーザーにレコメンドを提供"""
        if user_id not in self.user_to_idx:
            return f"User {user_id} not found in the dataset"
            
        user_idx = self.user_to_idx[user_id]
        
        # ユーザーが既に聴いたアーティストを取得
        user_items = set(self.df[self.df['user_id'] == user_id]['artist'].values)
        
        # レコメンドを取得（協調フィルタリングによる手動実装）
        user_vector = self.user_item_matrix[user_idx].toarray().flatten()
        
        # ユーザーファクターとアイテムファクターの内積でスコア計算
        if user_idx >= self.model.user_factors.shape[0]:
            # ユーザーが訓練データにない場合は、平均的なレコメンドを返す
            item_scores = np.mean(self.model.item_factors, axis=1)
        else:
            item_scores = self.model.item_factors.dot(self.model.user_factors[user_idx])
        
        # スコアの高い順にソート
        top_items = np.argsort(item_scores)[::-1]
        
        recommendations = []
        for item_idx in top_items:
            if item_idx < len(self.idx_to_artist):
                artist = self.idx_to_artist[item_idx]
                if artist not in user_items:
                    recommendations.append((artist, float(item_scores[item_idx])))
                if len(recommendations) >= n_recommendations:
                    break
                
        return recommendations
    
    def get_user_history(self, user_id):
        """ユーザーの再生履歴を取得"""
        if user_id not in self.user_to_idx:
            return f"User {user_id} not found in the dataset"
            
        user_data = self.df[self.df['user_id'] == user_id].sort_values('play_count', ascending=False)
        return user_data[['artist', 'play_count']].values.tolist()
    
    def save_model(self, path='recommender_model.pkl'):
        """モデルを保存"""
        model_data = {
            'model': self.model,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'artist_to_idx': self.artist_to_idx,
            'idx_to_artist': self.idx_to_artist,
            'user_item_matrix': self.user_item_matrix,
            'df': self.df
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path='recommender_model.pkl'):
        """モデルを読み込み"""
        if not os.path.exists(path):
            return False
            
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.user_to_idx = model_data['user_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.artist_to_idx = model_data['artist_to_idx']
        self.idx_to_artist = model_data['idx_to_artist']
        self.user_item_matrix = model_data['user_item_matrix']
        self.df = model_data['df']
        
        print(f"Model loaded from {path}")
        return True

def main():
    recommender = MusicRecommender()
    
    # モデルが既に存在する場合は読み込み、そうでなければ訓練
    if not recommender.load_model():
        print("Training new model...")
        recommender.load_data()
        recommender.prepare_data()
        recommender.train_model()
        recommender.save_model()
    
    # テスト用のレコメンド
    test_user = 1
    print(f"\nUser {test_user}'s listening history:")
    history = recommender.get_user_history(test_user)
    for artist, play_count in history:
        print(f"  {artist}: {play_count} plays")
    
    print(f"\nRecommendations for User {test_user}:")
    recommendations = recommender.get_recommendations(test_user)
    for artist, score in recommendations:
        print(f"  {artist}: {score:.3f}")

if __name__ == "__main__":
    main()