import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle
import os
import argparse

class MusicRecommender:
    def __init__(self, csv_path='user_artist_plays.csv', model_dir='weights/'):
        self.model = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.artist_to_idx = {}
        self.idx_to_artist = {}
        self.user_item_matrix = None
        self.csv_path = csv_path
        self.model_dir = model_dir
        
    def load_data(self, csv_path=None):
        """CSVファイルからデータを読み込み"""
        if csv_path is None:
            csv_path = self.csv_path
        self.df = pl.read_csv(csv_path)
        print(f"Loaded {len(self.df)} play records from {csv_path}")
        
    def prepare_data(self):
        """データを前処理してスパース行列を作成"""
        # ユーザーとアーティストのマッピングを作成
        unique_users = self.df['user_id'].unique().to_list()
        unique_artists = self.df['artist'].unique().to_list()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
        self.idx_to_artist = {idx: artist for artist, idx in self.artist_to_idx.items()}
        
        # スパース行列用のデータを準備
        user_indices = [self.user_to_idx[user] for user in self.df['user_id'].to_list()]
        artist_indices = [self.artist_to_idx[artist] for artist in self.df['artist'].to_list()]
        play_counts = self.df['play_count'].to_numpy()
        
        # ユーザー-アイテム行列を作成
        self.user_item_matrix = csr_matrix(
            (play_counts, (user_indices, artist_indices)),
            shape=(len(unique_users), len(unique_artists))
        )
        
        print(f"Created matrix with {len(unique_users)} users and {len(unique_artists)} artists")
        
    def train_model(self, factors=50, regularization=0.1, iterations=20, alpha=0.4):
        """ALSモデルを訓練"""
        self.alpha = alpha  # alpha値を保存
        
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        
        # implicitは confidence matrix を期待する
        # 再生回数をそのまま信頼度として使用（1 + alpha * play_count の形式）
        confidence_matrix = self.user_item_matrix.copy()
        confidence_matrix.data = 1 + confidence_matrix.data * alpha
        
        print(f"Training ALS model with alpha={alpha}...")
        self.model.fit(confidence_matrix)
        print("Training completed!")
        
    def get_recommendations(self, user_id, n_recommendations=5):
        """指定したユーザーにレコメンドを提供"""
        if user_id not in self.user_to_idx:
            return f"User {user_id} not found in the dataset"
            
        user_idx = self.user_to_idx[user_id]
        
        # ユーザーが既に聴いたアーティストを取得
        user_items = set(self.df.filter(pl.col('user_id') == user_id)['artist'].to_list())
        
        # レコメンドを取得（協調フィルタリングによる手動実装）
        
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
            
        user_data = self.df.filter(pl.col('user_id') == user_id).sort('play_count', descending=True)
        return user_data.select(['artist', 'play_count']).to_numpy().tolist()
    
    def save_model(self, alpha=None, path=None):
        """モデルを保存（alpha値ごとに別ファイル）"""
        if alpha is None:
            alpha = getattr(self, 'alpha', 0.4)
        if path is None:
            csv_basename = os.path.splitext(os.path.basename(self.csv_path))[0]
            filename = f'{csv_basename}_alpha_{alpha:.1f}.pkl'
            path = os.path.join(self.model_dir, filename)
            
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(path), exist_ok=True)
            
        model_data = {
            'model': self.model,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'artist_to_idx': self.artist_to_idx,
            'idx_to_artist': self.idx_to_artist,
            'user_item_matrix': self.user_item_matrix,
            'df': self.df,
            'alpha': alpha
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    def load_model(self, alpha=0.4, path=None):
        """モデルを読み込み（alpha値に基づいてファイルを選択）"""
        if path is None:
            csv_basename = os.path.splitext(os.path.basename(self.csv_path))[0]
            filename = f'{csv_basename}_alpha_{alpha:.1f}.pkl'
            path = os.path.join(self.model_dir, filename)
            
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
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
        self.alpha = model_data.get('alpha', alpha)
        
        print(f"Model loaded from {path}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Music Recommender System')
    parser.add_argument('--csv-path', default='user_artist_plays.csv',
                        help='Path to the CSV file containing user-artist play data (default: user_artist_plays.csv)')
    parser.add_argument('--model-dir', default='weights/',
                        help='Directory to store/load model files (default: weights/)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Alpha parameter for confidence calculation (default: 0.4)')
    parser.add_argument('--user-id', type=int, default=1,
                        help='User ID for testing recommendations (default: 1)')
    parser.add_argument('--n-recommendations', type=int, default=5,
                        help='Number of recommendations to generate (default: 5)')
    parser.add_argument('--train', action='store_true',
                        help='Force training a new model even if one exists')
    
    args = parser.parse_args()
    
    # CSVファイルとモデルディレクトリの存在確認
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    if not os.path.exists(args.model_dir):
        print(f"Creating model directory: {args.model_dir}")
        os.makedirs(args.model_dir, exist_ok=True)
    
    recommender = MusicRecommender(csv_path=args.csv_path, model_dir=args.model_dir)
    
    # モデルが既に存在する場合は読み込み、そうでなければ訓練
    model_loaded = False
    if not args.train:
        model_loaded = recommender.load_model(alpha=args.alpha)
    
    if not model_loaded:
        print("Training new model...")
        recommender.load_data()
        recommender.prepare_data()
        recommender.train_model(alpha=args.alpha)
        recommender.save_model(alpha=args.alpha)
    
    # テスト用のレコメンド
    print(f"\nUser {args.user_id}'s listening history:")
    history = recommender.get_user_history(args.user_id)
    if isinstance(history, str):
        print(f"  {history}")
        return
    
    for artist, play_count in history:
        print(f"  {artist}: {play_count} plays")
    
    print(f"\nRecommendations for User {args.user_id}:")
    recommendations = recommender.get_recommendations(args.user_id, args.n_recommendations)
    if isinstance(recommendations, str):
        print(f"  {recommendations}")
        return
    
    for artist, score in recommendations:
        print(f"  {artist}: {score:.3f}")

if __name__ == "__main__":
    main()