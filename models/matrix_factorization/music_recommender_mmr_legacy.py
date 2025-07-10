import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Union, Optional, Any

class MusicRecommenderMMR:
    def __init__(self, csv_path: str = 'user_artist_plays.csv', model_dir: str = 'weights/') -> None:
        """
        MMR拡張音楽推薦システムの初期化
        
        Args:
            csv_path: ユーザー-アーティスト再生データのCSVファイルパス
            model_dir: モデルを保存するディレクトリ
        """
        self.model: Optional[AlternatingLeastSquares] = None
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.artist_to_idx: Dict[str, int] = {}
        self.idx_to_artist: Dict[int, str] = {}
        self.user_item_matrix: Optional[csr_matrix] = None
        self.csv_path: str = csv_path
        self.model_dir: str = model_dir
        self.df: Optional[pl.DataFrame] = None
        self.alpha: float = 0.4
        
    def load_data(self, csv_path=None):
        """CSVファイルからデータを読み込み"""
        if csv_path is None:
            csv_path = self.csv_path
        self.df = pl.read_csv(csv_path)
        print(f"{csv_path}から{len(self.df)}件の再生記録を読み込みました")
        
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
        
        print(f"{len(unique_users)}ユーザーと{len(unique_artists)}アーティストの行列を作成しました")
        
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
        
        print(f"alpha={alpha}でALSモデルを訓練中...")
        self.model.fit(confidence_matrix)
        print("訓練完了!")
        
    def mmr_rerank(self, candidates, similarity_matrix, lambda_param=0.5, n_recommendations=5):
        """
        MMR (Maximal Marginal Relevance) を使用してリランキング
        
        Args:
            candidates: [(item_idx, relevance_score), ...] の候補リスト
            similarity_matrix: アイテム間の類似度行列
            lambda_param: 関連性と多様性のバランスパラメータ (0-1)
            n_recommendations: 最終的な推薦数
        
        Returns:
            MMRでリランキングされた推薦リスト
        """
        if not candidates:
            return []
        
        selected = []
        remaining = candidates.copy()
        
        # 全てのアイテムをMMRで選択（最初のアイテムも含む）
        while len(selected) < n_recommendations and remaining:
            mmr_scores = []
            
            for item_idx, relevance_score in remaining:
                # 関連性スコア
                relevance = relevance_score
                
                # 既に選択されたアイテムとの最大類似度
                max_similarity = 0.0
                if selected:  # 選択されたアイテムがある場合のみ類似度を計算
                    for selected_item_idx, _ in selected:
                        similarity = similarity_matrix[item_idx, selected_item_idx]
                        max_similarity = max(max_similarity, similarity)
                
                # MMRスコア計算
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((item_idx, relevance_score, mmr_score))
            
            # MMRスコアが最も高いアイテムを選択
            best_item = max(mmr_scores, key=lambda x: x[2])
            selected.append((best_item[0], best_item[1]))  # (item_idx, relevance_score)
            remaining = [(idx, score) for idx, score in remaining if idx != best_item[0]]
        
        return selected
    
    def get_artist_similarity_matrix(self):
        """アーティスト間の類似度行列を計算"""
        # アーティストの埋め込みベクトルを使用して類似度を計算
        artist_embeddings = self.model.item_factors
        similarity_matrix = cosine_similarity(artist_embeddings)
        return similarity_matrix
    
    def get_recommendations(self, user_id, n_recommendations=5, use_mmr=True, lambda_param=0.5, candidate_pool_size=20):
        """
        指定したユーザーにレコメンドを提供
        
        Args:
            user_id: ユーザーID
            n_recommendations: 最終推薦数
            use_mmr: MMRを使用するかどうか
            lambda_param: MMRのλパラメータ（関連性と多様性のバランス）
            candidate_pool_size: MMR前の候補数（通常は最終推薦数より多く）
        """
        if user_id not in self.user_to_idx:
            return f"ユーザー {user_id} がデータセットに見つかりません"
            
        user_idx = self.user_to_idx[user_id]
        
        # ユーザーが既に聴いたアーティストを取得
        user_items = set(self.df.filter(pl.col('user_id') == user_id)['artist'].to_list())
        
        # ユーザーファクターとアイテムファクターの内積でスコア計算
        if user_idx >= self.model.user_factors.shape[0]:
            # ユーザーが訓練データにない場合は、平均的なレコメンドを返す
            item_scores = np.mean(self.model.item_factors, axis=1)
        else:
            item_scores = self.model.item_factors.dot(self.model.user_factors[user_idx])
        
        # スコアの高い順にソート
        top_items = np.argsort(item_scores)[::-1]
        
        # 候補プールを作成（MMRを使用する場合はより多くの候補を取得）
        pool_size = candidate_pool_size if use_mmr else n_recommendations
        candidates = []
        
        for item_idx in top_items:
            if item_idx < len(self.idx_to_artist):
                artist = self.idx_to_artist[item_idx]
                if artist not in user_items:
                    candidates.append((item_idx, float(item_scores[item_idx])))
                if len(candidates) >= pool_size:
                    break
        
        if not use_mmr:
            # MMRを使用しない場合は、スコア順の上位をそのまま返す
            return [(self.idx_to_artist[item_idx], score) for item_idx, score in candidates[:n_recommendations]]
        
        # MMRを使用してリランキング
        print(f"MMRリランキング (λ={lambda_param}) を{len(candidates)}件の候補に適用中...")
        similarity_matrix = self.get_artist_similarity_matrix()
        mmr_results = self.mmr_rerank(candidates, similarity_matrix, lambda_param, n_recommendations)
        
        # アーティスト名と共に結果を返す
        recommendations = [(self.idx_to_artist[item_idx], score) for item_idx, score in mmr_results]
        return recommendations
    
    def get_user_history(self, user_id):
        """ユーザーの再生履歴を取得"""
        if user_id not in self.user_to_idx:
            return f"ユーザー {user_id} がデータセットに見つかりません"
            
        user_data = self.df.filter(pl.col('user_id') == user_id).sort('play_count', descending=True)
        return user_data.select(['artist', 'play_count']).to_numpy().tolist()
    
    def save_model(self, alpha=None, path=None):
        """モデルを保存（alpha値ごとに別ファイル）"""
        if alpha is None:
            alpha = getattr(self, 'alpha', 0.4)
        if path is None:
            csv_basename = os.path.splitext(os.path.basename(self.csv_path))[0]
            filename = f'{csv_basename}_mmr_alpha_{alpha:.1f}.pkl'
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
        print(f"モデルを{path}に保存しました")
    
    def load_model(self, alpha=0.4, path=None):
        """モデルを読み込み（alpha値に基づいてファイルを選択）"""
        if path is None:
            csv_basename = os.path.splitext(os.path.basename(self.csv_path))[0]
            filename = f'{csv_basename}_mmr_alpha_{alpha:.1f}.pkl'
            path = os.path.join(self.model_dir, filename)
            
        if not os.path.exists(path):
            print(f"モデルファイルが見つかりません: {path}")
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
        
        print(f"{path}からモデルを読み込みました")
        return True

def main():
    parser = argparse.ArgumentParser(description='MMR付き音楽推薦システム')
    parser.add_argument('--csv-path', default='user_artist_plays.csv',
                        help='ユーザー-アーティスト再生データを含むCSVファイルパス (デフォルト: user_artist_plays.csv)')
    parser.add_argument('--model-dir', default='weights/',
                        help='モデルファイルを保存/読み込みするディレクトリ (デフォルト: weights/)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='信頼度計算用alphaパラメータ (デフォルト: 0.4)')
    parser.add_argument('--user-id', type=int, default=1,
                        help='推薦テスト用のユーザーID (デフォルト: 1)')
    parser.add_argument('--n-recommendations', type=int, default=5,
                        help='生成する推薦数 (デフォルト: 5)')
    parser.add_argument('--train', action='store_true',
                        help='既存のモデルがあっても強制的に新しいモデルを訓練')
    
    # MMR特有のパラメータ
    parser.add_argument('--use-mmr', action='store_true', default=True,
                        help='リランキングにMMRを使用 (デフォルト: True)')
    parser.add_argument('--no-mmr', action='store_true',
                        help='MMRリランキングを無効化')
    parser.add_argument('--lambda-param', type=float, default=0.5,
                        help='MMR用のLambdaパラメータ (0=多様性, 1=関連性) (デフォルト: 0.5)')
    parser.add_argument('--candidate-pool-size', type=int, default=20,
                        help='MMRリランキング前の候補プールサイズ (デフォルト: 20)')
    
    args = parser.parse_args()
    
    # MMRフラグの処理
    if args.no_mmr:
        args.use_mmr = False
    
    # CSVファイルとモデルディレクトリの存在確認
    if not os.path.exists(args.csv_path):
        print(f"エラー: CSVファイルが見つかりません: {args.csv_path}")
        return
    
    if not os.path.exists(args.model_dir):
        print(f"モデルディレクトリを作成中: {args.model_dir}")
        os.makedirs(args.model_dir, exist_ok=True)
    
    recommender = MusicRecommenderMMR(csv_path=args.csv_path, model_dir=args.model_dir)
    
    # モデルが既に存在する場合は読み込み、そうでなければ訓練
    model_loaded = False
    if not args.train:
        model_loaded = recommender.load_model(alpha=args.alpha)
    
    if not model_loaded:
        print("新しいモデルを訓練中...")
        recommender.load_data()
        recommender.prepare_data()
        recommender.train_model(alpha=args.alpha)
        recommender.save_model(alpha=args.alpha)
    
    # テスト用のレコメンド
    print(f"\nユーザー {args.user_id} の再生履歴:")
    history = recommender.get_user_history(args.user_id)
    if isinstance(history, str):
        print(f"  {history}")
        return
    
    for artist, play_count in history:
        print(f"  {artist}: {play_count}回")
    
    # レコメンド方法の表示
    method = "MMR" if args.use_mmr else "標準"
    if args.use_mmr:
        print(f"\nユーザー {args.user_id} への推薦 ({method}, λ={args.lambda_param}):")
    else:
        print(f"\nユーザー {args.user_id} への推薦 ({method}):")
    
    recommendations = recommender.get_recommendations(
        args.user_id, 
        args.n_recommendations, 
        use_mmr=args.use_mmr,
        lambda_param=args.lambda_param,
        candidate_pool_size=args.candidate_pool_size
    )
    
    if isinstance(recommendations, str):
        print(f"  {recommendations}")
        return
    
    for i, (artist, score) in enumerate(recommendations, 1):
        print(f"  {i}. {artist}: {score:.3f}")

if __name__ == "__main__":
    main()