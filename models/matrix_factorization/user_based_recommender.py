import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares
import pickle
import os
import argparse
from typing import Dict, List, Tuple, Union, Optional, Any
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.base_recommender import BaseRecommender

class UserBasedRecommender(BaseRecommender):
    """
    ユーザーベース協調フィルタリング推薦システム
    ALSのuser_factorsを使用してユーザー間類似度を計算し、類似ユーザーの聴取履歴に基づいて推薦を行う
    """
    
    def __init__(self, csv_path: str = 'data/user_artist_plays.csv', model_dir: str = 'weights/') -> None:
        """
        ユーザーベース推薦システムの初期化
        
        Args:
            csv_path: ユーザー-アーティスト再生データのCSVファイルパス
            model_dir: モデルを保存するディレクトリ
        """
        super().__init__(csv_path, model_dir)
        self.model: Optional[AlternatingLeastSquares] = None
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.artist_to_idx: Dict[str, int] = {}
        self.idx_to_artist: Dict[int, str] = {}
        self.user_item_matrix: Optional[csr_matrix] = None
        self.user_similarity_matrix: Optional[np.ndarray] = None
        self.alpha: float = 0.4
        
    def get_model_type(self) -> str:
        """モデル種別を返す"""
        return "user_based"
    
    def get_training_param_names(self) -> List[str]:
        """訓練パラメータ名のリストを返す"""
        return ["alpha", "factors", "regularization", "iterations"]
    
    def get_inference_param_names(self) -> List[str]:
        """推論パラメータ名のリストを返す"""
        return ["n_similar_users"]
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータの検証とデフォルト値設定"""
        validated = params.copy()
        
        # 訓練パラメータのデフォルト値設定
        validated.setdefault('alpha', 0.4)
        validated.setdefault('factors', 50)
        validated.setdefault('regularization', 0.1)
        validated.setdefault('iterations', 20)
        
        # 推論パラメータのデフォルト値設定
        validated.setdefault('n_similar_users', 10)
        
        # 値の範囲チェック
        if validated['alpha'] < 0:
            raise ValueError("alphaは0以上である必要があります")
        if validated['factors'] <= 0:
            raise ValueError("factorsは正の整数である必要があります")
        if validated['regularization'] < 0:
            raise ValueError("regularizationは0以上である必要があります")
        if validated['iterations'] <= 0:
            raise ValueError("iterationsは正の整数である必要があります")
        if validated['n_similar_users'] < 1:
            raise ValueError("n_similar_usersは1以上である必要があります")
            
        return validated
        
    def prepare_data(self, train_df: Optional[pl.DataFrame] = None) -> None:
        """
        データを前処理してユーザー-アイテム行列を作成
        
        Args:
            train_df: 訓練用データフレーム（Noneの場合はself.dfを使用）
        """
        if train_df is not None:
            self.df = train_df
        
        if self.df is None:
            raise ValueError("データが読み込まれていません。load_data()を実行するか、train_dfを指定してください")
        
        # ユーザーとアーティストのマッピングを作成
        unique_users: List[int] = self.df['user_id'].unique().to_list()
        unique_artists: List[str] = self.df['artist'].unique().to_list()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
        self.idx_to_artist = {idx: artist for artist, idx in self.artist_to_idx.items()}
        
        # スパース行列用のデータを準備
        user_indices: List[int] = [self.user_to_idx[user] for user in self.df['user_id'].to_list()]
        artist_indices: List[int] = [self.artist_to_idx[artist] for artist in self.df['artist'].to_list()]
        play_counts: np.ndarray = self.df['play_count'].to_numpy()
        
        # ユーザー-アイテム行列を作成
        self.user_item_matrix = csr_matrix(
            (play_counts, (user_indices, artist_indices)),
            shape=(len(unique_users), len(unique_artists))
        )
        
        print(f"{len(unique_users)}ユーザーと{len(unique_artists)}アーティストの行列を作成しました")
        
    def train_model(self, **params: Any) -> None:
        """
        ALSモデルを訓練し、user_factorsからユーザー間類似度行列を計算
        
        Args:
            **params: 訓練パラメータ（alpha, factors, regularization, iterations）
        """
        # パラメータ検証
        validated_params = self.validate_params(params)
        
        alpha = validated_params['alpha']
        factors = validated_params['factors']
        regularization = validated_params['regularization']
        iterations = validated_params['iterations']
        
        self.alpha = alpha  # alpha値を保存
        
        if self.user_item_matrix is None:
            raise ValueError("データが準備されていません。prepare_data()を実行してください")
        
        # ALSモデルを訓練
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
        print("ALSモデル訓練完了!")
        
        # user_factorsからユーザー間類似度を計算
        print("user_factorsからユーザー間類似度を計算中...")
        user_embeddings = self.model.user_factors
        self.user_similarity_matrix = cosine_similarity(user_embeddings)
        
        # 自分自身との類似度は1.0に設定
        np.fill_diagonal(self.user_similarity_matrix, 1.0)
        
        print("ユーザー間類似度計算完了!")
        self.is_trained = True
        
    def get_similar_users(
        self, 
        user_id: int, 
        n_similar_users: int = 10
    ) -> List[Tuple[int, float]]:
        """
        指定ユーザーに類似したユーザーを取得
        
        Args:
            user_id: 対象ユーザーID
            n_similar_users: 類似ユーザー数
            
        Returns:
            類似ユーザーのリスト [(user_id, similarity), ...]
        """
        if not self.is_trained or self.user_similarity_matrix is None:
            return []
            
        if user_id not in self.user_to_idx:
            return []
            
        user_idx = self.user_to_idx[user_id]
        similarities = self.user_similarity_matrix[user_idx].copy()
        
        # 自分自身を除外
        similarities[user_idx] = -1.0
        
        # 効率的な上位k件取得
        unsorted_max_indices = np.argpartition(-similarities, n_similar_users)[:n_similar_users]
        y = similarities[unsorted_max_indices]
        indices = np.argsort(-y)
        max_k_indices = unsorted_max_indices[indices]
        
        # 結果作成
        return [(self.idx_to_user[idx], float(similarities[idx])) for idx in max_k_indices]
    
    def get_common_artists(self, user_id1: int, user_id2: int, top_n: int = 10) -> List[Tuple[str, int, int]]:
        """
        2人のユーザーが共通して聞いているアーティストを取得
        
        Args:
            user_id1: ユーザー1のID
            user_id2: ユーザー2のID
            top_n: 上位何件を取得するか
            
        Returns:
            共通アーティストのリスト [(artist, user1_play_count, user2_play_count), ...]
        """
        user1_data = self.df.filter(pl.col('user_id') == user_id1)
        user2_data = self.df.filter(pl.col('user_id') == user_id2)
        
        # 両方のユーザーが聞いたアーティストを特定
        user1_artists = set(user1_data['artist'].to_list())
        user2_artists = set(user2_data['artist'].to_list())
        common_artists = user1_artists & user2_artists
        
        if not common_artists:
            return []
        
        # 再生回数を取得
        common_data = []
        for artist in common_artists:
            user1_plays = user1_data.filter(pl.col('artist') == artist)['play_count'].sum()
            user2_plays = user2_data.filter(pl.col('artist') == artist)['play_count'].sum()
            total_plays = user1_plays + user2_plays
            common_data.append((artist, user1_plays, user2_plays, total_plays))
        
        # 合計再生回数でソート
        common_data.sort(key=lambda x: x[3], reverse=True)
        
        return [(artist, user1_plays, user2_plays) for artist, user1_plays, user2_plays, _ in common_data[:top_n]]
    
    def get_recommendations(
        self, 
        user_id: int, 
        n_recommendations: int = 5, 
        **kwargs: Any
    ) -> Union[List[Tuple[str, float]], str]:
        """
        指定したユーザーにレコメンドを提供
        
        Args:
            user_id: ユーザーID
            n_recommendations: 推薦数
            **kwargs: 推論パラメータ
                n_similar_users: 類似ユーザー数 (デフォルト: 10)
                
        Returns:
            推薦アイテムのリスト（アイテム名、スコア）のタプル
            エラーの場合は文字列メッセージ
        """
        if not self.is_trained:
            return "モデルが訓練されていません"
            
        # パラメータのデフォルト値設定
        n_similar_users = kwargs.get('n_similar_users', 10)
        
        if user_id not in self.user_to_idx:
            return f"ユーザー {user_id} がデータセットに見つかりません"
        
        # ユーザーが既に聴いたアーティストを取得
        user_listened_artists = set(self.df.filter(pl.col('user_id') == user_id)['artist'].to_list())
        
        # 類似ユーザーを取得
        similar_users = self.get_similar_users(user_id, n_similar_users)
        
        if not similar_users:
            return "類似ユーザーが見つかりません"
        
        # 類似ユーザーが聞いているアーティストを集計
        artist_scores = {}
        
        for similar_user_id, similarity in similar_users:
            similar_user_artists = self.df.filter(pl.col('user_id') == similar_user_id)
            
            for row in similar_user_artists.iter_rows(named=True):
                artist = row['artist']
                play_count = row['play_count']
                
                # 対象ユーザーがまだ聞いていないアーティストのみ
                if artist not in user_listened_artists:
                    # スコア = 類似度 × 再生回数
                    score = similarity * play_count
                    if artist in artist_scores:
                        artist_scores[artist] += score
                    else:
                        artist_scores[artist] = score
        
        if not artist_scores:
            return "推薦可能なアーティストがありません"
        
        # スコア順にソート
        sorted_artists = sorted(artist_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_artists[:n_recommendations]
    
    def get_user_history(self, user_id: int) -> Union[List[List], str]:
        """
        ユーザーの再生履歴を取得
        
        Args:
            user_id: ユーザーID
            
        Returns:
            再生履歴のリスト（アーティスト名、再生回数）
            エラーの場合は文字列メッセージ
        """
        if user_id not in self.user_to_idx:
            return f"ユーザー {user_id} がデータセットに見つかりません"
            
        user_data = self.df.filter(pl.col('user_id') == user_id).sort('play_count', descending=True)
        return user_data.select(['artist', 'play_count']).to_numpy().tolist()
    
    def save_model(self, model_path: str) -> None:
        """
        モデルを保存
        
        Args:
            model_path: 保存先パス
        """
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
        model_data = {
            'model': self.model,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'artist_to_idx': self.artist_to_idx,
            'idx_to_artist': self.idx_to_artist,
            'user_item_matrix': self.user_item_matrix,
            'user_similarity_matrix': self.user_similarity_matrix,
            'df': self.df,
            'alpha': self.alpha,
            'is_trained': self.is_trained
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"モデルを{model_path}に保存しました")
    
    def load_model(self, model_path: str) -> None:
        """
        モデルを読み込み
        
        Args:
            model_path: 読み込み元パス
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.user_to_idx = model_data['user_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.artist_to_idx = model_data['artist_to_idx']
        self.idx_to_artist = model_data['idx_to_artist']
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_similarity_matrix = model_data['user_similarity_matrix']
        self.df = model_data['df']
        self.alpha = model_data.get('alpha', 0.4)
        self.is_trained = model_data.get('is_trained', True)
        
        print(f"{model_path}からモデルを読み込みました")

def main():
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description='ユーザーベース推薦システム')
    parser.add_argument('--csv-path', default='data/user_artist_plays.csv',
                        help='ユーザー-アーティスト再生データを含むCSVファイルパス (デフォルト: data/user_artist_plays.csv)')
    parser.add_argument('--model-dir', default='weights/',
                        help='モデルファイルを保存/読み込みするディレクトリ (デフォルト: weights/)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='信頼度計算用alphaパラメータ (デフォルト: 0.4)')
    parser.add_argument('--factors', type=int, default=50,
                        help='潜在因子数 (デフォルト: 50)')
    parser.add_argument('--regularization', type=float, default=0.1,
                        help='正則化パラメータ (デフォルト: 0.1)')
    parser.add_argument('--iterations', type=int, default=20,
                        help='イテレーション数 (デフォルト: 20)')
    parser.add_argument('--n-similar-users', type=int, default=10,
                        help='類似ユーザー数 (デフォルト: 10)')
    parser.add_argument('--user-id', type=int, default=1,
                        help='推薦テスト用のユーザーID (デフォルト: 1)')
    parser.add_argument('--n-recommendations', type=int, default=5,
                        help='生成する推薦数 (デフォルト: 5)')
    parser.add_argument('--train', action='store_true',
                        help='既存のモデルがあっても強制的に新しいモデルを訓練')
    
    args = parser.parse_args()
    
    # CSVファイルとモデルディレクトリの存在確認
    if not os.path.exists(args.csv_path):
        print(f"エラー: CSVファイルが見つかりません: {args.csv_path}")
        return
    
    if not os.path.exists(args.model_dir):
        print(f"モデルディレクトリを作成中: {args.model_dir}")
        os.makedirs(args.model_dir, exist_ok=True)
    
    recommender = UserBasedRecommender(csv_path=args.csv_path, model_dir=args.model_dir)
    
    # モデルが既に存在する場合は読み込み、そうでなければ訓練
    csv_basename = os.path.splitext(os.path.basename(args.csv_path))[0]
    model_filename = f'{csv_basename}_user_based_alpha_{args.alpha:.1f}.pkl'
    model_path = os.path.join(args.model_dir, model_filename)
    
    model_loaded = False
    if not args.train and os.path.exists(model_path):
        try:
            recommender.load_model(model_path)
            model_loaded = True
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
    
    if not model_loaded:
        print("新しいモデルを訓練中...")
        recommender.load_data()
        recommender.prepare_data()
        recommender.train_model(
            alpha=args.alpha,
            factors=args.factors,
            regularization=args.regularization,
            iterations=args.iterations
        )
        recommender.save_model(model_path)
    
    # テスト用のレコメンド
    print(f"\nユーザー {args.user_id} の再生履歴:")
    history = recommender.get_user_history(args.user_id)
    if isinstance(history, str):
        print(f"  {history}")
        return
    
    for artist, play_count in history[:10]:  # 上位10件を表示
        print(f"  {artist}: {play_count}回")
    
    print(f"\nユーザー {args.user_id} の類似ユーザー:")
    similar_users = recommender.get_similar_users(
        args.user_id, 
        args.n_similar_users
    )
    for similar_user_id, similarity in similar_users[:5]:  # 上位5人を表示
        print(f"  ユーザー {similar_user_id}: 類似度 {similarity:.3f}")
    
    print(f"\nユーザー {args.user_id} への推薦:")
    recommendations = recommender.get_recommendations(
        args.user_id, 
        args.n_recommendations,
        n_similar_users=args.n_similar_users
    )
    
    if isinstance(recommendations, str):
        print(f"  {recommendations}")
        return
    
    for artist, score in recommendations:
        print(f"  {artist}: {score:.3f}")

if __name__ == "__main__":
    main()