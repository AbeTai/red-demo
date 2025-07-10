import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Union, Optional, Any
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.base_recommender import BaseRecommender

class MusicRecommenderMMR(BaseRecommender):
    """
    MMR（Maximal Marginal Relevance）を使用したALS音楽推薦システム
    """
    
    def __init__(self, csv_path: str = 'data/user_artist_plays.csv', model_dir: str = 'weights/') -> None:
        """
        MMR拡張音楽推薦システムの初期化
        
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
        self.alpha: float = 0.4
        
    def get_model_type(self) -> str:
        """モデル種別を返す"""
        return "matrix_factorization"
    
    def get_training_param_names(self) -> List[str]:
        """訓練パラメータ名のリストを返す"""
        return ["alpha", "factors", "regularization", "iterations"]
    
    def get_inference_param_names(self) -> List[str]:
        """推論パラメータ名のリストを返す"""
        return ["use_mmr", "lambda_param", "candidate_pool_size"]
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータの検証とデフォルト値設定"""
        validated = params.copy()
        
        # 訓練パラメータのデフォルト値設定
        validated.setdefault('alpha', 0.4)
        validated.setdefault('factors', 50)
        validated.setdefault('regularization', 0.1)
        validated.setdefault('iterations', 20)
        
        # 推論パラメータのデフォルト値設定
        validated.setdefault('use_mmr', True)
        validated.setdefault('lambda_param', 0.5)
        validated.setdefault('candidate_pool_size', 20)
        
        # 値の範囲チェック
        if validated['alpha'] < 0:
            raise ValueError("alphaは0以上である必要があります")
        if validated['factors'] <= 0:
            raise ValueError("factorsは正の整数である必要があります")
        if validated['regularization'] < 0:
            raise ValueError("regularizationは0以上である必要があります")
        if validated['iterations'] <= 0:
            raise ValueError("iterationsは正の整数である必要があります")
        if not 0 <= validated['lambda_param'] <= 1:
            raise ValueError("lambda_paramは0から1の間である必要があります")
        if validated['candidate_pool_size'] <= 0:
            raise ValueError("candidate_pool_sizeは正の整数である必要があります")
            
        return validated
        
    def prepare_data(self, train_df: Optional[pl.DataFrame] = None) -> None:
        """
        データを前処理してスパース行列を作成
        
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
        ALSモデルを訓練
        
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
        self.is_trained = True
        
    def mmr_rerank(
        self, 
        candidates: List[Tuple[int, float]], 
        similarity_matrix: np.ndarray, 
        lambda_param: float = 0.5, 
        n_recommendations: int = 5
    ) -> List[Tuple[int, float]]:
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
    
    def get_artist_similarity_matrix(self) -> np.ndarray:
        """アーティスト間の類似度行列を計算"""
        if not self.is_trained or self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        # アーティストの埋め込みベクトルを使用して類似度を計算
        artist_embeddings = self.model.item_factors
        similarity_matrix = cosine_similarity(artist_embeddings)
        return similarity_matrix
    
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
            n_recommendations: 最終推薦数
            **kwargs: 推論パラメータ
                use_mmr: MMRを使用するかどうか (デフォルト: True)
                lambda_param: MMRのλパラメータ（関連性と多様性のバランス） (デフォルト: 0.5)
                candidate_pool_size: MMR前の候補数 (デフォルト: 20)
                
        Returns:
            推薦アイテムのリスト（アイテム名、スコア）のタプル
            エラーの場合は文字列メッセージ
        """
        if not self.is_trained or self.model is None:
            return "モデルが訓練されていません"
            
        # パラメータのデフォルト値設定
        use_mmr = kwargs.get('use_mmr', True)
        lambda_param = kwargs.get('lambda_param', 0.5)
        candidate_pool_size = kwargs.get('candidate_pool_size', 20)
        
        if user_id not in self.user_to_idx:
            return f"ユーザー {user_id} がデータセットに見つかりません"
            
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
        
        # 候補を生成（ユーザーが聴いていないアイテムのみ）
        candidates = []
        for item_idx in top_items:
            if item_idx < len(self.idx_to_artist):
                artist = self.idx_to_artist[item_idx]
                if artist not in user_items:
                    candidates.append((item_idx, float(item_scores[item_idx])))
                # MMRを使用する場合は候補プールサイズまで、使用しない場合は最終推薦数まで
                target_size = candidate_pool_size if use_mmr else n_recommendations
                if len(candidates) >= target_size:
                    break
        
        if not use_mmr:
            # MMRを使用しない場合はそのまま返す
            final_recommendations = candidates[:n_recommendations]
        else:
            # MMRを使用してリランキング
            if len(candidates) <= n_recommendations:
                # 候補数が推薦数以下の場合はそのまま返す
                final_recommendations = candidates
            else:
                print(f"MMRリランキング (λ={lambda_param}) を{len(candidates)}件の候補に適用中...")
                similarity_matrix = self.get_artist_similarity_matrix()
                final_recommendations = self.mmr_rerank(
                    candidates, similarity_matrix, lambda_param, n_recommendations
                )
        
        # アーティスト名に変換
        recommendations = []
        for item_idx, score in final_recommendations:
            artist = self.idx_to_artist[item_idx]
            recommendations.append((artist, score))
                    
        return recommendations
    
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
        self.df = model_data['df']
        self.alpha = model_data.get('alpha', 0.4)
        self.is_trained = model_data.get('is_trained', True)
        
        print(f"{model_path}からモデルを読み込みました")

def main():
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description='MMR音楽推薦システム')
    parser.add_argument('--csv-path', default='data/user_artist_plays.csv',
                        help='ユーザー-アーティスト再生データを含むCSVファイルパス (デフォルト: data/user_artist_plays.csv)')
    parser.add_argument('--model-dir', default='weights/',
                        help='モデルファイルを保存/読み込みするディレクトリ (デフォルト: weights/)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='信頼度計算用alphaパラメータ (デフォルト: 0.4)')
    parser.add_argument('--lambda-param', type=float, default=0.5,
                        help='MMRのλパラメータ（関連性と多様性のバランス） (デフォルト: 0.5)')
    parser.add_argument('--candidate-pool-size', type=int, default=20,
                        help='MMR前の候補プールサイズ (デフォルト: 20)')
    parser.add_argument('--user-id', type=int, default=1,
                        help='推薦テスト用のユーザーID (デフォルト: 1)')
    parser.add_argument('--n-recommendations', type=int, default=5,
                        help='生成する推薦数 (デフォルト: 5)')
    parser.add_argument('--no-mmr', action='store_true',
                        help='MMRを使用しない（多様性なし）')
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
    
    recommender = MusicRecommenderMMR(csv_path=args.csv_path, model_dir=args.model_dir)
    
    # モデルが既に存在する場合は読み込み、そうでなければ訓練
    csv_basename = os.path.splitext(os.path.basename(args.csv_path))[0]
    model_filename = f'{csv_basename}_mmr_alpha_{args.alpha:.1f}.pkl'
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
        recommender.train_model(alpha=args.alpha)
        recommender.save_model(model_path)
    
    # テスト用のレコメンド
    print(f"\nユーザー {args.user_id} の再生履歴:")
    history = recommender.get_user_history(args.user_id)
    if isinstance(history, str):
        print(f"  {history}")
        return
    
    for artist, play_count in history:
        print(f"  {artist}: {play_count}回")
    
    use_mmr = not args.no_mmr
    mmr_info = f"MMR有効 (λ={args.lambda_param})" if use_mmr else "MMR無効"
    print(f"\nユーザー {args.user_id} への推薦 ({mmr_info}):")
    
    recommendations = recommender.get_recommendations(
        args.user_id, 
        args.n_recommendations,
        use_mmr=use_mmr,
        lambda_param=args.lambda_param,
        candidate_pool_size=args.candidate_pool_size
    )
    
    if isinstance(recommendations, str):
        print(f"  {recommendations}")
        return
    
    for artist, score in recommendations:
        print(f"  {artist}: {score:.3f}")

if __name__ == "__main__":
    main()