"""
Module: model_baselines
Responsibility:
- Wraps machine learning baseline rankers to match the heuristic ranking
  function signatures.
- Uses truncated SVD via scipy (no compiled C++ dependencies required).
- Trains purely on historical data and provides a `score_pool` method
  to rank candidates.

Design Note:
  I originally planned to use `benfred/implicit` (ALS), but it requires
  C++ build tools on Windows. This fallback uses scipy.sparse.linalg.svds
  to decompose the user-item interaction matrix, achieving a similar
  latent-factor collaborative filtering effect without extra dependencies.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds


class SVDBaseline:
    """
    Truncated SVD Matrix Factorization baseline using scipy.
    Decomposes the user-item implicit interaction matrix into latent factors
    and scores candidates via dot-product similarity.
    """

    def __init__(self, factors=50):
        """
        Args:
            factors: Number of latent dimensions for the SVD decomposition.
        """
        self.factors = factors

        # Mappings from raw IDs to contiguous integer matrix indices
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        # Learned latent factor matrices
        self.user_factors = None  # shape: (n_users, factors)
        self.item_factors = None  # shape: (n_items, factors)

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the SVD model on a historical training dataframe.
        Constructs a sparse user-item interaction matrix from implicit/explicit signals.
        """
        # Create contiguous internal integer indices
        unique_users = train_df["user_id"].unique()
        unique_items = train_df["video_id"].unique()

        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.item_to_idx = {iid: i for i, iid in enumerate(unique_items)}
        self.idx_to_item = {i: iid for iid, i in self.item_to_idx.items()}

        # Map dataframe to internal indices
        user_indices = train_df["user_id"].map(self.user_to_idx).values
        item_indices = train_df["video_id"].map(self.item_to_idx).values

        # Construct interaction weights (same formula as heuristic base_score)
        weights = (
            train_df["implicit_completion_ratio"].fillna(0.0)
            + train_df["explicit_positive_any"].fillna(0) * 2.0
        ).values
        # Floor at a small positive value so every interaction registers
        weights = np.clip(weights, a_min=0.01, a_max=None)

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Build sparse user x item matrix
        self.interaction_matrix = sparse.csr_matrix(
            (weights, (user_indices, item_indices)),
            shape=(n_users, n_items),
        )

        # Truncated SVD: cap factors at min(matrix dims) - 1
        k = min(self.factors, min(n_users, n_items) - 1)
        print(f"Fitting SVD model (k={k}) on {n_users} users × {n_items} items...")

        # Convert to float64 for numerical stability
        mat = self.interaction_matrix.astype(np.float64)
        U, sigma, Vt = svds(mat, k=k)

        # Store user and item factors with singular values folded into both sides
        sqrt_sigma = np.sqrt(sigma)
        self.user_factors = U * sqrt_sigma[np.newaxis, :]  # (n_users, k)
        self.item_factors = (Vt.T * sqrt_sigma[np.newaxis, :])  # (n_items, k)

        print("SVD model fit complete.")

    def score_pool(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks a candidate pool for a given user using learned latent factors.
        Returns a DataFrame matching the heuristic strategy output format.
        """
        if session_df.empty:
            return session_df

        user_id = session_df["user_id"].iloc[0]
        pool = session_df.copy()

        # Cold-start user: return random scores
        if user_id not in self.user_to_idx:
            pool["score"] = np.random.rand(len(pool)) * 0.01
            pool["rank"] = pool["score"].rank(ascending=False, method="first").astype(int)
            pool["strategy"] = "svd_baseline (cold user)"
            return pool.sort_values("rank")

        internal_uid = self.user_to_idx[user_id]
        user_vec = self.user_factors[internal_uid]  # (k,)

        # Score each candidate by dot product
        scores = []
        for vid in pool["video_id"]:
            if vid in self.item_to_idx:
                item_vec = self.item_factors[self.item_to_idx[vid]]
                scores.append(float(np.dot(user_vec, item_vec)))
            else:
                # Cold-start item: small random score
                scores.append(float(np.random.rand() * 0.001))

        pool["score"] = scores
        pool = pool.sort_values("score", ascending=False).reset_index(drop=True)
        pool["rank"] = pool.index + 1
        pool["strategy"] = "svd_baseline"

        return pool


def svd_ranker(session_df: pd.DataFrame, model_instance: SVDBaseline) -> pd.DataFrame:
    """
    Convenience wrapper matching the notebook evaluation loop signature.
    """
    return model_instance.score_pool(session_df)
