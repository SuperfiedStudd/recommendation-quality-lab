"""
Module: recommender_service
Responsibility:
- Provide a unified recommendation pipeline that wires together
  state lookup, candidate generation, and ranking.
- Reuses existing SessionCandidateGenerator and ranking strategies.
"""

import pandas as pd
import sys
import os

# Ensure the src directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from candidate_generation import SessionCandidateGenerator
import ranking_strategies


class RecommenderService:
    """
    End-to-end recommendation pipeline.

    Pipeline:
        1. Retrieve user/session state from StateManager
        2. Generate candidate pool via SessionCandidateGenerator
        3. Rank candidates using a pluggable strategy
        4. Return top-K recommendations
    """

    STRATEGY_MAP = {
        "popularity": ranking_strategies.popularity_based,
        "freshness_boosted": ranking_strategies.freshness_boosted,
        "diversity_aware": ranking_strategies.diversity_aware_rerank,
    }

    def __init__(self, master_df: pd.DataFrame, state_manager=None,
                 strategy: str = "popularity"):
        """
        Args:
            master_df: The full prepared pipeline dataframe.
            state_manager: Optional StateManager instance for state lookups.
            strategy: Name of the ranking strategy. One of:
                      'popularity', 'freshness_boosted', 'diversity_aware'.
        """
        self.candidate_gen = SessionCandidateGenerator(master_df)
        self.state_manager = state_manager
        self.strategy_name = strategy

        if strategy not in self.STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(self.STRATEGY_MAP.keys())}"
            )
        self.rank_fn = self.STRATEGY_MAP[strategy]

    def recommend(self, user_id: int, session_id: str, k: int = 20) -> pd.DataFrame:
        """
        Generate top-K recommendations for a user in a given session.

        Args:
            user_id: Target user ID.
            session_id: Current session ID.
            k: Number of recommendations to return.

        Returns:
            DataFrame with columns: user_id, session_id, video_id, rank, score, strategy
        """
        # 1. Generate candidate pool (reuses existing time-aware generator)
        try:
            pool = self.candidate_gen.generate_pool(user_id, session_id, pool_size=100)
        except ValueError:
            # Session not found — return empty
            return pd.DataFrame(columns=["user_id", "session_id", "video_id",
                                         "rank", "score", "strategy"])

        # 2. Rank candidates
        ranked = self.rank_fn(pool)

        # 3. Return top-K
        top_k = ranked.head(k).copy()
        return top_k

    def recommend_batch(self, user_session_pairs: list, k: int = 20) -> pd.DataFrame:
        """
        Batch recommendation for multiple (user_id, session_id) pairs.

        Args:
            user_session_pairs: List of (user_id, session_id) tuples.
            k: Number of recommendations per pair.

        Returns:
            Concatenated DataFrame of all recommendations.
        """
        results = []
        for user_id, session_id in user_session_pairs:
            recs = self.recommend(user_id, session_id, k=k)
            if not recs.empty:
                results.append(recs)

        if not results:
            return pd.DataFrame(columns=["user_id", "session_id", "video_id",
                                         "rank", "score", "strategy"])
        return pd.concat(results, ignore_index=True)
