"""
Module: ranking_strategies
Responsibility:
- Implement simple, non-ML baseline ranking strategies for DiscoveryRank evaluation.
- Each function accepts a session-level DataFrame (one session's interactions) and returns
  a ranked DataFrame with columns: user_id, session_id, video_id, rank, score.
- Strategies must use only fields already present in the prepared pipeline output.
"""

import numpy as np
import pandas as pd


def _base_score(session_df: pd.DataFrame) -> pd.Series:
    """
    Shared engagement-based score used as the foundation for all strategies.

    score = implicit_completion_ratio + explicit_positive_any * 2 + epsilon

    Rationale:
    - explicit_positive_any (like/follow/forward/comment) is a strong positive signal, weighted 2x.
    - implicit_completion_ratio provides a continuous proxy for relevance where explicit signals are absent.
    - is_hate is NOT penalized here: it drops y_relevant to 0, so it self-penalizes at metric time.
    - Added 1e-4 epsilon so unobserved/synthetic candidates (0 engagement) have a non-zero base score.
      This allows multiplicative freshness decay to differentiate them rather than collapsing to 0.
    """
    score = session_df["implicit_completion_ratio"] + session_df["explicit_positive_any"] * 2.0 + 1e-4
    return score


def _build_ranked_df(session_df: pd.DataFrame, scores: pd.Series, strategy: str) -> pd.DataFrame:
    """
    Takes a session DataFrame and a score Series (same index), returns a clean ranked output DataFrame.
    """
    out = session_df[["user_id", "session_id", "video_id"]].copy()
    out["score"] = scores.values
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1  # 1-indexed
    out["strategy"] = strategy
    return out[["user_id", "session_id", "video_id", "rank", "score", "strategy"]]


def popularity_based(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank items by a simple aggregate engagement score.

    score = implicit_completion_ratio + explicit_positive_any * 2

    Returns a ranked DataFrame with: user_id, session_id, video_id, rank, score, strategy.
    """
    scores = _base_score(session_df)
    return _build_ranked_df(session_df, scores, strategy="popularity_based")


def freshness_boosted(session_df: pd.DataFrame, decay_rate: float = 30.0) -> pd.DataFrame:
    """
    Apply an exponential temporal decay to the base engagement score to favour newer items.

    boosted_score = base_score * exp(-item_age_days / decay_rate)

    Notes on the decay:
    - This is an exponential decay with time constant `decay_rate` (e-folding time).
    - At item_age_days == decay_rate, the score is multiplied by 1/e (~0.368).
    - At item_age_days == 2 * decay_rate, the score is multiplied by 1/e^2 (~0.135).
    - This is NOT the half-life form. The equivalent half-life of this decay is:
        t_half = decay_rate * ln(2) ≈ decay_rate * 0.693
        With decay_rate=30, t_half ≈ 20.8 days.

    Returns a ranked DataFrame with: user_id, session_id, video_id, rank, score, strategy.
    """
    base = _base_score(session_df)
    age = session_df["item_age_days"].fillna(0).clip(lower=0)
    boosted = base * np.exp(-age / decay_rate)
    return _build_ranked_df(session_df, boosted, strategy="freshness_boosted")


def diversity_aware_rerank(session_df: pd.DataFrame, top_n: int = 20,
                           author_window: int = 2, tag_window: int = 3) -> pd.DataFrame:
    """
    Greedy diversity-aware reranking starting from the popularity_based top-N candidates.

    Algorithm:
    1. Take top_n items by popularity score as candidates.
    2. Greedily build a new ranked list. At each step, pick the highest-scored remaining item
       where author_id has NOT appeared in the last `author_window` slots AND
       tag has NOT appeared in the last `tag_window` slots.
    3. If all remaining items violate both constraints, fall back to the pure score-ordered pick
       to ensure progress (avoids infinite loops on small candidate pools).

    Assumption: `tag` NaN values (imputed to "Unknown" in data_prep) are treated as a valid
    cluster — many "Unknown" items in sequence will trigger the tag diversity constraint.

    Returns a ranked DataFrame with: user_id, session_id, video_id, rank, score, strategy.
    """
    base_ranked = popularity_based(session_df)
    candidates = base_ranked.head(top_n).copy()

    # Merge back the author_id and tag needed for constraint checking
    meta_cols = ["video_id", "author_id", "tag"]
    available = [c for c in meta_cols if c in session_df.columns]
    meta = session_df[available].drop_duplicates("video_id")
    candidates = candidates.merge(meta, on="video_id", how="left")

    reranked = []
    remaining = candidates.copy()

    while len(remaining) > 0:
        # Build recent-window sets from already-placed items
        recent_authors = set(
            r["author_id"] for r in reranked[-author_window:]
            if "author_id" in r and pd.notna(r.get("author_id"))
        )
        recent_tags = set(
            r["tag"] for r in reranked[-tag_window:]
            if "tag" in r and pd.notna(r.get("tag"))
        )

        # Prefer items not violating either constraint
        eligible = remaining[
            (~remaining["author_id"].isin(recent_authors)) &
            (~remaining["tag"].isin(recent_tags))
        ]

        if len(eligible) == 0:
            # Soft fallback: only enforce author constraint (less strict)
            eligible = remaining[~remaining["author_id"].isin(recent_authors)]

        if len(eligible) == 0:
            # Hard fallback: take next by score regardless of constraints
            eligible = remaining

        chosen = eligible.iloc[0]
        reranked.append(chosen.to_dict())
        remaining = remaining[remaining["video_id"] != chosen["video_id"]]

    out = pd.DataFrame(reranked)
    out["rank"] = range(1, len(out) + 1)
    out["strategy"] = "diversity_aware_rerank"
    keep_cols = ["user_id", "session_id", "video_id", "rank", "score", "strategy"]
    return out[[c for c in keep_cols if c in out.columns]].reset_index(drop=True)
