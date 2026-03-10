"""
Module: eval_metrics
Responsibility:
- Implement first-pass evaluation metrics for the 4 DiscoveryRank dimensions:
  relevance, freshness, diversity, and repetition_risk.
- All metrics accept a ranked DataFrame (output from a ranking_strategies function)
  joined with the necessary feature columns from the pipeline.
- Returns are plain Python dicts for easy tabulation and comparison.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


def relevance_score(ranked_df: pd.DataFrame, k: int = None) -> dict:
    """
    Measures how relevant the recommended items are based on observed user behavior.

    Metrics:
    - mean_y_relevant: proportion of items where y_relevant == 1 (explicit positive or >80% completion)
    - mean_completion_ratio: average implicit_completion_ratio (clip already applied upstream)

    Args:
        ranked_df: Ranked session DataFrame (output of a strategy function, joined with pipeline features)
        k: If set, evaluate only the top-k items. If None, evaluate all items.

    Assumption: y_relevant is a binary label derived in relevance_labels.py:
        1 if explicit positive (like/follow/forward/comment) OR completion > 0.8, else 0 (with hate = 0 override).
        This is an offline proxy — I am measuring how many items in the ranked set the user actually engaged with.
    """
    df = ranked_df.sort_values("rank")
    if k is not None:
        df = df.head(k)

    n = len(df)
    if n == 0:
        return {"mean_y_relevant": float("nan"), "mean_completion_ratio": float("nan"), "n_items": 0}

    return {
        "mean_y_relevant": float(df["y_relevant"].mean()),
        "mean_completion_ratio": float(df["implicit_completion_ratio"].mean()),
        "n_items": n,
    }


def freshness_score(ranked_df: pd.DataFrame, fresh_threshold_days: float = 7.0) -> dict:
    """
    Measures how fresh (recently uploaded) the recommended items are.

    Metrics:
    - mean_item_age_days: average age (lower = better; a strategy surfacing older content scores higher here)
    - median_item_age_days: median age (robust to outlier viral very-old videos)
    - fresh_item_ratio: proportion of items newer than `fresh_threshold_days`

    Args:
        ranked_df: Ranked session DataFrame with item_age_days column
        fresh_threshold_days: Items below this age count as "fresh" (default: 7 days)

    Assumption: item_age_days was calculated in freshness_features.py as:
        (interaction_time_ms - upload_time_ms) / ms_per_day
        Missing upload dates were imputed as the first interaction time (age = 0), not as null.
    """
    df = ranked_df.dropna(subset=["item_age_days"])
    n = len(df)
    if n == 0:
        return {
            "mean_item_age_days": float("nan"),
            "median_item_age_days": float("nan"),
            "fresh_item_ratio": float("nan"),
            "n_items": 0,
        }

    return {
        "mean_item_age_days": float(df["item_age_days"].mean()),
        "median_item_age_days": float(df["item_age_days"].median()),
        "fresh_item_ratio": float((df["item_age_days"] < fresh_threshold_days).mean()),
        "n_items": n,
    }


def diversity_score(ranked_df: pd.DataFrame) -> dict:
    """
    Measures categorical breadth within the recommendation set.

    Metrics:
    - unique_author_ratio: proportion of items with a distinct author_id (1.0 = no author repeated)
    - unique_tag_ratio: proportion of items with a distinct tag (1.0 = no tag repeated)
    - author_entropy: Shannon entropy of the author_id distribution (higher = more spread)

    Assumption: tag NaN values were imputed to "Unknown" in data_prep.py — many "Unknown" items
    will REDUCE unique_tag_ratio, which is the correct behavior (opaque clustering should penalize diversity).
    """
    n = len(ranked_df)
    if n == 0:
        return {
            "unique_author_ratio": float("nan"),
            "unique_tag_ratio": float("nan"),
            "author_entropy": float("nan"),
            "n_items": 0,
        }

    unique_authors = ranked_df["author_id"].nunique() if "author_id" in ranked_df.columns else 0
    unique_tags = ranked_df["tag"].nunique() if "tag" in ranked_df.columns else 0

    # Shannon entropy of author distribution
    if "author_id" in ranked_df.columns:
        author_counts = ranked_df["author_id"].value_counts(normalize=True)
        author_ent = float(scipy_entropy(author_counts))
    else:
        author_ent = float("nan")

    return {
        "unique_author_ratio": unique_authors / n,
        "unique_tag_ratio": unique_tags / n,
        "author_entropy": author_ent,
        "n_items": n,
    }


def repetition_risk_score(ranked_df: pd.DataFrame,
                           prior_session_video_ids: set = None) -> dict:
    """
    Measures how much the recommendation set re-exposes the user to redundant content.

    Metrics:
    - consecutive_author_rate: proportion of consecutive position pairs where author_id repeats
    - consecutive_tag_rate: proportion of consecutive position pairs where tag repeats
    - deja_vu_rate: proportion of items the user already consumed in a PRIOR session
      (Only computed if `prior_session_video_ids` is provided — otherwise reported as NaN)

    Args:
        ranked_df: Ranked session DataFrame sorted by rank
        prior_session_video_ids: Optional set of video_ids the user already watched before this session.
            If None, deja_vu_rate is NaN (cannot be computed without stateful user history).

    Assumption: Ranking order is determined by the `rank` column, so the sequence matters here.
    """
    df = ranked_df.sort_values("rank").reset_index(drop=True)
    n = len(df)

    if n < 2:
        return {
            "consecutive_author_rate": float("nan"),
            "consecutive_tag_rate": float("nan"),
            "deja_vu_rate": float("nan"),
            "n_items": n,
        }

    # Consecutive rate helpers
    def _consecutive_rate(series: pd.Series) -> float:
        """Proportion of (i, i+1) pairs where values are equal and not NaN."""
        vals = series.reset_index(drop=True)
        matches = sum(
            v == vals[i + 1]
            for i, v in enumerate(vals[:-1])
            if pd.notna(v) and pd.notna(vals[i + 1])
        )
        return matches / (n - 1)

    consec_author = _consecutive_rate(df["author_id"]) if "author_id" in df.columns else float("nan")
    consec_tag = _consecutive_rate(df["tag"]) if "tag" in df.columns else float("nan")

    # Deja-vu rate (requires optional cross-session history)
    if prior_session_video_ids is not None and "video_id" in df.columns:
        deja_vu = float((df["video_id"].isin(prior_session_video_ids)).mean())
    else:
        deja_vu = float("nan")  # Cannot compute without stateful history

    return {
        "consecutive_author_rate": consec_author,
        "consecutive_tag_rate": consec_tag,
        "deja_vu_rate": deja_vu,
        "n_items": n,
    }


def advanced_discovery_score(ranked_df: pd.DataFrame, item_popularity_dict: dict, total_train_interactions: int) -> dict:
    """
    Measures advanced discovery metrics: Novelty and Serendipity.
    
    Novelty answers: "How long-tail or obscure are the recommended items?"
    Calculated as -log2( P(item) ), where P(item) = item_interactions / total_interactions.
    Higher novelty means the items have fewer global interactions.
    
    Serendipity answers: "How often did the user actually interact with these novel/obscure items?"
    Calculated as Novelty * y_relevant.
    """
    df = ranked_df.dropna(subset=["video_id"]).copy()
    n = len(df)
    
    if n == 0 or not item_popularity_dict or not total_train_interactions:
        return {
            "mean_novelty": float("nan"),
            "mean_serendipity": float("nan"),
        }
        
    # Calculate P(item)
    # Give unseen items a small smoothing value of 1 interaction to avoid log(0)
    item_counts = df["video_id"].map(item_popularity_dict).fillna(1.0)
    p_item = item_counts / total_train_interactions
    
    # Novelty: -log2(P(item))
    # Cap p_item at 1.0 just in case
    p_item = np.clip(p_item, a_min=1e-9, a_max=1.0)
    df["item_novelty"] = -np.log2(p_item)
    
    # Serendipity: Novelty * Relevance
    if "y_relevant" in df.columns:
        df["item_serendipity"] = df["item_novelty"] * df["y_relevant"]
        mean_serendipity = float(df["item_serendipity"].mean())
    else:
        mean_serendipity = float("nan")
        
    return {
        "mean_novelty": float(df["item_novelty"].mean()),
        "mean_serendipity": mean_serendipity
    }


def score_all_metrics(ranked_df: pd.DataFrame, strategy_name: str,
                       k: int = None, prior_video_ids: set = None,
                       item_popularity_dict: dict = None, total_train_interactions: int = None) -> dict:
    """
    Convenience function: compute all metric groups at once and return a single flat dict.
    Intent: used downstream in loops over (user, session, strategy) combos.
    """
    row = {"strategy": strategy_name}
    row.update({f"rel_{k_}": v for k_, v in relevance_score(ranked_df, k=k).items()})
    row.update({f"fresh_{k_}": v for k_, v in freshness_score(ranked_df).items()})
    row.update({f"div_{k_}": v for k_, v in diversity_score(ranked_df).items()})
    row.update({f"rep_{k_}": v for k_, v in repetition_risk_score(ranked_df, prior_video_ids).items()})
    
    if item_popularity_dict is not None and total_train_interactions is not None:
        row.update({f"adv_{k_}": v for k_, v in advanced_discovery_score(ranked_df, item_popularity_dict, total_train_interactions).items()})
        
    return row
