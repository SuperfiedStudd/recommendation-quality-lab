"""
Module: metrics_extensions
Responsibility:
- Provide online-loop-specific evaluation metrics beyond the existing
  offline eval_metrics.py.
- CTR proxy, watch_time proxy, catalog coverage, diversity, creator spread.
"""

import pandas as pd
import numpy as np


def ctr_proxy(outcome_df: pd.DataFrame) -> float:
    """
    Click-through rate proxy: clicks / total impressions.

    Args:
        outcome_df: DataFrame with 'event_type' column.

    Returns:
        CTR as a float between 0 and 1.
    """
    if outcome_df.empty:
        return 0.0
    impressions = (outcome_df["event_type"] == "impression").sum()
    clicks = (outcome_df["event_type"] == "click").sum()
    if impressions == 0:
        return 0.0
    return float(clicks / impressions)


def watch_time_proxy(outcome_df: pd.DataFrame) -> float:
    """
    Average watch time per impression (ms).

    Args:
        outcome_df: DataFrame with 'event_type' and 'watch_time' columns.

    Returns:
        Mean watch time per impression in milliseconds.
    """
    if outcome_df.empty:
        return 0.0
    impressions = (outcome_df["event_type"] == "impression").sum()
    total_watch = outcome_df["watch_time"].sum()
    if impressions == 0:
        return 0.0
    return float(total_watch / impressions)


def catalog_coverage(exposure_df: pd.DataFrame, total_items: int) -> float:
    """
    Fraction of the total item catalog that was recommended at least once.

    Args:
        exposure_df: Flattened exposure DataFrame with 'item_id' column.
        total_items: Total number of items in the catalog.

    Returns:
        Coverage ratio between 0 and 1.
    """
    if exposure_df.empty or total_items == 0:
        return 0.0
    unique_shown = exposure_df["item_id"].nunique()
    return float(unique_shown / total_items)


def diversity_metric(exposure_df: pd.DataFrame,
                     item_metadata: dict = None) -> float:
    """
    Average number of unique categories per exposure set (session).

    Args:
        exposure_df: Flattened exposure DataFrame with 'item_id' and 'session_id'.
        item_metadata: Optional dict mapping item_id → {category: str}.

    Returns:
        Mean unique categories per session.
    """
    if exposure_df.empty:
        return 0.0

    if item_metadata is not None and "category" not in exposure_df.columns:
        exposure_df = exposure_df.copy()
        exposure_df["category"] = exposure_df["item_id"].map(
            lambda x: item_metadata.get(x, {}).get("category", "Unknown")
        )

    if "category" not in exposure_df.columns:
        return 0.0

    per_session = exposure_df.groupby("session_id")["category"].nunique()
    return float(per_session.mean())


def creator_spread(exposure_df: pd.DataFrame,
                   item_metadata: dict = None) -> float:
    """
    Ratio of unique creators to total impressions.
    Higher = more diverse creator representation.

    Args:
        exposure_df: Flattened exposure DataFrame with 'item_id'.
        item_metadata: Optional dict mapping item_id → {creator_id: int}.

    Returns:
        Creator spread ratio between 0 and 1.
    """
    if exposure_df.empty:
        return 0.0

    if item_metadata is not None and "creator_id" not in exposure_df.columns:
        exposure_df = exposure_df.copy()
        exposure_df["creator_id"] = exposure_df["item_id"].map(
            lambda x: item_metadata.get(x, {}).get("creator_id")
        )

    if "creator_id" not in exposure_df.columns:
        return 0.0

    unique_creators = exposure_df["creator_id"].dropna().nunique()
    total = len(exposure_df)
    if total == 0:
        return 0.0
    return float(unique_creators / total)


def compute_all_loop_metrics(exposure_df: pd.DataFrame,
                             outcome_df: pd.DataFrame,
                             total_items: int,
                             item_metadata: dict = None) -> dict:
    """
    Convenience function: compute all online-loop metrics at once.

    Args:
        exposure_df: Flattened exposure log DataFrame.
        outcome_df: Outcome log DataFrame.
        total_items: Total items in catalog.
        item_metadata: Optional dict mapping item_id → {creator_id, category}.

    Returns:
        Dict of metric_name → value.
    """
    return {
        "ctr_proxy": ctr_proxy(outcome_df),
        "watch_time_proxy_ms": watch_time_proxy(outcome_df),
        "catalog_coverage": catalog_coverage(exposure_df, total_items),
        "diversity": diversity_metric(exposure_df, item_metadata),
        "creator_spread": creator_spread(exposure_df, item_metadata),
    }
