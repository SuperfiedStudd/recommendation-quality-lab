"""
Module: exposure_logger
Responsibility:
- Log each recommendation exposure (what was shown to the user).
- Store logs as pandas DataFrames, saveable to CSV.
"""

import pandas as pd
import os
from typing import List


class ExposureLogger:
    """
    Logs recommendation exposures.

    Each log entry captures what items were recommended to a user,
    at what rank positions, and when.
    """

    def __init__(self):
        self._records = []

    def log(self, timestamp: int, user_id: int, session_id: str,
            recommended_items: List[int], rank_positions: List[int]) -> None:
        """
        Record a single exposure event.

        Args:
            timestamp: UNIX epoch milliseconds when recommendations were served.
            user_id: The user who received the recommendations.
            session_id: The session context.
            recommended_items: List of item IDs that were recommended.
            rank_positions: Corresponding rank positions (1-indexed).
        """
        self._records.append({
            "timestamp": timestamp,
            "user_id": user_id,
            "session_id": session_id,
            "recommended_items": recommended_items,
            "rank_positions": rank_positions,
            "n_items": len(recommended_items),
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Return all logged exposures as a DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=["timestamp", "user_id", "session_id",
                                         "recommended_items", "rank_positions", "n_items"])
        return pd.DataFrame(self._records)

    def to_flat_dataframe(self) -> pd.DataFrame:
        """
        Return a flattened DataFrame with one row per (user, item, rank).
        More useful for metric computation than the nested list format.
        """
        rows = []
        for rec in self._records:
            for item_id, rank in zip(rec["recommended_items"], rec["rank_positions"]):
                rows.append({
                    "timestamp": rec["timestamp"],
                    "user_id": rec["user_id"],
                    "session_id": rec["session_id"],
                    "item_id": item_id,
                    "rank_position": rank,
                })
        if not rows:
            return pd.DataFrame(columns=["timestamp", "user_id", "session_id",
                                         "item_id", "rank_position"])
        return pd.DataFrame(rows)

    def save(self, path: str = "outputs/logs/exposure_log.csv") -> str:
        """
        Save the flattened exposure log to CSV.

        Args:
            path: Output file path.

        Returns:
            The path the file was saved to.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.to_flat_dataframe()
        df.to_csv(path, index=False)
        return path

    def __len__(self) -> int:
        return len(self._records)
