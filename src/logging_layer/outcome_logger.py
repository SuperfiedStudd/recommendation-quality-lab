"""
Module: outcome_logger
Responsibility:
- Log user interaction outcomes (what the user did after seeing recommendations).
- Store logs as pandas DataFrames, saveable to CSV.
"""

import pandas as pd
import os


class OutcomeLogger:
    """
    Logs user interaction outcomes following recommendation exposures.

    Each log entry captures a single user action on a single item.
    """

    def __init__(self):
        self._records = []

    def log(self, timestamp: int, user_id: int, item_id: int,
            event_type: str, watch_time: float = 0.0,
            session_id: str = None) -> None:
        """
        Record a single outcome event.

        Args:
            timestamp: UNIX epoch milliseconds when the interaction happened.
            user_id: The user who interacted.
            item_id: The item they interacted with.
            event_type: Type of interaction (click, watch, like, skip, save).
            watch_time: Watch duration in milliseconds (0 for non-watch events).
            session_id: Optional session context.
        """
        self._records.append({
            "timestamp": timestamp,
            "user_id": user_id,
            "item_id": item_id,
            "event_type": event_type,
            "watch_time": watch_time,
            "session_id": session_id,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Return all logged outcomes as a DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=["timestamp", "user_id", "item_id",
                                         "event_type", "watch_time", "session_id"])
        return pd.DataFrame(self._records)

    def save(self, path: str = "outputs/logs/outcome_log.csv") -> str:
        """
        Save the outcome log to CSV.

        Args:
            path: Output file path.

        Returns:
            The path the file was saved to.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path

    def __len__(self) -> int:
        return len(self._records)
