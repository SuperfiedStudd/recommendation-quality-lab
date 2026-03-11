"""
Module: item_state
Responsibility:
- Track per-item state in-memory as events arrive.
- Maintains impression/click counts, watch time, popularity, and freshness.
"""

from collections import deque


class ItemState:
    """
    Lightweight in-memory state for a single item (video).
    Updated incrementally as events are processed.
    """

    def __init__(self, item_id: int, popularity_window: int = 200):
        self.item_id = item_id
        self.impression_count = 0
        self.click_count = 0
        self.watch_time_sum = 0.0
        self.first_seen_time = None
        self.last_seen_time = None

        # Rolling window of recent interaction timestamps for popularity
        self._recent_timestamps = deque(maxlen=popularity_window)

    @property
    def recent_popularity(self) -> int:
        """Number of interactions in the rolling window."""
        return len(self._recent_timestamps)

    @property
    def freshness(self) -> float:
        """
        Time in milliseconds since the item was first seen.
        Returns 0.0 if only seen once or never.
        """
        if self.first_seen_time is None or self.last_seen_time is None:
            return 0.0
        return float(self.last_seen_time - self.first_seen_time)

    def update(self, event) -> None:
        """
        Update item state from a single Event.

        Args:
            event: An Event dataclass instance.
        """
        # Track first/last seen
        if self.first_seen_time is None:
            self.first_seen_time = event.timestamp
        self.last_seen_time = event.timestamp

        # Count impressions (every event is at minimum an impression)
        self.impression_count += 1
        self._recent_timestamps.append(event.timestamp)

        # Count clicks
        if event.event_type in ("click", "like", "save"):
            self.click_count += 1

        # Accumulate watch time
        if event.watch_time > 0:
            self.watch_time_sum += event.watch_time

    def to_dict(self) -> dict:
        """Snapshot of current state as a plain dict."""
        return {
            "item_id": self.item_id,
            "impression_count": self.impression_count,
            "click_count": self.click_count,
            "watch_time_sum": self.watch_time_sum,
            "recent_popularity": self.recent_popularity,
            "freshness": self.freshness,
        }
