"""
Module: user_state
Responsibility:
- Track per-user state in-memory as events arrive.
- Maintains recent creators, categories, average watch time,
  session depth, and last active time.
"""

from collections import deque
from typing import Optional


class UserState:
    """
    Lightweight in-memory state for a single user.
    Updated incrementally as events are processed.
    """

    def __init__(self, user_id: int, history_size: int = 50):
        self.user_id = user_id
        self.recent_creators = deque(maxlen=history_size)
        self.recent_categories = deque(maxlen=history_size)
        self.avg_watch_time = 0.0
        self.session_depth = 0
        self.last_active_time = 0
        self._total_watch_time = 0.0
        self._interaction_count = 0

    def update(self, event) -> None:
        """
        Update user state from a single Event.

        Args:
            event: An Event dataclass instance.
        """
        # Track recent creators and categories
        if event.creator_id is not None:
            self.recent_creators.append(event.creator_id)
        if event.category is not None:
            self.recent_categories.append(event.category)

        # Update running average watch time (cumulative mean)
        if event.watch_time > 0:
            self._total_watch_time += event.watch_time
            self._interaction_count += 1
            self.avg_watch_time = self._total_watch_time / self._interaction_count

        # Session depth increments per interaction
        self.session_depth += 1

        # Track latest activity
        if event.timestamp > self.last_active_time:
            self.last_active_time = event.timestamp

    def reset_session(self) -> None:
        """Reset session-specific counters (e.g., when a new session starts)."""
        self.session_depth = 0

    def to_dict(self) -> dict:
        """Snapshot of current state as a plain dict."""
        return {
            "user_id": self.user_id,
            "recent_creators": list(self.recent_creators),
            "recent_categories": list(self.recent_categories),
            "avg_watch_time": self.avg_watch_time,
            "session_depth": self.session_depth,
            "last_active_time": self.last_active_time,
        }
