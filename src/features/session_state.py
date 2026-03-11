"""
Module: session_state
Responsibility:
- Track per-session state in-memory as events arrive.
- Maintains items seen and interaction count for the session.
"""


class SessionState:
    """
    Lightweight in-memory state for a single session.
    Updated incrementally as events are processed.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.items_seen = set()
        self.interaction_count = 0

    def update(self, event) -> None:
        """
        Update session state from a single Event.

        Args:
            event: An Event dataclass instance.
        """
        self.items_seen.add(event.item_id)
        self.interaction_count += 1

    def to_dict(self) -> dict:
        """Snapshot of current state as a plain dict."""
        return {
            "session_id": self.session_id,
            "items_seen": list(self.items_seen),
            "interaction_count": self.interaction_count,
        }
