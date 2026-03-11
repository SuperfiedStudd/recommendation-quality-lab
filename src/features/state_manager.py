"""
Module: state_manager
Responsibility:
- Central coordinator that routes incoming events to UserState, ItemState,
  and SessionState stores.
- Provides lookup methods for the serving layer.
"""

from features.user_state import UserState
from features.item_state import ItemState
from features.session_state import SessionState


class StateManager:
    """
    Manages all in-memory state stores and routes events to the correct state objects.
    """

    def __init__(self):
        self.user_states = {}      # user_id → UserState
        self.item_states = {}      # item_id → ItemState
        self.session_states = {}   # session_id → SessionState

    def process_event(self, event) -> None:
        """
        Route a single Event to all three state stores.
        Creates new state objects on first encounter.

        Args:
            event: An Event dataclass instance.
        """
        # --- User state ---
        if event.user_id not in self.user_states:
            self.user_states[event.user_id] = UserState(event.user_id)
        self.user_states[event.user_id].update(event)

        # --- Item state ---
        if event.item_id not in self.item_states:
            self.item_states[event.item_id] = ItemState(event.item_id)
        self.item_states[event.item_id].update(event)

        # --- Session state ---
        if event.session_id is not None:
            if event.session_id not in self.session_states:
                self.session_states[event.session_id] = SessionState(event.session_id)
            self.session_states[event.session_id].update(event)

    def get_user_state(self, user_id: int) -> UserState:
        """Retrieve UserState, creating a default if not yet seen."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState(user_id)
        return self.user_states[user_id]

    def get_item_state(self, item_id: int) -> ItemState:
        """Retrieve ItemState, creating a default if not yet seen."""
        if item_id not in self.item_states:
            self.item_states[item_id] = ItemState(item_id)
        return self.item_states[item_id]

    def get_session_state(self, session_id: str) -> SessionState:
        """Retrieve SessionState, creating a default if not yet seen."""
        if session_id not in self.session_states:
            self.session_states[session_id] = SessionState(session_id)
        return self.session_states[session_id]

    def summary(self) -> dict:
        """Quick summary of state store sizes."""
        return {
            "n_users": len(self.user_states),
            "n_items": len(self.item_states),
            "n_sessions": len(self.session_states),
        }
