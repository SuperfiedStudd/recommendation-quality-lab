"""
Module: interaction_simulator
Responsibility:
- Simulate user reactions to recommendations using simple stochastic heuristics.
- Produces outcome events compatible with the Event schema.
"""

import numpy as np
from typing import List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.event_schema import Event, EventType


class InteractionSimulator:
    """
    Simulates user behavior in response to a list of recommended items.

    Heuristics:
    - Click probability is boosted if the item's creator is in the user's recent history.
    - Watch probability is boosted if the item's category matches the user's preferences.
    - Watch time follows an exponential distribution scaled by item popularity.
    - Small probability of like or save after watching.
    """

    def __init__(self, base_click_prob: float = 0.3,
                 base_watch_prob: float = 0.5,
                 mean_watch_time: float = 15000.0,
                 like_prob: float = 0.05,
                 save_prob: float = 0.03,
                 seed: int = 42):
        """
        Args:
            base_click_prob: Base probability of clicking a recommended item.
            base_watch_prob: Base probability of watching after clicking.
            mean_watch_time: Mean watch time in milliseconds (exponential distribution).
            like_prob: Probability of liking after watching.
            save_prob: Probability of saving after watching.
            seed: Random seed for reproducibility.
        """
        self.base_click_prob = base_click_prob
        self.base_watch_prob = base_watch_prob
        self.mean_watch_time = mean_watch_time
        self.like_prob = like_prob
        self.save_prob = save_prob
        self.rng = np.random.RandomState(seed)

    def simulate_response(self, user_id: int, item_id: int,
                          session_id: str, timestamp: int,
                          creator_id: Optional[int] = None,
                          category: Optional[str] = None,
                          user_state=None,
                          item_state=None) -> List[Event]:
        """
        Simulate a user's reaction to a single recommended item.

        Returns a list of Event objects representing the interaction chain
        (e.g., impression → click → watch → like).

        Args:
            user_id: The user receiving the recommendation.
            item_id: The recommended item.
            session_id: Current session context.
            timestamp: Current timestamp in epoch ms.
            creator_id: Item's creator ID (for affinity boost).
            category: Item's category (for preference boost).
            user_state: Optional UserState for personalized probabilities.
            item_state: Optional ItemState for popularity-based adjustments.
        """
        events = []

        # --- Always log an impression ---
        events.append(Event(
            user_id=user_id,
            item_id=item_id,
            timestamp=timestamp,
            event_type=EventType.IMPRESSION.value,
            watch_time=0.0,
            creator_id=creator_id,
            category=category,
            session_id=session_id,
        ))

        # --- Click decision ---
        click_prob = self.base_click_prob

        # Boost if creator is in user's recent history
        if user_state is not None and creator_id is not None:
            if creator_id in user_state.recent_creators:
                click_prob = min(click_prob * 1.5, 0.9)

        # Boost if category matches user preferences
        if user_state is not None and category is not None:
            if category in user_state.recent_categories:
                click_prob = min(click_prob * 1.3, 0.9)

        # Slight boost for popular items
        if item_state is not None and item_state.recent_popularity > 50:
            click_prob = min(click_prob * 1.2, 0.9)

        if self.rng.random() > click_prob:
            # No click → skip
            events.append(Event(
                user_id=user_id,
                item_id=item_id,
                timestamp=timestamp + 500,  # half-second later
                event_type=EventType.SKIP.value,
                watch_time=0.0,
                creator_id=creator_id,
                category=category,
                session_id=session_id,
            ))
            return events

        # --- Clicked → log click ---
        events.append(Event(
            user_id=user_id,
            item_id=item_id,
            timestamp=timestamp + 1000,
            event_type=EventType.CLICK.value,
            watch_time=0.0,
            creator_id=creator_id,
            category=category,
            session_id=session_id,
        ))

        # --- Watch decision ---
        watch_prob = self.base_watch_prob
        if user_state is not None and category is not None:
            if category in user_state.recent_categories:
                watch_prob = min(watch_prob * 1.4, 0.95)

        if self.rng.random() < watch_prob:
            watch_time = float(self.rng.exponential(self.mean_watch_time))
            watch_time = max(1000.0, min(watch_time, 120000.0))  # clamp 1s–2min

            events.append(Event(
                user_id=user_id,
                item_id=item_id,
                timestamp=timestamp + 2000,
                event_type=EventType.WATCH.value,
                watch_time=watch_time,
                creator_id=creator_id,
                category=category,
                session_id=session_id,
            ))

            # --- Like / Save decisions ---
            if self.rng.random() < self.like_prob:
                events.append(Event(
                    user_id=user_id,
                    item_id=item_id,
                    timestamp=timestamp + 3000,
                    event_type=EventType.LIKE.value,
                    watch_time=0.0,
                    creator_id=creator_id,
                    category=category,
                    session_id=session_id,
                ))

            if self.rng.random() < self.save_prob:
                events.append(Event(
                    user_id=user_id,
                    item_id=item_id,
                    timestamp=timestamp + 3500,
                    event_type=EventType.SAVE.value,
                    watch_time=0.0,
                    creator_id=creator_id,
                    category=category,
                    session_id=session_id,
                ))

        return events

    def simulate_session(self, user_id: int, recommended_items: list,
                         session_id: str, base_timestamp: int,
                         item_metadata: dict = None,
                         user_state=None,
                         item_states: dict = None) -> List[Event]:
        """
        Simulate a user's reactions to an entire list of recommended items.

        Args:
            user_id: The user.
            recommended_items: List of item IDs in display order.
            session_id: Current session.
            base_timestamp: Starting timestamp.
            item_metadata: Optional dict mapping item_id → {creator_id, category}.
            user_state: Optional UserState for personalized simulation.
            item_states: Optional dict mapping item_id → ItemState.

        Returns:
            List of all outcome Event objects from the session.
        """
        all_events = []
        time_offset = 0

        for item_id in recommended_items:
            meta = (item_metadata or {}).get(item_id, {})
            creator_id = meta.get("creator_id")
            category = meta.get("category")
            istate = (item_states or {}).get(item_id)

            outcomes = self.simulate_response(
                user_id=user_id,
                item_id=item_id,
                session_id=session_id,
                timestamp=base_timestamp + time_offset,
                creator_id=creator_id,
                category=category,
                user_state=user_state,
                item_state=istate,
            )
            all_events.extend(outcomes)
            time_offset += 5000  # 5 seconds between items

        return all_events
