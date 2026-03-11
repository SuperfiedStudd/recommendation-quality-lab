"""
Module: policy_backtest
Responsibility:
- Run the full recommendation loop (replay → serve → simulate → log → evaluate)
  for multiple ranking policies.
- Produce a comparison DataFrame of metrics across policies.
"""

import pandas as pd
import numpy as np
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.event_replay import load_interactions, replay_stream
from data.event_schema import Event, EventType
from features.state_manager import StateManager
from serving.recommender_service import RecommenderService
from simulation.interaction_simulator import InteractionSimulator
from logging_layer.exposure_logger import ExposureLogger
from logging_layer.outcome_logger import OutcomeLogger
from evaluation.metrics_extensions import compute_all_loop_metrics


class PolicyBacktest:
    """
    Runs the end-to-end recommendation loop for multiple ranking policies
    and produces a side-by-side metric comparison.

    Loop per policy:
        1. Replay historical events to build initial state
        2. For sampled (user, session) pairs, generate recommendations
        3. Simulate user outcomes
        4. Log exposures and outcomes
        5. Compute metrics
    """

    def __init__(self, master_df: pd.DataFrame,
                 policies: dict = None,
                 n_users: int = 50,
                 n_sessions_per_user: int = 3,
                 k: int = 20,
                 warmup_events: int = 5000,
                 seed: int = 42):
        """
        Args:
            master_df: Full pipeline DataFrame.
            policies: Dict mapping policy_name → strategy key.
                      Defaults to popularity, freshness_boosted, diversity_aware.
            n_users: Number of users to sample for evaluation.
            n_sessions_per_user: Number of sessions per user to evaluate.
            k: Number of recommendations per request.
            warmup_events: Number of replay events to process before evaluation.
            seed: Random seed for reproducibility.
        """
        self.master_df = master_df
        self.policies = policies or {
            "popularity": "popularity",
            "recency_decay": "freshness_boosted",
            "hybrid": "diversity_aware",
        }
        self.n_users = n_users
        self.n_sessions_per_user = n_sessions_per_user
        self.k = k
        self.warmup_events = warmup_events
        self.seed = seed
        self.results = {}

    def _build_item_metadata(self) -> dict:
        """Build item_id → {creator_id, category} lookup from master data."""
        meta = {}
        for _, row in self.master_df[["video_id", "author_id", "tag"]].drop_duplicates("video_id").iterrows():
            meta[int(row["video_id"])] = {
                "creator_id": int(row["author_id"]) if pd.notna(row["author_id"]) else None,
                "category": str(row["tag"]) if pd.notna(row["tag"]) else "Unknown",
            }
        return meta

    def _sample_user_sessions(self) -> list:
        """
        Sample (user_id, session_id) pairs for evaluation.
        Picks users with enough sessions to be meaningful.
        """
        rng = np.random.RandomState(self.seed)

        session_counts = self.master_df.groupby("user_id")["session_id"].nunique()
        eligible_users = session_counts[
            session_counts >= self.n_sessions_per_user
        ].index.tolist()

        if len(eligible_users) == 0:
            # Fallback: use all users with at least 1 session
            eligible_users = session_counts[session_counts >= 1].index.tolist()

        n_sample = min(self.n_users, len(eligible_users))
        sampled_users = rng.choice(eligible_users, size=n_sample, replace=False)

        pairs = []
        for uid in sampled_users:
            user_sessions = self.master_df[
                self.master_df["user_id"] == uid
            ]["session_id"].unique()

            n_sess = min(self.n_sessions_per_user, len(user_sessions))
            chosen = rng.choice(user_sessions, size=n_sess, replace=False)
            for sid in chosen:
                pairs.append((int(uid), str(sid)))

        return pairs

    def run(self) -> pd.DataFrame:
        """
        Execute the backtest loop for all policies.

        Returns:
            DataFrame with one row per policy and columns for each metric.
        """
        print(f"PolicyBacktest: sampling user-session pairs...")
        eval_pairs = self._sample_user_sessions()
        print(f"  → {len(eval_pairs)} (user, session) pairs selected")

        item_metadata = self._build_item_metadata()
        total_items = self.master_df["video_id"].nunique()

        all_results = []

        for policy_name, strategy_key in self.policies.items():
            print(f"\n{'='*50}")
            print(f"Running policy: {policy_name} (strategy={strategy_key})")
            print(f"{'='*50}")

            # Fresh state and loggers per policy
            state_mgr = StateManager()
            exposure_logger = ExposureLogger()
            outcome_logger = OutcomeLogger()
            simulator = InteractionSimulator(seed=self.seed)

            # Initialize recommender service
            service = RecommenderService(
                master_df=self.master_df,
                state_manager=state_mgr,
                strategy=strategy_key,
            )

            # --- Warmup: replay some events to populate state ---
            print(f"  Warming up state with {self.warmup_events} events...")
            warmup_df = self.master_df.sort_values("time_ms").head(self.warmup_events)
            for _, row in warmup_df.iterrows():
                evt = Event(
                    user_id=int(row["user_id"]),
                    item_id=int(row["video_id"]),
                    timestamp=int(row["time_ms"]),
                    event_type="watch" if row.get("play_time_ms", 0) > 0 else "skip",
                    watch_time=float(row.get("play_time_ms", 0)),
                    creator_id=int(row["author_id"]) if pd.notna(row.get("author_id")) else None,
                    category=str(row["tag"]) if pd.notna(row.get("tag")) else None,
                    session_id=str(row["session_id"]) if pd.notna(row.get("session_id")) else None,
                )
                state_mgr.process_event(evt)

            print(f"  State: {state_mgr.summary()}")

            # --- Evaluation loop ---
            print(f"  Evaluating {len(eval_pairs)} sessions...")
            for i, (user_id, session_id) in enumerate(eval_pairs):
                # Get recommendations
                recs = service.recommend(user_id, session_id, k=self.k)
                if recs.empty:
                    continue

                rec_items = recs["video_id"].tolist()
                rec_ranks = recs["rank"].tolist()

                # Determine timestamp context
                session_data = self.master_df[
                    (self.master_df["user_id"] == user_id) &
                    (self.master_df["session_id"] == session_id)
                ]
                if session_data.empty:
                    continue
                base_ts = int(session_data["time_ms"].min())

                # Log exposure
                exposure_logger.log(
                    timestamp=base_ts,
                    user_id=user_id,
                    session_id=session_id,
                    recommended_items=rec_items,
                    rank_positions=rec_ranks,
                )

                # Simulate outcomes
                user_state = state_mgr.get_user_state(user_id)
                outcomes = simulator.simulate_session(
                    user_id=user_id,
                    recommended_items=rec_items,
                    session_id=session_id,
                    base_timestamp=base_ts,
                    item_metadata=item_metadata,
                    user_state=user_state,
                    item_states=state_mgr.item_states,
                )

                # Log outcomes and update state
                for evt in outcomes:
                    outcome_logger.log(
                        timestamp=evt.timestamp,
                        user_id=evt.user_id,
                        item_id=evt.item_id,
                        event_type=evt.event_type,
                        watch_time=evt.watch_time,
                        session_id=evt.session_id,
                    )
                    state_mgr.process_event(evt)

            # --- Compute metrics ---
            exposure_flat = exposure_logger.to_flat_dataframe()
            outcome_df = outcome_logger.to_dataframe()

            metrics = compute_all_loop_metrics(
                exposure_df=exposure_flat,
                outcome_df=outcome_df,
                total_items=total_items,
                item_metadata=item_metadata,
            )
            metrics["policy"] = policy_name
            metrics["strategy"] = strategy_key
            metrics["n_sessions_evaluated"] = len(eval_pairs)
            metrics["n_exposures"] = len(exposure_logger)
            metrics["n_outcomes"] = len(outcome_logger)

            all_results.append(metrics)

            # Store loggers for potential export
            self.results[policy_name] = {
                "metrics": metrics,
                "exposure_logger": exposure_logger,
                "outcome_logger": outcome_logger,
            }

            print(f"  Metrics: {metrics}")

        comparison_df = pd.DataFrame(all_results)
        return comparison_df

    def save_logs(self, output_dir: str = "outputs/logs") -> None:
        """Save all exposure and outcome logs per policy."""
        for policy_name, data in self.results.items():
            safe_name = policy_name.replace(" ", "_")
            data["exposure_logger"].save(
                os.path.join(output_dir, f"exposure_{safe_name}.csv")
            )
            data["outcome_logger"].save(
                os.path.join(output_dir, f"outcome_{safe_name}.csv")
            )
        print(f"Logs saved to {output_dir}")

    def save_comparison(self, path: str = "outputs/policy_comparison.csv") -> None:
        """Save the metric comparison table."""
        if self.results:
            rows = [d["metrics"] for d in self.results.values()]
            df = pd.DataFrame(rows)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Comparison saved to {path}")
