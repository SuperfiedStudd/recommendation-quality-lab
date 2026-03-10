"""
Module: candidate_generation
Responsibility:
- Given a dataframe of master interaction logs, user_id, and session_id:
  generate a merged pool of candidates (observed + synthetic) strictly
  avoiding future leaks.
"""

import pandas as pd
import numpy as np


class SessionCandidateGenerator:
    """
    Generates time-aware candidate pools for a given session.
    Mixes actual observed session items with historical popular items,
    items from historically engaged creators, and random exploration items.
    """

    def __init__(self, master_df: pd.DataFrame):
        """
        Args:
            master_df: The full prepared pipeline dataframe (loaded from pipeline_sample.csv).
                Must contain session_id, user_id, time_ms, video_id, and features.
        """
        # We need the dataframe sorted chronologically to safely subset past data
        self.df = master_df.sort_values("time_ms").copy()

        # Cache item metadata that is static (doesn't leak future engagement)
        meta_cols = ["video_id", "author_id", "tag"]
        available_meta = [c for c in meta_cols if c in self.df.columns]
        self.item_meta = self.df[available_meta].drop_duplicates("video_id")

    def generate_pool(self, user_id: int, session_id: str, pool_size: int = 100) -> pd.DataFrame:
        """
        Generate a candidate pool for the target session.

        Logic:
        1. Find session start time.
        2. Filter full dataset to strictly before this start time (avoids future leaks).
        3. Identify actual observed items in the session.
        4. Fill the rest of the pool with candidates derived ONLY from the past data.
        5. Return a unified dataframe with assumed-negative labels where appropriate.
        """
        # --- 1. Isolate the target session ---
        session_mask = (self.df["user_id"] == user_id) & (self.df["session_id"] == session_id)
        session_df = self.df[session_mask].copy()

        if session_df.empty:
            raise ValueError(f"Session {session_id} for user {user_id} not found in master data.")

        session_start_ms = session_df["time_ms"].min()

        # Flag observed items natively
        session_df["is_observed_in_session"] = 1

        # --- 2. Filter historical data strictly before this session ---
        # Note: we use ALL users' past interactions here to compute popularity,
        # and this user's specific past interactions for history.
        past_df = self.df[self.df["time_ms"] < session_start_ms]

        candidates_needed = pool_size - len(session_df)

        if candidates_needed <= 0:
            # Session is already huge, return as-is
            return session_df

        synthetic_vids = set()
        source_map = {}  # video_id -> source string
        observed_vids = set(session_df["video_id"].tolist())

        # If we have no past data at all (e.g., very first session in the dataset globally),
        # we can only fall back to random sampling from the known universe (assuming universe is static).
        # In a strict real-world pipeline, we'd only sample items uploaded before session_start.
        if past_df.empty:
            # Fallback: purely random from the full item universe. Not ideal but prevents crashing
            # on the absolute earliest timestamp in the sample.
            all_vids = set(self.item_meta["video_id"]) - observed_vids
            fallback_sample = np.random.choice(list(all_vids), size=min(candidates_needed, len(all_vids)), replace=False)
            for vid in fallback_sample:
                synthetic_vids.add(vid)
                source_map[vid] = "random"

        else:
            # --- 3a. Source > Popularity (Global past 7 days) ---
            # Using 7 days = 7 * 24 * 60 * 60 * 1000 ms
            seven_days_ms = 7 * 24 * 60 * 60 * 1000
            recent_past = past_df[past_df["time_ms"] >= (session_start_ms - seven_days_ms)]

            # Define popularity loosely as raw interaction volume + likes in the recent window
            # (In KuaiRand, almost everything is an impression, so count is proxy for impressions)
            if not recent_past.empty:
                pop_scores = recent_past.groupby("video_id").size()
                top_popular = pop_scores.nlargest(candidates_needed).index.tolist()
                for vid in top_popular:
                    if vid not in observed_vids and len(synthetic_vids) < (candidates_needed * 0.5):
                        synthetic_vids.add(vid)
                        source_map[vid] = "popular"

            # --- 3b. Source > User History Adjacency ---
            user_past = past_df[past_df["user_id"] == user_id]
            if not user_past.empty:
                # Find authors they liked or watched heavily before
                positive_past = user_past[(user_past["explicit_positive_any"] == 1) |
                                          (user_past["implicit_completion_ratio"] > 0.8)]
                past_authors = set(positive_past["author_id"].dropna().unique())

                if past_authors:
                    # Find other videos by these authors in the global past pool
                    # that the user hasn't seen yet.
                    history_pool = past_df[past_df["author_id"].isin(past_authors)]
                    history_vids = history_pool["video_id"].unique()
                    # Shuffle to pick random history-adjacent items
                    np.random.shuffle(history_vids)

                    for vid in history_vids:
                        if vid not in observed_vids and vid not in synthetic_vids:
                            synthetic_vids.add(vid)
                            source_map[vid] = "history"
                            if len(synthetic_vids) >= (candidates_needed * 0.8):
                                break

            # --- 3c. Source > Random Exploration ---
            # Fill the rest with random items from the past universe that we haven't picked yet
            past_universe_vids = set(past_df["video_id"].unique())
            remaining_needed = candidates_needed - len(synthetic_vids)

            if remaining_needed > 0:
                available_random = list(past_universe_vids - observed_vids - synthetic_vids)
                if available_random:
                    random_sample = np.random.choice(
                        available_random,
                        size=min(remaining_needed, len(available_random)),
                        replace=False
                    )
                    for vid in random_sample:
                        synthetic_vids.add(vid)
                        source_map[vid] = "random"

        # --- 4. Package synthetic candidates ---
        session_df["candidate_source"] = "observed"

        if not synthetic_vids:
            return session_df

        synth_df = pd.DataFrame({
            "video_id": list(synthetic_vids),
            "candidate_source": [source_map[vid] for vid in synthetic_vids]
        })

        # Attach standard session tracking context
        synth_df["user_id"] = user_id
        synth_df["session_id"] = session_id
        # Assign them to the exact start time of the session so downstream features (like item_age) calculate correctly
        synth_df["time_ms"] = session_start_ms

        # Impute missing engagement labels (Assumed Negatives)
        synth_df["is_observed_in_session"] = 0
        synth_df["y_relevant"] = 0
        synth_df["implicit_completion_ratio"] = 0.0
        synth_df["explicit_positive_any"] = 0
        synth_df["explicit_negative"] = 0
        # Optional: We could impute is_click = 0, is_like = 0, etc. if needed by strategies.
        # But strategies should only rely on the aggregate features strictly.

        # Merge metadata (author_id, tag)
        synth_df = synth_df.merge(self.item_meta, on="video_id", how="left")

        # --- 5. Unify and recalculate strictly-time-dependent features (item_age_days) ---
        pool_df = pd.concat([session_df, synth_df], ignore_index=True)

        # We must recalculate item_age_days for the synthetic items relative to the session start time.
        # We rely on 'upload_time_ms' being statically merged if available, mapped from master_df
        if "upload_time_ms" in self.df.columns:
            upload_map = self.df[["video_id", "upload_time_ms"]].drop_duplicates("video_id")
            # Drop existing if it came along for the ride in session_df, to cleanly merge
            if "upload_time_ms" in pool_df.columns:
                pool_df = pool_df.drop(columns=["upload_time_ms"])
            pool_df = pool_df.merge(upload_map, on="video_id", how="left")

            # Fallback for completely missing uploads: assume it's relatively fresh but add uniform noise
            # (0 to 7 days) so they don't all tie at exactly age=0 and perfectly preserve popularity sort order
            missing_mask = pool_df["upload_time_ms"].isna()
            if missing_mask.any():
                noise_ms = np.random.uniform(0, 7 * 24 * 60 * 60 * 1000, size=missing_mask.sum())
                pool_df.loc[missing_mask, "upload_time_ms"] = pool_df.loc[missing_mask, "time_ms"] - noise_ms

            age_ms = pool_df["time_ms"] - pool_df["upload_time_ms"]
            pool_df["item_age_days"] = age_ms.clip(lower=0) / (1000.0 * 60.0 * 60.0 * 24.0)

        # Ensure types (especially float for label columns so metrics don't break on nans implicitly)
        pool_df["implicit_completion_ratio"] = pool_df["implicit_completion_ratio"].fillna(0.0)

        return pool_df

