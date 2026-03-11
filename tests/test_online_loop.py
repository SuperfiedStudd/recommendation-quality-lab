"""
Smoke test for the online recommendation loop.
Verifies all modules import, wire together, and produce expected outputs.
"""

import sys
import os

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np


def create_test_data(n_rows=200):
    """Create a small synthetic dataset matching pipeline_sample.csv schema."""
    rng = np.random.RandomState(42)
    n_users = 10
    n_items = 50
    n_authors = 20
    tags = ["Music", "Dance", "Comedy", "Food", "Tech", "Unknown"]

    users = rng.randint(1, n_users + 1, size=n_rows)
    items = rng.randint(1, n_items + 1, size=n_rows)
    authors = rng.randint(1, n_authors + 1, size=n_rows)
    base_time = 1650000000000  # ~Apr 2022 in ms

    df = pd.DataFrame({
        "user_id": users,
        "video_id": items,
        "date": 20220415,
        "hourmin": rng.randint(0, 2400, size=n_rows),
        "time_ms": np.int64(base_time) + np.sort(rng.randint(0, 86400000, size=n_rows).astype(np.int64)),
        "is_click": rng.binomial(1, 0.3, size=n_rows),
        "is_like": rng.binomial(1, 0.05, size=n_rows),
        "is_follow": rng.binomial(1, 0.02, size=n_rows),
        "is_comment": rng.binomial(1, 0.03, size=n_rows),
        "is_forward": rng.binomial(1, 0.02, size=n_rows),
        "is_hate": rng.binomial(1, 0.01, size=n_rows),
        "long_view": rng.binomial(1, 0.4, size=n_rows),
        "play_time_ms": rng.exponential(10000, size=n_rows).astype(int),
        "duration_ms": rng.randint(5000, 60000, size=n_rows),
        "profile_stay_time": 0,
        "comment_stay_time": 0,
        "is_profile_enter": 0,
        "is_rand": 0,
        "tab": 1,
        "author_id": authors,
        "video_type": "NORMAL",
        "upload_dt": "2022-04-10",
        "upload_type": "normal",
        "visible_status": 1.0,
        "video_duration": rng.randint(5, 60, size=n_rows).astype(float),
        "server_width": 720.0,
        "server_height": 1280.0,
        "music_id": rng.randint(1, 100, size=n_rows),
        "music_type": 1.0,
        "tag": [tags[i % len(tags)] for i in range(n_rows)],
    })

    # Derived columns (matching pipeline_sample.csv)
    df["explicit_positive_any"] = df[["is_like", "is_follow", "is_forward", "is_comment"]].max(axis=1)
    df["explicit_negative"] = df["is_hate"]
    df["implicit_completion_ratio"] = np.clip(df["play_time_ms"] / df["duration_ms"], 0, 2.0)
    df["y_relevant"] = np.where(
        df["explicit_negative"] == 1, 0,
        np.where(df["explicit_positive_any"] == 1, 1,
                 np.where(df["implicit_completion_ratio"] > 0.8, 1, 0))
    )

    # Session IDs
    df = df.sort_values(["user_id", "time_ms"]).reset_index(drop=True)
    df["session_id"] = df["user_id"].astype(str) + "_1"

    # Freshness
    upload_ms = pd.to_datetime("2022-04-10").timestamp() * 1000
    df["upload_time_ms"] = upload_ms
    df["item_age_ms"] = df["time_ms"] - df["upload_time_ms"]
    df["item_age_days"] = df["item_age_ms"] / (1000 * 60 * 60 * 24)

    return df


def test_event_schema():
    print("Testing event_schema...")
    from data.event_schema import Event, EventType

    evt = Event(user_id=1, item_id=100, timestamp=1650000000000,
                event_type="click", watch_time=5000.0,
                creator_id=10, category="Music", session_id="1_1")
    assert evt.validate()
    d = evt.to_dict()
    evt2 = Event.from_dict(d)
    assert evt2.user_id == 1
    assert evt2.event_type == "click"
    print("  [OK] Event dataclass, serialization, validation OK")


def test_event_replay(test_csv):
    print("Testing event_replay...")
    from data.event_replay import load_interactions, generate_events, replay_stream

    df = load_interactions(test_csv)
    assert len(df) > 0
    print(f"  [OK] Loaded {len(df)} rows")

    events = generate_events(df.head(20))
    assert len(events) == 20
    assert events[0].user_id > 0
    print(f"  [OK] Generated {len(events)} events")

    stream = list(replay_stream(test_csv, max_events=10))
    assert len(stream) == 10
    # Verify sorted by timestamp
    timestamps = [e.timestamp for e in stream]
    assert timestamps == sorted(timestamps)
    print(f"  [OK] Replay stream yields {len(stream)} sorted events")


def test_state_layer():
    print("Testing state layer...")
    from features.state_manager import StateManager
    from data.event_schema import Event

    mgr = StateManager()
    events = [
        Event(user_id=1, item_id=10, timestamp=1000, event_type="click",
              watch_time=5000, creator_id=100, category="Music", session_id="1_1"),
        Event(user_id=1, item_id=20, timestamp=2000, event_type="watch",
              watch_time=10000, creator_id=200, category="Dance", session_id="1_1"),
        Event(user_id=2, item_id=10, timestamp=3000, event_type="skip",
              watch_time=0, creator_id=100, category="Music", session_id="2_1"),
    ]
    for e in events:
        mgr.process_event(e)

    summary = mgr.summary()
    assert summary["n_users"] == 2
    assert summary["n_items"] == 2
    assert summary["n_sessions"] == 2

    u1 = mgr.get_user_state(1)
    assert u1.avg_watch_time == 7500.0
    assert 100 in u1.recent_creators
    print(f"  [OK] StateManager: {summary}")
    print(f"  [OK] UserState avg_watch_time={u1.avg_watch_time}")


def test_recommender_service(test_df):
    print("Testing recommender_service...")
    from serving.recommender_service import RecommenderService

    # Pick a valid user and session
    sample = test_df.groupby(["user_id", "session_id"]).size().reset_index(name="cnt")
    sample = sample[sample["cnt"] >= 5].iloc[0]
    uid, sid = int(sample["user_id"]), str(sample["session_id"])

    service = RecommenderService(test_df, strategy="popularity")
    recs = service.recommend(uid, sid, k=10)
    assert len(recs) > 0
    assert "video_id" in recs.columns
    assert "rank" in recs.columns
    print(f"  [OK] Recommended {len(recs)} items for user={uid}, session={sid}")


def test_logging():
    print("Testing logging layer...")
    from logging_layer.exposure_logger import ExposureLogger
    from logging_layer.outcome_logger import OutcomeLogger

    exp = ExposureLogger()
    exp.log(1000, 1, "1_1", [10, 20, 30], [1, 2, 3])
    flat = exp.to_flat_dataframe()
    assert len(flat) == 3
    print(f"  [OK] ExposureLogger: {len(exp)} records, {len(flat)} flat rows")

    out = OutcomeLogger()
    out.log(2000, 1, 10, "click", 0)
    out.log(3000, 1, 10, "watch", 5000)
    df = out.to_dataframe()
    assert len(df) == 2
    print(f"  [OK] OutcomeLogger: {len(out)} records")


def test_simulator():
    print("Testing interaction simulator...")
    from simulation.interaction_simulator import InteractionSimulator

    sim = InteractionSimulator(seed=42)
    events = sim.simulate_response(
        user_id=1, item_id=10, session_id="1_1", timestamp=1000,
        creator_id=100, category="Music"
    )
    assert len(events) >= 1  # At minimum: impression
    assert events[0].event_type == "impression"
    print(f"  [OK] Simulated {len(events)} events for single item")

    all_events = sim.simulate_session(
        user_id=1, recommended_items=[10, 20, 30],
        session_id="1_1", base_timestamp=1000,
    )
    assert len(all_events) >= 3  # At least one impression per item
    print(f"  [OK] Simulated {len(all_events)} events for session of 3 items")


def test_metrics():
    print("Testing metrics extensions...")
    from evaluation.metrics_extensions import compute_all_loop_metrics

    exposure_df = pd.DataFrame({
        "item_id": [1, 2, 3, 1, 2],
        "session_id": ["s1", "s1", "s1", "s2", "s2"],
    })
    outcome_df = pd.DataFrame({
        "event_type": ["impression", "click", "watch", "impression", "skip"],
        "watch_time": [0, 0, 5000, 0, 0],
    })

    metrics = compute_all_loop_metrics(exposure_df, outcome_df, total_items=10)
    assert "ctr_proxy" in metrics
    assert "catalog_coverage" in metrics
    assert metrics["catalog_coverage"] == 0.3  # 3 unique / 10 total
    print(f"  [OK] Metrics computed: {metrics}")


if __name__ == "__main__":
    print("=" * 60)
    print("Online Recommendation Loop — Smoke Test")
    print("=" * 60)

    # Create and save test data
    test_df = create_test_data(200)
    test_csv = os.path.join(os.path.dirname(__file__), "test_pipeline_sample.csv")
    test_df.to_csv(test_csv, index=False)

    try:
        test_event_schema()
        test_event_replay(test_csv)
        test_state_layer()
        test_recommender_service(test_df)
        test_logging()
        test_simulator()
        test_metrics()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60)
    finally:
        # Cleanup
        if os.path.exists(test_csv):
            os.remove(test_csv)
