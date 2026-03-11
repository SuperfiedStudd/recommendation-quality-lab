"""
run_simulation.py — CLI Runner for the Online Recommendation Loop
Author: Jasjyot Singh

Executes the recommendation loop for a single policy and appends
metrics to the experiment tracking CSV.

Usage:
    python run_simulation.py --policy hybrid --events 10000 --top_k 20
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath('src'))

from data.event_replay import load_interactions, replay_stream, generate_events
from data.event_schema import Event
from features.state_manager import StateManager
from serving.recommender_service import RecommenderService
from simulation.interaction_simulator import InteractionSimulator
from logging_layer.exposure_logger import ExposureLogger
from logging_layer.outcome_logger import OutcomeLogger
from evaluation.metrics_extensions import compute_all_loop_metrics

# Map CLI names to internal strategy names
POLICY_MAP = {
    "popularity": "popularity",
    "recency_decay": "freshness_boosted",
    "hybrid": "diversity_aware",
}


def _build_item_metadata(master_df: pd.DataFrame) -> dict:
    meta = {}
    for _, row in master_df[["video_id", "author_id", "tag"]].drop_duplicates("video_id").iterrows():
        meta[int(row["video_id"])] = {
            "creator_id": int(row["author_id"]) if pd.notna(row["author_id"]) else None,
            "category": str(row["tag"]) if pd.notna(row["tag"]) else "Unknown",
        }
    return meta


def _plot_comparisons(csv_path: str, out_png: str):
    """Generate and save a bar chart comparing policies from the CSV."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    # Keep only the latest run for each policy
    df = df.drop_duplicates(subset=["policy"], keep="last")

    sns.set_theme(style='whitegrid', palette='deep')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Online Loop Policy Comparison', fontsize=14, fontweight='bold', y=1.05)

    metrics = [
        ('ctr_proxy', 'CTR Proxy'),
        ('watch_time_proxy_ms', 'Watch Time (ms)'),
        ('diversity', 'Category Diversity'),
        ('creator_spread', 'Creator Spread'),
    ]

    colors = sns.color_palette("deep", len(df["policy"].unique()))

    for idx, (col, title) in enumerate(metrics):
        if col not in df.columns:
            continue
        ax = axes[idx]
        sns.barplot(data=df, x="policy", y=col, ax=ax, palette=colors, hue="policy", legend=False)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run DiscoveryRank Online Simulation Loop")
    parser.add_argument("--policy", type=str, required=True, choices=list(POLICY_MAP.keys()),
                        help="Ranking policy to simulate")
    parser.add_argument("--events", type=int, default=10000,
                        help="Number of historical events to replay for state warmup")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of recommendations to serve per session")
    parser.add_argument("--eval_sessions", type=int, default=50,
                        help="Number of sessions to evaluate after warmup")
    
    args = parser.parse_args()
    strategy_key = POLICY_MAP[args.policy]

    print(f"==================================================")
    print(f"DiscoveryRank - Online Simulation Runner")
    print(f"Policy: {args.policy} ({strategy_key})")
    print(f"Warmup Events: {args.events:,} | Top-K: {args.top_k}")
    print(f"==================================================\n")

    # 1. Load Data
    print("Loading dataset...")
    try:
        master_df = load_interactions("outputs/pipeline_sample.csv")
    except FileNotFoundError:
        print("Error: pipeline_sample.csv not found in outputs/. Please run data prep first.")
        sys.exit(1)

    item_metadata = _build_item_metadata(master_df)
    total_items = master_df["video_id"].nunique()

    # 2. Initialize Components
    state_mgr = StateManager()
    exposure_logger = ExposureLogger()
    outcome_logger = OutcomeLogger()
    simulator = InteractionSimulator(seed=42)
    service = RecommenderService(master_df=master_df, state_manager=state_mgr, strategy=strategy_key)

    # 3. State Warmup
    print(f"Warming up state with {args.events:,} events...")
    warmup_df = master_df.sort_values("time_ms").head(args.events)
    events = generate_events(warmup_df)
    for evt in events:
        state_mgr.process_event(evt)
    
    print(f"State summary: {state_mgr.summary()}")

    # 4. Find evaluation sessions
    # Get sessions that occur AFTER the warmup period to simulate future traffic
    if args.events < len(master_df):
        eval_df = master_df.sort_values("time_ms").iloc[args.events:]
    else:
        eval_df = master_df.tail(args.eval_sessions * 5) # Fallback
        
    session_keys = eval_df[["user_id", "session_id"]].drop_duplicates().head(args.eval_sessions)
    eval_pairs = [tuple(x) for x in session_keys.to_numpy()]

    # 5. Simulation Loop
    print(f"\nSimulating and evaluating {len(eval_pairs)} sessions...")
    for user_id, session_id in eval_pairs:
        # Get recommendations
        recs = service.recommend(user_id, session_id, k=args.top_k)
        if recs.empty:
            continue

        rec_items = recs["video_id"].tolist()
        rec_ranks = recs["rank"].tolist()

        # Find timestamp context
        sess_data = master_df[(master_df["user_id"] == user_id) & (master_df["session_id"] == session_id)]
        if sess_data.empty:
            continue
        base_ts = int(sess_data["time_ms"].min())

        # Log exposure
        exposure_logger.log(timestamp=base_ts, user_id=user_id, session_id=session_id,
                            recommended_items=rec_items, rank_positions=rec_ranks)

        # Simulate outcomes
        user_state = state_mgr.get_user_state(user_id)
        outcomes = simulator.simulate_session(
            user_id=user_id, recommended_items=rec_items, session_id=session_id,
            base_timestamp=base_ts, item_metadata=item_metadata,
            user_state=user_state, item_states=state_mgr.item_states
        )

        # Log and update state
        for evt in outcomes:
            outcome_logger.log(timestamp=evt.timestamp, user_id=evt.user_id, item_id=evt.item_id,
                               event_type=evt.event_type, watch_time=evt.watch_time, session_id=evt.session_id)
            state_mgr.process_event(evt)

    # 6. Metrics & Final Output
    print("\nComputing metrics...")
    metrics = compute_all_loop_metrics(
        exposure_df=exposure_logger.to_flat_dataframe(),
        outcome_df=outcome_logger.to_dataframe(),
        total_items=total_items,
        item_metadata=item_metadata
    )
    
    metrics["policy"] = args.policy
    metrics["events_replayed"] = args.events
    metrics["top_k"] = args.top_k

    print(f"\nFINAL SUMMARY ({args.policy}):")
    print(f"  CTR proxy        : {metrics['ctr_proxy']:.4f}")
    print(f"  avg watch time   : {metrics['watch_time_proxy_ms']:.0f} ms")
    print(f"  coverage         : {metrics['catalog_coverage']:.4f}")
    print(f"  diversity        : {metrics['diversity']:.4f}")
    print(f"  creator spread   : {metrics['creator_spread']:.4f}")

    # 7. Write Outputs
    out_dir = "outputs/experiments"
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "policy_comparison.csv")
    png_path = os.path.join(out_dir, "policy_comparison.png")

    # Append to CSV
    row_df = pd.DataFrame([metrics])
    
    # Specific column order requested
    cols = ["policy", "ctr_proxy", "watch_time_proxy_ms", "catalog_coverage", "diversity", "creator_spread"]
    # Add any extra tracking columns that aren't in the strict list
    extra_cols = [c for c in row_df.columns if c not in cols]
    final_cols = cols + extra_cols
    row_df = row_df[final_cols]

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        # Drop previous run for the same policy if requested, or just append
        new_df = pd.concat([existing_df, row_df], ignore_index=True)
        # Deduplicate to keep latest run per policy
        new_df = new_df.drop_duplicates(subset=["policy"], keep="last")
        new_df.to_csv(csv_path, index=False)
    else:
        row_df.to_csv(csv_path, index=False)
        
    print(f"\nSaved metrics to {csv_path}")

    # Generate comparative plot
    _plot_comparisons(csv_path, png_path)
    if os.path.exists(png_path):
        print(f"Saved comparison plot to {png_path}")

if __name__ == "__main__":
    main()
