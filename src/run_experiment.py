import argparse
import sys
import os
import pandas as pd
from datetime import datetime

# Ensure src/ and project root are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.config import load_config
from data.event_replay import load_interactions, generate_events
from features.state_manager import StateManager
from serving.recommender_service import RecommenderService
from simulation.interaction_simulator import InteractionSimulator
from logging_layer.exposure_logger import ExposureLogger
from logging_layer.outcome_logger import OutcomeLogger
from evaluation.metrics_extensions import compute_all_loop_metrics
from src.history import save_run, get_run_index
from src.plotting import generate_local_plots, generate_comparison_plot

def _build_item_metadata(master_df: pd.DataFrame) -> dict:
    meta = {}
    for _, row in master_df[["video_id", "author_id", "tag"]].drop_duplicates("video_id").iterrows():
        meta[int(row["video_id"])] = {
            "creator_id": int(row["author_id"]) if pd.notna(row["author_id"]) else None,
            "category": str(row["tag"]) if pd.notna(row["tag"]) else "Unknown",
        }
    return meta

def run_compare(run_id_1: str, run_id_2: str):
    """Utility to print a comparison table and generate a plot."""
    from src.history import get_run_details
    d1 = get_run_details(run_id_1)
    d2 = get_run_details(run_id_2)
    
    if not d1 or not d2:
        print("Could not find both runs to compare.")
        return
        
    print(f"\nComparing {run_id_1} vs {run_id_2}")
    print(f"{'Metric':<20} | {run_id_1:<20} | {run_id_2:<20} | Diff")
    print("-" * 70)
    
    m1 = d1.get('metrics', {})
    m2 = d2.get('metrics', {})
    
    for k in m1.keys():
        v1 = m1.get(k, 0)
        v2 = m2.get(k, 0)
        
        if isinstance(v1, float):
            diff = v2 - v1
            print(f"{k:<20} | {v1:<20.4f} | {v2:<20.4f} | {diff:+.4f}")
        else:
            print(f"{k:<20} | {str(v1):<20} | {str(v2):<20} |")
            
    plot_path = generate_comparison_plot(run_id_1, run_id_2)
    if plot_path:
        print(f"\nGenerated comparison plot at: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Recommendation Strategy Lab")
    parser.add_argument("--preset", type=str, default="cold_start_catalog", help="Scenario preset name")
    parser.add_argument("--strategy", type=str, default="popularity_first", help="Strategy configuration name")
    parser.add_argument("--override", type=str, action="append", help="Override config, e.g. key=value", default=[])
    parser.add_argument("--compare", type=str, nargs=2, metavar=('RUN_ID_1', 'RUN_ID_2'), help="Compare two run IDs directly and exit")
    parser.add_argument("--compare-last", action="store_true", help="Compare this run to the very last recorded run")
    parser.add_argument("--open-report", action="store_true", help="Print summary report to terminal after run")
    
    args = parser.parse_args()
    
    if args.compare:
        run_compare(args.compare[0], args.compare[1])
        return

    # Parse overrides
    overrides = {}
    for o in args.override:
        if "=" in o:
            k, v = o.split("=", 1)
            overrides[k] = v

    print("Loading lab configuration...")
    config = load_config(args.preset, args.strategy, overrides)
    
    print(f"==================================================")
    print(f"Recommendation Strategy Lab")
    print(f"Scenario: {config.scenario_name} | Strategy: {config.strategy.name}")
    print(f"Events: {config.simulation.events:,} | Top-K: {config.simulation.top_k}")
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

    # 2. Setup Configured Components
    state_mgr = StateManager()
    exposure_logger = ExposureLogger()
    outcome_logger = OutcomeLogger()
    
    simulator = InteractionSimulator(
        base_click_prob=config.simulation.base_click_prob,
        base_watch_prob=config.simulation.base_watch_prob,
        like_prob=config.simulation.like_prob,
        save_prob=config.simulation.save_prob,
        seed=42
    )
    
    cg_kwargs = {
        "history_pool_ratio": config.simulation.history_pool_ratio,
        "popular_pool_ratio": config.simulation.popular_pool_ratio
    }
    
    service = RecommenderService(
        master_df=master_df, 
        state_manager=state_mgr, 
        strategy=config.strategy.name,
        strategy_kwargs=config.strategy.strategy_kwargs,
        candidate_generation_kwargs=cg_kwargs
    )

    # 3. State Warmup
    print(f"Warming up state with {config.simulation.events:,} events...")
    warmup_df = master_df.sort_values("time_ms").head(config.simulation.events)
    events = generate_events(warmup_df)
    for evt in events:
        state_mgr.process_event(evt)

    # 4. Find evaluation sessions
    if config.simulation.events < len(master_df):
        eval_df = master_df.sort_values("time_ms").iloc[config.simulation.events:]
    else:
        eval_df = master_df.tail(config.simulation.eval_sessions * 5)
        
    session_keys = eval_df[["user_id", "session_id"]].drop_duplicates().head(config.simulation.eval_sessions)
    eval_pairs = [tuple(x) for x in session_keys.to_numpy()]

    # 5. Simulation Loop
    print(f"\nSimulating and evaluating {len(eval_pairs)} sessions...")
    
    all_recs = []
    
    for user_id, session_id in eval_pairs:
        recs = service.recommend(user_id, session_id, k=config.simulation.top_k)
        if recs.empty:
            continue
            
        all_recs.append(recs)

        rec_items = recs["video_id"].tolist()
        rec_ranks = recs["rank"].tolist()

        sess_data = master_df[(master_df["user_id"] == user_id) & (master_df["session_id"] == session_id)]
        if sess_data.empty:
            continue
        base_ts = int(sess_data["time_ms"].min())

        exposure_logger.log(timestamp=base_ts, user_id=user_id, session_id=session_id,
                            recommended_items=rec_items, rank_positions=rec_ranks)

        user_state = state_mgr.get_user_state(user_id)
        outcomes = simulator.simulate_session(
            user_id=user_id, recommended_items=rec_items, session_id=session_id,
            base_timestamp=base_ts, item_metadata=item_metadata,
            user_state=user_state, item_states=state_mgr.item_states
        )

        for evt in outcomes:
            outcome_logger.log(timestamp=evt.timestamp, user_id=evt.user_id, item_id=evt.item_id,
                               event_type=evt.event_type, watch_time=evt.watch_time, session_id=evt.session_id)
            state_mgr.process_event(evt)
            
    recs_df = pd.concat(all_recs, ignore_index=True) if all_recs else pd.DataFrame()

    # 6. Metrics & Final Output
    print("\nComputing metrics...")
    metrics = compute_all_loop_metrics(
        exposure_df=exposure_logger.to_flat_dataframe(),
        outcome_df=outcome_logger.to_dataframe(),
        total_items=total_items,
        item_metadata=item_metadata
    )

    print(f"\n===== FINAL SUMMARY =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<16}: {v:.4f}")
        else:
            print(f"  {k:<16}: {v}")

    # 7. Write Outputs
    print("\nSaving artifacts...")
    index_df = get_run_index()
    last_run_id = None
    if not index_df.empty:
        last_run_id = index_df.iloc[-1]['run_id']
        
    run_id = save_run(config, metrics, recs_df)
    generate_local_plots(run_id, metrics)
    
    print(f"Run {run_id} completed and saved to outputs/runs/{run_id}/")
    
    if args.compare_last and last_run_id:
        run_compare(last_run_id, run_id)
        
    if args.open_report:
        from src.history import get_run_details
        d = get_run_details(run_id)
        print("\n" + "="*50 + "\nREPORT:\n" + "="*50)
        print(d.get('summary', ''))

if __name__ == "__main__":
    main()
