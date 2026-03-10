"""
run_all.py — Orchestrated reproduction script for DiscoveryRank.
Author: Jasjyot Singh

Executes all project notebooks in order, tracks the run with MLflow,
and saves versioned outputs and plots without altering notebook logic.
"""

import subprocess
import sys
import os
import shutil
import pandas as pd
from datetime import datetime

# Import local modules
sys.path.append(os.path.abspath('src'))
from config import load_config
from experiment_tracking import setup_mlflow, log_experiment_params, log_metrics_safely, log_local_artifact
from plotting import save_tradeoff_plot, save_bar_chart

NOTEBOOKS = [
    "notebooks/01_pipeline_check.ipynb",
    "notebooks/02_validation_checks.ipynb",
    "notebooks/03_baseline_strategy_comparison.ipynb",
    "notebooks/04_candidate_pool_strategy_comparison.ipynb",
    "notebooks/05_ml_baseline_and_advanced_eval.ipynb",
]

def run_notebook(path):
    print(f"\n{'='*60}")
    print(f"Running: {path}")
    print(f"{'='*60}")
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", path,
            "--output", os.path.basename(path),
            "--output-dir", "notebooks/",
            "--ExecutePreprocessor.timeout=600",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"FAILED: {path}")
        print(result.stderr)
        return False
    print(f"OK: {path}")
    return True

if __name__ == "__main__":
    print("DiscoveryRank — Full Pipeline Execution")
    print("=" * 60)
    
    # 1. Load config
    config = load_config()
    
    # 2. Setup Versioned Output Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"outputs/runs/{timestamp}"
    tables_dir = os.path.join(out_dir, "tables")
    plots_dir = os.path.join(out_dir, "plots")
    artifacts_dir = os.path.join(out_dir, "artifacts")
    for d in [tables_dir, plots_dir, artifacts_dir]:
        os.makedirs(d, exist_ok=True)
    print(f"Created versioned output directory structure: {out_dir}")
    
    # 3. Setup MLflow
    run = setup_mlflow(config, run_name=f"pipeline_run_{timestamp}")
    print(f"Started MLflow run: {run.info.run_name}")
    log_experiment_params(config)
    
    # 4. Run Notebooks
    failed = []
    for nb in NOTEBOOKS:
        if not run_notebook(nb):
            failed.append(nb)
            break # Stop pipeline on failure
            
    if failed:
        print(f"\n{'='*60}")
        print(f"FAILED notebooks: {failed}")
        sys.exit(1)
        
    print("\nAll notebooks executed successfully.")
    
    # 5. Process Outputs & Generate Plots
    try:
        comparison_csv = "outputs/phase3_strategy_comparison.csv"
        if os.path.exists(comparison_csv):
            df = pd.read_csv(comparison_csv)
            # Log metrics to MLflow
            for _, row in df.iterrows():
                strategy = row['strategy']
                metrics = {k: v for k, v in row.items() if k != 'strategy'}
                for k, v in metrics.items():
                    log_metrics_safely({f"{strategy}_{k}": v})
            
            # Generate Plots
            plot1_path = os.path.join(plots_dir, "relevance_vs_novelty.png")
            save_tradeoff_plot(df, "rel_mean_y_relevant", "adv_mean_novelty", "strategy", 
                               "Relevance vs. Novelty Tradeoff", plot1_path)
            
            plot2_path = os.path.join(plots_dir, "global_coverage.png")
            save_bar_chart(df, "strategy", "global_coverage_ratio", 
                           "Global Coverage by Strategy", plot2_path)
                           
            plot3_path = os.path.join(plots_dir, "serendipity.png")
            save_bar_chart(df, "strategy", "adv_mean_serendipity",
                           "Serendipity by Strategy", plot3_path)
                           
            # Copy CSV to versioned folder
            dest_csv = os.path.join(tables_dir, "phase3_strategy_comparison.csv")
            shutil.copy2(comparison_csv, dest_csv)
            
            # Save a demo copy of the tradeoff plot for recruiter review
            demo_plot = "docs/relevance_vs_novelty_demo.png"
            os.makedirs("docs", exist_ok=True)
            shutil.copy2(plot1_path, demo_plot)
            
            # Also save config into artifacts dir
            config_copy = os.path.join(artifacts_dir, "default_config.yaml")
            if os.path.exists("configs/default_config.yaml"):
                shutil.copy2("configs/default_config.yaml", config_copy)
            
            # Log to MLflow
            log_local_artifact(dest_csv)
            log_local_artifact(plot1_path)
            log_local_artifact(plot2_path)
            log_local_artifact(plot3_path)
            if os.path.exists(config_copy):
                log_local_artifact(config_copy)
                
            print(f"Processed artifacts and saved to {out_dir}")
        else:
            print("Warning: phase3_strategy_comparison.csv not found.")
            
    except Exception as e:
        print(f"Error processing outputs: {e}")
        
    import mlflow
    mlflow.end_run()
    
    print("\nExperiment run complete.")
