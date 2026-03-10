"""
run_all.py — Minimal reproduction script for DiscoveryRank.
Author: Jasjyot Singh

Executes all project notebooks in order using nbconvert.
Requires: pip install -r requirements.txt

Usage:
    python run_all.py
"""

import subprocess
import sys

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
            "--output", path.split("/")[-1],
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
    print("DiscoveryRank — Full Pipeline Reproduction")
    print("=" * 60)
    failed = []
    for nb in NOTEBOOKS:
        if not run_notebook(nb):
            failed.append(nb)
    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED notebooks: {failed}")
        sys.exit(1)
    else:
        print("All notebooks executed successfully.")
        print("Results written to outputs/")
