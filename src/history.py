import os
import pandas as pd
from datetime import datetime
import yaml
from pathlib import Path
import json

from src.config import PROJECT_ROOT, ExperimentConfig

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "runs"
INDEX_PATH = PROJECT_ROOT / "outputs" / "experiment_index.csv"

def get_run_index() -> pd.DataFrame:
    """Load the global experiment index."""
    if INDEX_PATH.exists():
        return pd.read_csv(INDEX_PATH)
    return pd.DataFrame()

def save_run(config: ExperimentConfig, metrics: dict, recommendations: pd.DataFrame) -> str:
    """
    Persist an experiment run's artifacts and metrics to disk.
    
    Returns:
        The run_id string.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp_str}_{config.scenario_name}_{config.strategy.name}"
    run_dir = OUTPUTS_DIR / run_id
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = config.model_dump()
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f)
        
    # Save metrics
    metrics_series = pd.Series(metrics)
    metrics_series.to_csv(run_dir / "metrics.csv")
    
    # Save recommendations limit to 1000 for size
    if not recommendations.empty:
        recommendations.head(1000).to_csv(run_dir / "recommendations.csv", index=False)
        
    # Generate Summary MD
    _generate_summary(run_dir, run_id, config, metrics)
    
    # Append to experiment index
    index_row = {
        "run_id": run_id,
        "timestamp": timestamp_str,
        "preset": config.scenario_name,
        "strategy": config.strategy.name,
        **metrics,
        "summary_path": str((run_dir / "summary.md").relative_to(PROJECT_ROOT)),
    }
    
    df_row = pd.DataFrame([index_row])
    
    if INDEX_PATH.exists():
        df_existing = pd.read_csv(INDEX_PATH)
        df_combined = pd.concat([df_existing, df_row], ignore_index=True)
        df_combined.to_csv(INDEX_PATH, index=False)
    else:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_row.to_csv(INDEX_PATH, index=False)
        
    return run_id
    
def _generate_summary(run_dir: Path, run_id: str, config: ExperimentConfig, metrics: dict):
    """
    Generates a deterministic summary report based on metrics.
    """
    summary = []
    summary.append(f"# Experiment Report: {run_id}")
    summary.append(f"\n**Scenario:** {config.scenario_name}")
    summary.append(f"**Strategy:** {config.strategy.name}")
    
    summary.append("\n## Metrics Summary")
    summary.append("| Metric | Value |")
    summary.append("|---|---|")
    for k, v in metrics.items():
        if isinstance(v, float):
            summary.append(f"| {k} | {v:.4f} |")
        else:
            summary.append(f"| {k} | {v} |")
            
    summary.append("\n## Interpretation")
    # Rule-based interpretation
    ctr = metrics.get('ctr_proxy', 0)
    diversity = metrics.get('diversity', 0)
    watch_time = metrics.get('watch_time_proxy_ms', 0)
    coverage = metrics.get('catalog_coverage', 0)
    creator = metrics.get('creator_spread', 0)
    
    # CTR
    if ctr > 0.15:
        summary.append("- **High CTR Proxy**: The strategy successfully identified items with high immediate engagement potential.")
    elif ctr > 0.05:
        summary.append("- **Moderate CTR Proxy**: Reasonable click-through rate, suggesting a balanced approach between relevance and exploration.")
    else:
        summary.append("- **Low CTR Proxy**: The strategy struggled to find immediately engaging items, possibly focusing too much on exploration or cold items.")
        
    # Watch Time
    if watch_time > 3000:
        summary.append("- **Strong Watch Time**: Users spent significant time with recommended content, indicating good content-fit.")
    elif watch_time > 1500:
        summary.append("- **Moderate Watch Time**: Average engagement duration suggests acceptable but improvable relevance.")
    else:
        summary.append("- **Low Watch Time**: Users disengaged quickly, possibly due to poor relevance or over-exploration.")
        
    # Diversity
    if diversity > 15:
        summary.append("- **High Diversity**: Good spread across categories, reducing the risk of filter bubbles.")
    elif diversity > 8:
        summary.append("- **Moderate Diversity**: Reasonable category variety. Some risk of concentration but not extreme.")
    elif diversity > 5:
        summary.append("- **Low Diversity**: Recommendations lean toward a narrow set of categories. Consider diversification strategies.")
    else:
        summary.append("- **Very Low Diversity**: Recommendations are heavily concentrated in few categories (potential Popularity Trap).")
        
    # Coverage
    if coverage > 0.05:
        summary.append("- **Good Coverage**: A substantial portion of the catalog was exposed to users.")
    else:
        summary.append("- **Narrow Coverage**: The strategy relied on a very small set of winner-take-all items.")
    
    # Creator Spread
    if creator > 0.4:
        summary.append("- **Strong Creator Spread**: Recommendations surface content from a wide range of creators.")
    elif creator > 0.2:
        summary.append("- **Moderate Creator Spread**: Some creator diversity, but a few dominant creators may receive disproportionate exposure.")
    else:
        summary.append("- **Low Creator Spread**: A small group of creators dominates the recommendations.")
    
    # Tradeoff summary
    summary.append("\n## Tradeoff Analysis")
    if ctr > 0.2 and diversity < 10:
        summary.append("This configuration traded **diversity for engagement**. While CTR is strong, the narrow category spread may lead to filter bubbles over time.")
    elif diversity > 12 and ctr < 0.2:
        summary.append("This configuration traded **immediate engagement for discovery**. Lower CTR is offset by broader catalog and category exposure.")
    elif ctr > 0.15 and diversity > 10:
        summary.append("This configuration achieved a **reasonable balance** between engagement and diversity -- a strong starting point for production experimentation.")
    else:
        summary.append("The tradeoffs in this scenario are nuanced. Review the metrics above to assess whether this strategy fits your product goals.")
        
    with open(run_dir / "summary.md", "w") as f:
        f.write("\n".join(summary))

def get_run_details(run_id: str) -> dict:
    """Helper to load a specific run's artifacts."""
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return {}
        
    result = {}
    
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            result['config'] = yaml.safe_load(f)
            
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        result['metrics'] = pd.read_csv(metrics_path, index_col=0).squeeze("columns").to_dict()
        
    summary_path = run_dir / "summary.md"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            result['summary'] = f.read()
            
    return result
