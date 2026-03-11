import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
from src.config import PROJECT_ROOT

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "runs"

def generate_local_plots(run_id: str, metrics: dict):
    """
    Generate charts for a single run and save them in the run's plots folder.
    """
    run_dir = OUTPUTS_DIR / run_id
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Radar Chart of normalized metrics (approximate scale for visuals)
    # Since we can't reliably normalize against a global max without history, 
    # we'll build a simple bar chart of key metrics.
    
    sns.set_theme(style='whitegrid', palette='deep')
    
    df = pd.DataFrame([metrics])
    
    cols_to_plot = [
        ('ctr_proxy', 'CTR Proxy'),
        ('catalog_coverage', 'Coverage'),
        ('diversity', 'Diversity'),
        ('creator_spread', 'Creator Spread')
    ]
    
    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(4 * len(cols_to_plot), 4))
    fig.suptitle(f'Metrics Profile: {run_id}', fontsize=12, fontweight='bold', y=1.05)
    
    for idx, (col, title) in enumerate(cols_to_plot):
        if col not in df.columns:
            continue
        ax = axes[idx]
        sns.barplot(data=df, y=col, ax=ax, color='steelblue')
        ax.set_title(title)
        ax.set_ylabel("")
        ax.set_xticklabels([])
        
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_summary.png", dpi=120, bbox_inches='tight')
    plt.close()

def generate_comparison_plot(run_id_1: str, run_id_2: str) -> Path:
    """
    Generates a side-by-side comparison of two runs.
    Returns the path to the temporary visualization.
    """
    from src.history import get_run_details
    
    run_1 = get_run_details(run_id_1)
    run_2 = get_run_details(run_id_2)
    
    if not run_1 or not run_2:
        return None
        
    metrics_1 = run_1.get("metrics", {})
    metrics_2 = run_2.get("metrics", {})
    
    metrics_1['run_id'] = run_id_1
    metrics_2['run_id'] = run_id_2
    
    # Use short labels instead of full run IDs
    df = pd.DataFrame([metrics_1, metrics_2])
    short_labels = ['Run A', 'Run B']
    df['label'] = short_labels
    
    sns.set_theme(style='whitegrid', palette='deep')
    
    cols_to_plot = [
        ('ctr_proxy', 'CTR Proxy'),
        ('catalog_coverage', 'Coverage'),
        ('diversity', 'Diversity'),
        ('creator_spread', 'Creator Spread')
    ]
    
    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(4 * len(cols_to_plot), 4.5))
    fig.suptitle('Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
    
    colors = ['#2ecc71', '#e74c3c']
    
    for idx, (col, title) in enumerate(cols_to_plot):
        if col not in df.columns:
            continue
        ax = axes[idx]
        bars = ax.bar(short_labels, df[col].values, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Add value annotations on bars
        for bar, val in zip(bars, df[col].values):
            if isinstance(val, float):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add a legend below the figure
    # Extract the strategy names from the run_ids for clearer labeling
    legend_a = run_id_1.split('_', 2)[-1] if '_' in run_id_1 else run_id_1
    legend_b = run_id_2.split('_', 2)[-1] if '_' in run_id_2 else run_id_2
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors[0], label=f'A: {legend_a}'),
               mpatches.Patch(color=colors[1], label=f'B: {legend_b}')]
    fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, -0.08), frameon=True)
            
    plt.tight_layout()
    
    out_dir = PROJECT_ROOT / "outputs" / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{run_id_1}_vs_{run_id_2}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return out_path
