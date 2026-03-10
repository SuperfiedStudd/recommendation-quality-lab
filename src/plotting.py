import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def save_tradeoff_plot(df, x_col, y_col, hue_col, title, output_path):
    """
    Saves a scatter plot showing tradeoffs (e.g. relevance vs diversity).
    """
    plt.figure(figsize=(10, 6))
    
    # Check if columns exist
    if not all(c in df.columns for c in [x_col, y_col, hue_col]):
        print(f"Skipping plot {title}, missing columns.")
        return
        
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=200, style=hue_col, markers=["o", "s", "D", "^"])
    
    plt.title(title, pad=20)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_bar_chart(df, category_col, value_col, title, output_path):
    """
    Saves a basic bar chart (e.g. coverage per strategy).
    """
    plt.figure(figsize=(10, 6))
    
    if not all(c in df.columns for c in [category_col, value_col]):
        print(f"Skipping plot {title}, missing columns.")
        return
        
    sns.barplot(data=df, x=category_col, y=value_col, hue=category_col, legend=False)
    
    plt.title(title, pad=20)
    plt.xlabel(category_col.replace('_', ' ').title())
    plt.ylabel(value_col.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
