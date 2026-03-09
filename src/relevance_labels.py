"""
Module: relevance_labels
Responsibility:
- Derive explicit binary labels (e.g., converting is_like/is_follow into a single positive boolean).
- Derive continuous implicit labels (e.g., calculating play_time_ms / duration_ms).
"""

import pandas as pd
import numpy as np

def create_relevance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates initial relevance signals from the available log fields as defined in the eval implementation track.
    """
    df = df.copy()
    
    # 1. Explicit Positive Signals
    # Max acts as a logical OR (if any of these is 1, explicit_positive_any = 1)
    df['explicit_positive_any'] = df[['is_like', 'is_follow', 'is_forward', 'is_comment']].max(axis=1)
    
    # 2. Explicit Negative Signals
    df['explicit_negative'] = df['is_hate']
    
    # 3. Implicit Completion Signal
    # Prevent divide by zero using numpy where constraints
    df['implicit_completion_ratio'] = np.where(
        df['duration_ms'] > 0,
        df['play_time_ms'] / df['duration_ms'],
        0.0
    )
    # Cap to prevent insane outliers (due to looping, caching, or corrupted play_time data)
    df['implicit_completion_ratio'] = df['implicit_completion_ratio'].clip(lower=0.0, upper=2.0)
    
    # 4. Practical Derived Ground Truth Label: y_relevant
    # A single defensible composite score to track overall ranking goodness for v1.
    # Rules:
    #   - If explicitly hated: 0.
    #   - If explicitly liked/forwarded: 1.
    #   - If completion threshold > 0.8: 1. (Defensible heuristic for implicit relevance)
    #   - Else: 0.
    df['y_relevant'] = np.where(
        df['explicit_negative'] == 1, 0,
        np.where(
            df['explicit_positive_any'] == 1, 1,
            np.where(df['implicit_completion_ratio'] > 0.8, 1, 0)
        )
    )
    
    return df

