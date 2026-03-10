"""
Module: validation_checks
Responsibility:
- Defines standardized functions for contradiction checking and edge-case highlighting.
"""

import pandas as pd
import numpy as np

def check_logical_contradictions(df: pd.DataFrame) -> dict:
    """
    Scans the dataframe for logical impossibilities or edge case groupings.
    Returns a dictionary of boolean masks indicating violating rows.
    """
    masks = {}
    
    # 1. Hate but Relevant
    # I explicitly defined y_relevant to yield 0 if is_hate == 1. 
    # If this is violated, the label logic is broken.
    masks['hate_but_relevant'] = (df['is_hate'] == 1) & (df['y_relevant'] == 1)
    
    # 2. Impossible negative duration
    masks['invalid_duration'] = df['duration_ms'] <= 0
    
    # 3. High implicit completion (e.g., > 1.5)
    # Users can loop short videos, but an extreme ratio implies data errors (like play_time > 1 hr on a 5s clip).
    masks['high_completion_ratio'] = df['implicit_completion_ratio'] > 1.5
    
    # 4. Duplicate Interactions
    # A user shouldn't interact with the exact same video at the exact same millisecond entirely redundantly.
    # Note: KuaiRand allows the same user/video across different times, so I explicitly check (user, video, time).
    masks['exact_duplicate_event'] = df.duplicated(subset=['user_id', 'video_id', 'time_ms'], keep=False)
    
    # 5. Missing Upload Date but age fields calculated
    masks['missing_upload_dt'] = df['upload_dt'].isna()
    
    return masks

def sample_edge_cases(df: pd.DataFrame, masks: dict, n_normal: int = 50, n_edge: int = 150) -> pd.DataFrame:
    """
    Combines normal rows with flagged edge-case rows for manual review.
    """
    df = df.copy()
    
    # Tag edge cases directly in dataframe
    df['is_edge_case'] = False
    df['edge_case_reason'] = ""
    
    for reason, mask in masks.items():
        if mask.any():
            df.loc[mask, 'is_edge_case'] = True
            # Append reason
            df.loc[mask, 'edge_case_reason'] = df.loc[mask, 'edge_case_reason'] + reason + " | "
            
    # Sample from each pool
    edge_pool = df[df['is_edge_case']]
    normal_pool = df[~df['is_edge_case']]
    
    sampled_edge = edge_pool.sample(min(len(edge_pool), n_edge), random_state=42)
    sampled_normal = normal_pool.sample(min(len(normal_pool), n_normal), random_state=42)
    
    review_df = pd.concat([sampled_edge, sampled_normal]).sort_values(by=['user_id', 'time_ms'])
    return review_df
