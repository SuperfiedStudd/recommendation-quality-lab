"""
Module: session_builder
Responsibility:
- Parse chronological user interactions from log files.
- Calculate time gaps between consecutive events.
- Assign internal session_id boundaries based on inactivity thresholds.
"""

import pandas as pd

def assign_sessions(df: pd.DataFrame, time_col: str = "time_ms", user_col: str = "user_id", gap_threshold_ms: int = 30 * 60 * 1000) -> pd.DataFrame:
    """
    Takes interaction rows sorted by user_id and time_ms and dynamically slices them into sessions.
    
    Args:
        df: Master dataframe containing User + Time variables.
        time_col: Column indicating the UNIX epoch timestamp in ms.
        user_col: Column bounding the user scope.
        gap_threshold_ms: Configurable threshold for inactivity defining a new session boundary (default 30 min).
        
    Returns:
        Dataframe modified heavily inline with a new `session_id`.
    """
    # 1. Sort values critically 
    df = df.sort_values(by=[user_col, time_col]).copy()
    
    # 2. Find explicit time gap between subsequent row indices
    df['time_diff'] = df.groupby(user_col)[time_col].diff()
    
    # 3. Gap rule assignment 
    # Starts a new session if gap > threshold or if it's the very first row for a user (time_diff is NaN)
    df['is_new_session'] = (df['time_diff'] > gap_threshold_ms) | df['time_diff'].isna()
    
    # 4. Generate incrementing ID component directly via cumulative sum
    df['session_id_part'] = df.groupby(user_col)['is_new_session'].cumsum()
    
    # 5. Composite uniquely identifiable session keys
    df['session_id'] = df[user_col].astype(str) + "_" + df['session_id_part'].astype(int).astype(str)
    
    # Cleanup logic
    df = df.drop(columns=['time_diff', 'is_new_session', 'session_id_part'])
    
    return df

