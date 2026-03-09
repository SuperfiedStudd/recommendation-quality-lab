"""
Module: freshness_features
Responsibility:
- Parse timestamp strings and UNIX epoch integers.
- Calculate relative item age (e.g., interaction_time_ms - upload_dt).
"""

import pandas as pd
import numpy as np

def calculate_freshness(df: pd.DataFrame, upload_dt_col: str = "upload_dt", interaction_time_col: str = "time_ms") -> pd.DataFrame:
    """
    Parses `upload_dt` strings defensively and compares them to `time_ms` event records.
    """
    df = df.copy()
    
    if upload_dt_col not in df.columns:
        print(f"Warning: {upload_dt_col} not found in dataframe. Returning unmodified.")
        return df

    # Parse standard string upload_dt (e.g., "2022-04-20") to pandas datetime
    df['upload_dt_parsed'] = pd.to_datetime(df[upload_dt_col], errors='coerce')
    
    # 1. Convert successful strings into standard UNIX epoch milliseconds natively
    # Using float to support NaNs cleanly (NaT conversion produces messy negative artifacts otherwise)
    df['upload_time_ms'] = df['upload_dt_parsed'].astype(np.int64) // 10**6
    df['upload_time_ms'] = np.where(df['upload_dt_parsed'].isna(), np.nan, df['upload_time_ms'])
    
    # 2. Handle missing/null assumption (Clearly noted in implementation_plan_v1.md)
    # If missing upload_dt, we fall back to the exact interaction time safely -> resulting in item_age 0
    df['upload_time_ms'] = df['upload_time_ms'].fillna(df[interaction_time_col])
    
    # 3. Create discrete item age logic
    df['item_age_ms'] = df[interaction_time_col] - df['upload_time_ms']
    
    # Edge case clip logic: We zero out "negative" ages.
    # Why? Timezone mismatches between upload_dt and interaction_time_ms can cause slightly negative bounds.
    df['item_age_ms'] = df['item_age_ms'].clip(lower=0)
    
    # 4. Derived metric
    df['item_age_days'] = df['item_age_ms'] / (1000 * 60 * 60 * 24)
    
    return df

