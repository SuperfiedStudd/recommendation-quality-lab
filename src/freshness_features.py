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

    Implementation note on units (critical):
    - `time_ms` (interaction) is UNIX epoch in milliseconds (~1.65e12 for April 2022).
    - `upload_dt` strings (e.g. "2022-04-11") are parsed by pandas to datetime objects.
    - pd.Timestamp.value is ALWAYS nanoseconds since 1970-01-01 UTC.
    - Dividing by 10**6 converts nanoseconds → milliseconds (epoch ms), matching time_ms scale.
    - NaT values produce Python int -9223372036854775808 (min int64) when cast directly;
      must be masked out before any int64 operation.
    - The result upload_time_ms must be kept as float64 (for NaN support), but 13-digit
      ms timestamps are safely representable in float64 (max precision loss < 1 ms).
    """
    df = df.copy()

    if upload_dt_col not in df.columns:
        print(f"Warning: {upload_dt_col} not found in dataframe. Returning unmodified.")
        return df

    # Parse standard string upload_dt (e.g., "2022-04-20") to pandas datetime
    df['upload_dt_parsed'] = pd.to_datetime(df[upload_dt_col], errors='coerce')

    # 1. Convert valid datetimes to UNIX epoch milliseconds.
    # Strategy: use .apply() on valid rows to call .timestamp() (→ seconds float) * 1000 (→ ms).
    # .timestamp() is explicit about timezone handling (assumes local tz for naive datetimes;
    # KuaiRand upload_dt dates have no tz, so we normalize to UTC before conversion).
    def _to_epoch_ms(ts):
        """Convert a pandas Timestamp to UNIX epoch milliseconds (float64)."""
        return ts.tz_localize('UTC').timestamp() * 1000.0

    valid_mask = df['upload_dt_parsed'].notna()
    df['upload_time_ms'] = np.nan  # float64 column, NaN for missing rows

    if valid_mask.any():
        df.loc[valid_mask, 'upload_time_ms'] = (
            df.loc[valid_mask, 'upload_dt_parsed'].apply(_to_epoch_ms)
        )

    # 2. Handle missing/null assumption (Clearly noted in implementation_plan_v1.md):
    # If missing upload_dt, fall back to the exact interaction time → item_age = 0.
    df['upload_time_ms'] = df['upload_time_ms'].fillna(df[interaction_time_col].astype(float))

    # 3. Compute item age in milliseconds
    df['item_age_ms'] = df[interaction_time_col].astype(float) - df['upload_time_ms']

    # Edge case clip: zero out "negative" ages (timezone/data-entry mismatches).
    df['item_age_ms'] = df['item_age_ms'].clip(lower=0)

    # 4. Convert to days for human-readable metric
    df['item_age_days'] = df['item_age_ms'] / (1000.0 * 60.0 * 60.0 * 24.0)

    return df
