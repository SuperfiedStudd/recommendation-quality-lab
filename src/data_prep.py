"""
Module: data_prep
Responsibility: 
- Load raw CSVs from the data/ directory.
- Handle null imputation (e.g., fallback for missing upload_dt or tag).
- Merge video and user features onto interaction logs to create a master denormalized dataset.
"""

import pandas as pd
import os

def load_and_merge_data(data_dir: str, log_files: list, sample_size: int = None) -> pd.DataFrame:
    """
    Loads raw log files and merges them with video features.
    
    Args:
        data_dir: Path to the workspace data folder.
        log_files: List of log filenames to process.
        sample_size: Optional integer to limit the number of rows loaded per log file.
    
    Returns:
        pd.DataFrame containing the merged pipeline starting point.
    """
    # 1. Load video features
    video_features_path = os.path.join(data_dir, "video_features_basic_1k.csv")
    video_df = pd.read_csv(video_features_path)
    
    # 2. Iterate and load log files
    all_logs = []
    for log_file in log_files:
        log_path = os.path.join(data_dir, log_file)
        if sample_size:
            df = pd.read_csv(log_path, nrows=sample_size)
        else:
            df = pd.read_csv(log_path)
        all_logs.append(df)
        
    logs_df = pd.concat(all_logs, ignore_index=True)
    
    # 3. Join logic
    # We do a left join to keep all log interactions even if video features are missing
    merged_df = logs_df.merge(video_df, on="video_id", how="left")
    
    # Simple explicit imputation for missing tags as outlined in our assumption plan
    if 'tag' in merged_df.columns:
        merged_df['tag'] = merged_df['tag'].fillna("Unknown")
        
    return merged_df

