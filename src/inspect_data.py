import pandas as pd
import os

files_to_inspect = [
    r"c:\Code\reco-ag\data\log_random_4_22_to_5_08_1k.csv",
    r"c:\Code\reco-ag\data\log_standard_4_08_to_4_21_1k.csv",
    r"c:\Code\reco-ag\data\log_standard_4_22_to_5_08_1k.csv",
    r"c:\Code\reco-ag\data\user_features_1k.csv",
    r"c:\Code\reco-ag\data\video_features_basic_1k.csv",
    r"c:\Code\reco-ag\data\video_features_statistic_1k.csv"
]

inventory = []

def infer_columns(cols):
    id_cols = [c for c in cols if 'id' in c.lower()]
    time_cols = [c for c in cols if 'time' in c.lower() or 'date' in c.lower()]
    eng_cols = [c for c in cols if 'click' in c.lower() or 'like' in c.lower() or 'play' in c.lower() or 'view' in c.lower() or 'duration' in c.lower() or 'is_' in c.lower()]
    return id_cols, time_cols, eng_cols

print("="*50)
print("RAW DATA INSPECTION")
print("="*50)

all_columns_across_files = {}

# Pass 1: Gather all columns to find common join keys
for fpath in files_to_inspect:
    fname = os.path.basename(fpath)
    try:
        # Just read a few rows to get columns
        df_head = pd.read_csv(fpath, nrows=5)
        all_columns_across_files[fname] = set(df_head.columns)
    except Exception:
        all_columns_across_files[fname] = set()

for fpath in files_to_inspect:
    fname = os.path.basename(fpath)
    print(f"\n1. File Name: {fname}")
    try:
        df = pd.read_csv(fpath)
        shape = df.shape
        cols = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        null_counts = df.isnull().sum()
        top_nulls = null_counts.sort_values(ascending=False).head(10).to_dict()
        
        print(f"2. Shape: {shape}")
        print("3. Full Column List:", cols)
        print("4. Inferred DTypes:")
        for c in cols:
            print(f"  - {c}: {dtypes[c]}")
        print("5. Null Counts (All columns):")
        for c in cols:
            print(f"  - {c}: {null_counts[c]}")
        print("6. Top 10 Null Columns:")
        for c, count in top_nulls.items():
            if count > 0:
                print(f"  - {c}: {count}")
            elif count == 0 and top_nulls[c] == 0:
                pass
        
        if sum(top_nulls.values()) == 0:
            print("  - None")
            
        print("\n7. First 5 rows:")
        print(df.head())
        
        id_cols, time_cols, eng_cols = infer_columns(cols)
        
        # Suspected Primary Key
        primary_keys = []
        if fname.startswith("user") and "user_id" in id_cols:
            primary_keys = ["user_id"]
        elif fname.startswith("video") and "video_id" in id_cols:
            primary_keys = ["video_id"]
        elif fname.startswith("log") and "user_id" in id_cols and "video_id" in id_cols:
            # Interaction log, likely composite PK or no true PK
            primary_keys = ["user_id", "video_id", "time_ms"] # Uncertain composite
            
        # Suspected Join Keys (Shared with ANY other file)
        join_keys = []
        for other_fname, other_cols in all_columns_across_files.items():
            if other_fname != fname:
                shared = set(cols).intersection(other_cols)
                # Only keep ones that look like keys, e.g. end with id
                shared_ids = [c for c in shared if 'id' in c.lower()]
                join_keys.extend(shared_ids)
        join_keys = list(set(join_keys))
        
        print("\n8. Suspected Primary Key or Unique ID columns:", primary_keys, "(Uncertain)" if not primary_keys else "")
        print("9. Suspected Join Key columns shared with other files:", join_keys)
        print("10. Suspected Timestamp/Date/Time-related columns:", time_cols)
        print("11. Suspected Engagement or Label columns:", eng_cols)
        
        grain = "Uncertain"
        if "user_id" in cols and "video_id" in cols:
            grain = "user-video interaction"
        elif "user_id" in cols:
            grain = "user"
        elif "video_id" in cols:
            grain = "video"
            
        notes = "Looks clean." if df.isnull().sum().sum() == 0 else "Contains null values."
        notes += " (Uncertain grain/keys)"
        print(f"12. Short Notes: {notes}")
            
        inventory.append({
            "file_name": fname,
            "file_path": fpath,
            "row_count": shape[0],
            "column_count": shape[1],
            "full_column_list": ", ".join(cols),
            "suspected_grain": grain,
            "suspected_primary_keys": ", ".join(primary_keys),
            "suspected_join_keys": ", ".join(join_keys),
            "suspected_timestamp_columns": ", ".join(time_cols),
            "suspected_engagement_columns": ", ".join(eng_cols),
            "top_null_columns": ", ".join([f"{k}:{v}" for k, v in top_nulls.items() if v > 0]),
            "notes": notes
        })
    except Exception as e:
        print(f"Failed to read {fpath}: {e}")
        inventory.append({
            "file_name": fname,
            "file_path": fpath,
            "row_count": "Error",
            "column_count": "Error",
            "full_column_list": "",
            "suspected_grain": "Error",
            "suspected_primary_keys": "",
            "suspected_join_keys": "",
            "suspected_timestamp_columns": "",
            "suspected_engagement_columns": "",
            "top_null_columns": "",
            "notes": str(e)
        })

print("\n\n" + "="*50)
print("INVENTORY SUMMARY PREVIEW")
print("="*50)
inventory_df = pd.DataFrame(inventory)
print(inventory_df[['file_name', 'suspected_grain', 'row_count', 'column_count']])

os.makedirs(r"C:\Code\reco-ag\outputs", exist_ok=True)
output_path = r"C:\Code\reco-ag\outputs\inventory_summary.csv"
inventory_df.to_csv(output_path, index=False)
print(f"\nInventory successfully saved to {output_path}")
