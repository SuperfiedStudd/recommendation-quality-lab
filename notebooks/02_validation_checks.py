#!/usr/bin/env python
# coding: utf-8

# # Phase 2: Pipeline Validation & Sanity Checks
# 
# This notebook comprehensively tests the output of `pipeline_sample.csv` to ensure the data mapping 
# and derived features are behaving logically before I begin evaluating models.

# In[ ]:


import sys
import os
import pandas as pd
import numpy as np

# Add src dynamically
sys.path.append(os.path.abspath('../src'))
import validation_checks


# ## 1. Global Shape & Types

# In[ ]:


df = pd.read_csv('../outputs/pipeline_sample.csv')
print(f"Pipeline output shape: {df.shape}\n")
print("Data Types:\n", df.dtypes)


# ## 2. Nulls & Distributions

# In[ ]:


print("Top 10 Null Columns:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

print("\nNumeric Summary Stats (Focus columns):")
cols = ['implicit_completion_ratio', 'item_age_days', 'play_time_ms', 'duration_ms']
print(df[cols].describe())

print("\ny_relevant Distribution:")
print(df['y_relevant'].value_counts(normalize=True))


# ## 3. Session Statistics

# In[ ]:


sess_lengths = df.groupby('session_id').size()
print("Session Length Distribution:")
print(sess_lengths.describe())

# Flag overly massive sessions (e.g. > 500 impressions in a single 30-min bounded timeframe)
print(f"\nSessions with > 200 items: {(sess_lengths > 200).sum()}")


# ## 4. Logical Contradiction Checking

# In[ ]:


masks = validation_checks.check_logical_contradictions(df)

print("Contradiction Report (Violation Counts):\n")
for key, mask in masks.items():
    print(f"  - {key}: {mask.sum()} flags")


# ## 5. Explicit Edge Case Review Output

# In[ ]:


review_df = validation_checks.sample_edge_cases(df, masks)

cols_to_view = ['user_id', 'video_id', 'time_ms', 'session_id', 'is_hate', 'y_relevant', 
                'duration_ms', 'play_time_ms', 'implicit_completion_ratio', 
                'upload_dt', 'item_age_days', 'is_edge_case', 'edge_case_reason']

output_path = '../outputs/validation_sample_rows.csv'
review_df[cols_to_view].to_csv(output_path, index=False)
print(f"Exported {len(review_df)} sample rows (mixed edge cases and normals) to {output_path}.")

