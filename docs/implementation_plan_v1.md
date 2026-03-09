# DiscoveryRank: Implementation Plan v1

This document maps the available raw KuaiRand-1K data into the offline evaluation framework for our four key dimensions. 

## 1. Data to Evaluation Mapping

### Relevance
* **Explicit Positives:** Use `is_like`, `is_follow`, `is_forward`, and `is_comment` from `log_*` files to indicate a highly relevant item.
* **Explicit Negatives:** Use `is_hate` from `log_*` to indicate an explicit failure.
* **Implicit Relevance:** Use `play_time_ms` and `duration_ms` from `log_*` to measure implied interest. 

### Freshness
* **Item Creation:** Use `upload_dt` from `video_features_basic_1k.csv`.
* **Interaction Time:** Use `time_ms` from `log_*`.

### Diversity
* **Item Categories:** Use `tag` and `music_id` from `video_features_basic_1k.csv`.
* **Creator Diversity:** Use `author_id` from `video_features_basic_1k.csv`.

### Repetition Risk
* **Sequence Tracking:** Use `video_id`, `author_id`, and timeline ordering (`time_ms`) to detect consecutive similarity or historical repetition.

## 2. Derived Features to Create

* **Implicit Completion Ratio:** Derive by calculating `play_time_ms / duration_ms`. Cap at 1.0 (or a small multiple like 2.0 to handle caching/looping artifacts) to prevent extreme outliers skewing averages.
* **Item Age (Delta):** Derive `age_ms` by converting `upload_dt` into an epoch timestamp and subtracting it from the log's `time_ms`.
* **Session ID:** Derive internal `session_id` constraints by chronologically sorting a user's `time_ms` events and artificially splitting into a new session if the difference between row $N$ and row $N-1$ exceeds a strict threshold (e.g., 30 minutes).

## 3. Quiet Implementation Assumptions

* **Missing Upload Dates:** When `upload_dt` is missing (58 nulls), we will fall back to the first observed interaction `time_ms` for that `video_id` in the dataset, under the assumption that the video was freshly uploaded just prior to its first view.
* **Missing Tags:** We assume missing `tag` values represent a valid single "Unknown" cluster, rather than treating them as perfectly distinct.
* **Impression Guarantee:** We assume every row in the `log_*` dataset represents an item that was actually presented and visible to the user, regardless of `play_time_ms`.
* **Unordered Sets vs Ranked Lists:** Because we lack the exact ranked sequence of the impression payload, we will initially evaluate the grouped "session" items as an unordered set for diversity and freshness metrics, and use chronological interaction as a loose proxy for rank.

## 4. Internal Modules to Build (src/)

* **`data_prep.py`**: Should output clean, joined, and imputed master dataframes. Merges basic video features and user features onto the logs.
* **`session_builder.py`**: Should output the same log dataframes but strictly partitioned with a new `session_id` column based on time gaps.
* **`relevance_labels.py`**: Should output consolidated binary or continuous target variables (`y_relevant`) for a given log row.
* **`freshness_features.py`**: Should output the delta `age_ms` array.
* **`diversity_features.py`**: Should output clustered group IDs or vectorized categorical representations for distance calculations.
* **`eval_metrics.py`**: Should output the mathematical aggregate scores (e.g., Diversity Score, Average Freshness) given a sequence or set of items.
* **`ranking_strategies.py`**: Should output dummy ranked lists (e.g., Random, Most Popular, Chronological) for us to battle-test the `eval_metrics.py` system against realistic baselines.
