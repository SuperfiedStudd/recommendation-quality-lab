# Metric Definitions (First-Pass Draft)

This document outlines the structured mathematical and programmatic targets for the DiscoveryRank evaluation framework. These metrics are strictly constrained by the data available in the KuaiRand logs.

## 1. Relevance Metrics
* **Positive Interaction Rate (PIR):** A binary indicator evaluating if a recommended list generated *any* explicit positive interaction. 
  * *Implementation:* `max(is_like, is_follow, is_forward, is_comment)` across the generated list.
* **Mean Completion Rate (MCR):** The average implicit consumption depth of the recommended items.
  * *Implementation:* `average(clip(play_time_ms / duration_ms, 0, 1.0))`
* **Hate Rate:** Penalty metric representing the proportion of explicit negative feedback.
  * *Implementation:* `sum(is_hate) / list_length`

## 2. Freshness Metrics
* **Mean Item Age (MIA):** Calculates how stale the inventory is on average when presented to the user.
  * *Implementation:* `average(interaction_time - item_upload_time)` across the list. Lower is better.
* **Fresh Inventory Ratio:** The percentage of items in a recommended block that are newer than an arbitrary threshold (e.g., < 24 hours old).
  * *Implementation:* `count(item_age < 86400_000ms) / list_length`

## 3. Diversity Metrics
* **Tag Coverage:** Measures topical breadth within a session or list.
  * *Implementation:* `count(unique(tag)) / list_length`
* **Creator Entropy:** Measures the spread of authors within a feed. A feed of entirely one author has 0 entropy.
  * *Implementation:* Shannon entropy of the discrete distribution of `author_id` values in the list.

## 4. Repetition Risk Metrics
* **Consecutive Author Rate (CAR):** Detects if the ranking algorithm gets stuck recommending the same creator back-to-back.
  * *Implementation:* `count(author_id[i] == author_id[i-1]) / (list_length - 1)`
* **Historical Deja Vu Rate:** The percentage of recommended items that the user has already consumed in a *previous* session (where `play_time_ms` was significant).
  * *Implementation:* Requires a stateful lookup of the user's past `video_id` interactions prior to the current session boundary.
