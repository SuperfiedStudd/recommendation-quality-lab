# Data Fit for Recommendation Evaluation: KuaiRand-1K

## Project Goal
This project focuses on **recommendation-quality evaluation** rather than building a generic recommender system. It aims to evaluate how good a recommended set of content items is for a user across four key dimensions: **Relevance**, **Freshness**, **Diversity**, and **Repetition Risk**.

## How the Available Data Supports the Problem

### 1. Relevance
* **Primary Signals (Log files):** The engagement logs (`log_*`) contain rich behavioral signals that can help with relevance evaluation. Specifically, explicit positive interactions like `is_like`, `is_follow`, `is_forward`, and `is_comment` are strong indicators of relevance.
* **Implicit Signals (Log files):** The `play_time_ms`, `duration_ms`, and `long_view` fields can help with implicitly deriving relevance because users tend to watch relevant videos proportionally longer. I can calculate ratios like play time over duration to define a continuous implicit preference score.
* **Negative Signals:** The explicit `is_hate` column or abandoning a video quickly (low `play_time_ms` vs `duration_ms`) can help me heavily penalize rankings that surface irrelevant or disliked content.

### 2. Freshness
* **Item Age (Video Basic Features):** The `upload_dt` field in `video_features_basic_1k.csv` provides the general creation or upload time of the content.
* **Interaction Time (Log files):** The logs provide `date` (YYYYMMDD) and `time_ms` (UNIX epoch). 
* **Evaluation Fit:** This array of timestamps supports freshness analysis because I can calculate the exact age of a video at the moment a user interacted with it (Interaction `time_ms` - `upload_dt`). I can evaluate if ranking strategies over-index on stale content or appropriately surface freshly minted items.
* **Uncertainty:** `upload_dt` contains a small number of null values, requiring minor imputation or filtering before freshness metrics can be fully computed.

### 3. Diversity
* **Item Attributes (Video Basic Features):** Features like `tag`, `music_id`, and `video_type` provide basic categorizations of content.
* **Evaluation Fit:** This data likely supports diversity by allowing us to calculate intra-list diversity metrics. For instance, I can measure the uniqueness of `tag` or `music_id` within a user's session or top-K recommended items. If a ranking strategy only surfaces videos with identical tags to a specific user, it would score poorly on topical diversity.
* **Uncertainty & Weakness:** It is uncertain if the `tag` field is rich enough to capture true semantic diversity. The inspection showed 137k+ nulls for `tag`. Without a clearer topic taxonomy, the diversity metrics might be artificially hindered by missing data.

### 4. Repetition Risk
* **User-Item Interactions (Log files):** I have exact tracking of `video_id` appearances for a given `user_id` over time (`time_ms`).
* **Evaluation Fit:** This data can help with repetition risk because I can sequence a user's historical interactions chronologically. If a user is consecutively shown highly similar videos (e.g., same `author_id`, same `tag`) or even the exact same `video_id`, I can detect this and penalize the ranking strategy for high stringency / low discovery.

## Areas of Weakness and Missing Data

While the KuaiRand-1K dataset provides an excellent foundation for offline metrics, it has specific limitations for a comprehensive evaluation:

* **Missing Impression Position (Rank):** This data is weak for traditional ranking metric evaluations (like NDCG or MRR) because there is no explicit field indicating *where* in the feed the video was displayed (e.g., position 1 vs position 10). I only see what was interacted with, but miss the explicit ranked ordering presented to the user.
* **Missing Session Boundaries:** This data is weak for session-level analysis because explicit session IDs are missing. I will have to artificially infer session boundaries by looking for large time gaps in the `time_ms` interaction logs, which can be noisy.
* **Missing Clear Topic Taxonomy:** The `tag` field has many nulls and its structure is opaque. Additional robust item metadata (like hierarchical topics or NLP embeddings of the video content) would be highly helpful to genuinely measure semantic diversity.
* **Missing Explicit Preference Labels:** While `is_hate` is helpful, I lack explicit 1-5 star ratings or detailed qualitative feedback from users.
* **Limited Creator-Level Metadata:** Aside from `author_id` and basic item tallies, richer creator metadata (such as creator credibility, overall follower count history, or creator topic focus) would be helpful for evaluating fairness and creator-side objectives.
