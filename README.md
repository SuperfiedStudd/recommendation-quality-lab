# DiscoveryRank: Recommendation Quality Lab

I built this Python-first recommendation quality evaluation lab to objectively measure ranking tradeoffs using the KuaiRand-1K dataset. 

Recommendation quality is inherently multi-dimensional. While raw engagement (clicks, watch time) is easy to optimize for, purely engagement-driven systems often trap users in repetitive filter bubbles. This framework explicitly evaluates strategies beyond simple relevance, scoring them across six dimensions: **Relevance**, **Freshness**, **Diversity**, **Repetition Risk**, **Novelty**, and **Serendipity**. 

My main contribution here is not a production serving app, but rather a clean offline evaluation framework. The project systematically pipelines data preparation, multi-source candidate generation, and ranking, allowing for an honest, explainable comparison between simple heuristics and learned matrix factorization models.

---

## 🏗️ Architecture & Pipeline

The project follows a modular offline recommender architecture:

`Data Prep` → `Sessionization` → `Candidate Generation` → `Ranking` → `Evaluation`

1. **Data Prep & Sessionization**: Merges raw KuaiRand-1K logs and features (`src/data_prep.py`), dropping corrupted data, and sequences interactions into logical viewing sessions (`src/session_builder.py`).
2. **Feature Engineering**: Derives binary relevance (`src/relevance_labels.py`) and calculates time-relative item freshness based on upload timestamps (`src/freshness_features.py`).
3. **Candidate Generation**: To prevent data leakage, pools of 100 items per session are generated using strictly prior data (`src/candidate_generation.py`). The sourcing blends: *Observed*, *Popular*, *History-Adjacent*, and *Random* items.
4. **Ranking Baseline Models**: Candidates are scored by heuristic rankers (popularity, freshness-boosted, diversity-aware rerank in `src/ranking_strategies.py`) and a learned matrix factorization baseline (Scipy Truncated SVD in `src/model_baselines.py`).
5. **Evaluation**: Evaluates the ranked outputs on all 6 dimensions simultaneously (`src/eval_metrics.py`).

---

## 🌟 Key Findings & Tradeoffs (Phase 3 Results)

I evaluated heuristic and learned strategies over realistic 100-item candidate pools. The findings highlight structural tradeoffs rather than claiming a single "winner":

1. **Diversity Reranking Works**: A greedy diversity-aware reranker successfully doubled tag diversity and practically eliminated consecutive repetitive content without sacrificing Top-20 relevance. It simultaneously achieved the highest serendipity score.
2. **Learned Baselines vs. Heuristics**: The learned matrix factorization baseline (Truncated SVD) did not beat simple heuristics on immediate, proxy engagement signals. This is an honest, expected result in sparse short-video settings where heuristic rankers explicitly exploit raw engagement counts.
3. **Sourcing Shifts**: While the SVD model lost on raw relevance, it shifted sourcing patterns considerably—surfacing significantly more *history-adjacent* (creator-matched) items than global-popularity approaches. This demonstrates how learned models implicitly personalize catalog discovery, even if it hurts short-term proxy metrics.

---

## 📂 Repository Structure

```text
.
├── data/           # Requires KuaiRand-1K raw CSVs (not tracked)
├── docs/           # Design plans and metric definitions
├── eval/           # Evaluation scaffolding 
├── notebooks/      # Phased execution notebooks (see order below)
├── outputs/        # Generated samples, full strategy comparisons, session results
├── src/            # Core pipeline and strategy modules
└── requirements.txt
```

---

## 🚀 Setup & Reproduction

The setup uses lightweight tools and avoids heavy dependencies.

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execution Order
The analysis is structured linearly. I recommend running the notebooks in this exact order:

1. `notebooks/01_pipeline_check.ipynb` — Builds the master interaction log (`outputs/pipeline_sample.csv`).
2. `notebooks/02_validation_checks.ipynb` — Validates labeling, missing data, and target features.
3. `notebooks/03_baseline_strategy_comparison.ipynb` — Phase 1: Reranks only observed ground-truth items.
4. `notebooks/04_candidate_pool_strategy_comparison.ipynb` — Phase 2: Reranks realistic 100-item multi-source pools.
5. `notebooks/05_ml_baseline_and_advanced_eval.ipynb` — Phase 3: Strict temporal splits, SVD baseline, and advanced evaluation.

*Note: Final artifacts and tradeoff tables are written directly to the `outputs/` directory.*

---

## ⚠️ Limitations
- **Dataset Boundaries**: KuaiRand-1K lacks rich semantic content features (e.g., embeddings or audio/visual data). Tags and Authors are the primary metadata vectors.
- **Scope**: This repository is designed exclusively for offline batch evaluation. It does not include serving infrastructure or online A/B testing components. Future work could integrate semantic embedding distances directly into the diversity penalization steps.
