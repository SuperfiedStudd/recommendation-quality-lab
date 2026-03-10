# DiscoveryRank: Recommendation Quality Lab

A Python-first recommendation quality evaluation lab built on the KuaiRand-1K dataset.

This project is a modular evaluation framework intended to measure recommendation quality tradeoffs in a clean, explainable way. It is not a production serving system or a UI app, but rather an offline sandbox for comparing ranking strategies. 

Recommendation quality is inherently multi-dimensional. While raw engagement (clicks/completions) is easy to optimize for, purely engagement-driven rankers often trap users in filter bubbles. This lab evaluates rankers across six dimensions:
**Relevance**, **Freshness**, **Diversity**, **Repetition Risk**, **Novelty**, and **Serendipity**.

---

## 🌟 Key Findings & Tradeoffs

Through an iterative development process, we evaluated heuristic and learned strategies over realistic 100-item candidate pools. Our key findings highlight structural tradeoffs:

1. **Diversity Reranking Works**: A simple greedy diversity-aware reranker successfully doubles tag diversity and nearly eliminates consecutive repetitive content (`consecutive_tag_rate` dropped to almost 0) without sacrificing Top-20 relevance. It simultaneously achieves the highest serendipity score.
2. **Learned Baselines vs. Heuristics**: Our learned matrix factorization baseline (Truncated SVD) did not beat simple heuristics on immediate, proxy engagement signals. This is an honest, expected result in sparse short-video settings where heuristic rankers explicitly rely on engagement data.
3. **Sourcing Shifts**: While the SVD model lost on raw relevance, it shifted sourcing patterns considerably—surfacing significantly more *history-adjacent* (creator-matched) items than global-popularity approaches. This demonstrates how learned models implicitly personalize catalog discovery.

---

## 🏗️ Architecture & Pipeline

The project follows a standard offline recommender split: **Data Prep** → **Candidate Generation (Recall)** → **Ranking (Scoring)** → **Evaluation**.

### 1. Data Processing
Raw logs and metadata are merged (`src/data_prep.py`), organized into sessions (`src/session_builder.py`), and labeled for implicit/explicit relevance (`src/relevance_labels.py`). Item freshness features are attached globally based on video upload times (`src/freshness_features.py`).

### 2. Time-Safe Candidate Generation (`src/candidate_generation.py`)
To prevent data leakage, pools of 100 items per test session are generated using *strictly prior data*. Sourcing blends:
- **Observed items**: Actual ground truth interactions in the session.
- **Popular items**: Globally popular content from the recent past.
- **History items**: Unseen items from creators the user historically engaged with.
- **Random items**: Pure exploration baseline.

### 3. Ranking Strategies (`src/ranking_strategies.py` & `src/model_baselines.py`)
Candidates are scored and sorted by various rankers:
- Pure Popularity baseline
- Freshness-boosted (exponential age decay)
- Diversity-aware (greedy tag/author penalty reranker)
- Matrix Factorization Baseline (Scipy Truncated SVD)

### 4. Advanced Evaluation (`src/eval_metrics.py`)
Evaluates the ranked outputs on the 6 key dimensions: relevance, freshness (age), diversity (author/tag uniqueness), repetition (consecutive drops), novelty (global catalog rarity), and serendipity (relevance of novel items).

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

We use lightweight tools and avoid heavy dependencies (ALS matrix factorization is proxied by `scipy.sparse.linalg.svds` for compatibility).

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execution Order
The analysis is structured linearly through Jupyter Notebooks. Run them in this order:

1. `notebooks/01_pipeline_check.ipynb` — Builds the master interaction log (`outputs/pipeline_sample.csv`).
2. `notebooks/02_validation_checks.py/ipynb` — Validates labeling, missing data, and target features.
3. `notebooks/03_baseline_strategy_comparison.ipynb` — Phase 1: Reranks only observed ground-truth items.
4. `notebooks/04_candidate_pool_strategy_comparison.ipynb` — Phase 2: Reranks realistic 100-item multi-source pools.
5. `notebooks/05_ml_baseline_and_advanced_eval.ipynb` — Phase 3: Strict temporal splits, SVD baseline, and advanced novelty/serendipity metrics.

*Note: Final artifacts and tradeoff tables are written to the `outputs/` folder.*

---

## 📈 Project Phases

- **Phase 1 (Setup)**: End-to-end data pipeline, feature engineering, and evaluating heuristics on restricted pools.
- **Phase 2 (Realistic Retrieval)**: Transitioned from reranking ground-truths to retrieving 100 items from multi-source pools (Popular, History, Random), simulating a real recall step.
- **Phase 3 (Advanced Eval & ML)**: Added a strict temporal train/test split. Integrated advanced discovery metrics (Novelty, Coverage) and a learned matrix factorization baseline, tracking candidate sourcing provenance.

---

## ⚠️ Limitations & Future Scope
- **Dataset Boundaries**: KuaiRand-1K lacks rich semantic content features (e.g., embeddings or audio/visual data). Tags and Authors are the primary metadata vectors.
- **No Deployment**: This repository is designed exclusively for offline batch evaluation. It does not include serving infrastructure or online A/B testing components.
- **Future Improvements**: We could integrate semantic embedding distances directly into the diversity/repetition penalization steps if content vectors become available.
