# DiscoveryRank

A Python-first recommendation quality evaluation lab built on KuaiRand-1K interaction data.
Evaluates ranking strategies across six dimensions: **Relevance**, **Freshness**, **Diversity**, **Repetition Risk**, **Novelty**, and **Serendipity**.

## Directory Structure
- `data/` — KuaiRand-1K raw CSVs (interaction logs, user/video features)
- `src/` — Pipeline modules: data prep, feature engineering, candidate generation, ranking strategies, ML baselines, evaluation metrics
- `notebooks/` — Phased analysis notebooks (01–05)
- `outputs/` — Pipeline artifacts, strategy comparison CSVs, per-session results
- `docs/` — Design documents, metric definitions, implementation plans
- `eval/` — Evaluation scripts (planned)

## Core Modules

| Module | Purpose |
|---|---|
| `data_prep.py` | Loads and merges KuaiRand CSVs into a unified interaction log |
| `relevance_labels.py` | Derives binary `y_relevant` from explicit/implicit signals |
| `freshness_features.py` | Computes `item_age_days` from upload timestamps |
| `candidate_generation.py` | Multi-source candidate pool generator (observed, popular, history-adjacent, random) |
| `ranking_strategies.py` | Heuristic rankers: popularity, freshness-boosted, diversity-aware rerank |
| `model_baselines.py` | SVD matrix factorization baseline (scipy truncated SVD) |
| `eval_metrics.py` | Six-dimension evaluation: relevance, freshness, diversity, repetition, novelty, serendipity |

## Notebooks

1. **01** — Pipeline sanity check
2. **02** — Validation checks
3. **03** — Baseline strategy comparison (observed-only pools)
4. **04** — Candidate pool strategy comparison (N=100 pools)
5. **05** — ML baseline + advanced evaluation (temporal split, SVD, novelty/serendipity/coverage)

## Setup
```
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
