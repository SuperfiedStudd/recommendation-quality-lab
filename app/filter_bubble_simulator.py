"""
Filter Bubble Simulator — DiscoveryRank
Author: Jasjyot Singh

An interactive Streamlit app that demonstrates how different ranking
strategies influence user exposure diversity over repeated sessions.

All paths are resolved relative to the project root via pathlib,
so this works portably regardless of where Streamlit is invoked from.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Portable path setup — project root is repo root (parent of app/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config, resolve_path
import ranking_strategies

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Filter Bubble Simulator", layout="wide")

# ---------------------------------------------------------------------------
# Load data using config-driven path
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    config = load_config()
    data_cfg = config.get("data", {})

    # Primary: use processed pipeline sample from outputs/
    rel_path = data_cfg.get("pipeline_sample", "outputs/pipeline_sample.csv")
    data_path = resolve_path(rel_path)

    if not data_path.exists():
        st.error(
            f"Dataset not found at `{rel_path}` (resolved to `{data_path}`).\n\n"
            "Please run notebook 01 or `python run_all.py` first to generate "
            "`outputs/pipeline_sample.csv`."
        )
        return pd.DataFrame()

    df = pd.read_csv(data_path)

    # Tag handling: use real tags if available, else create a documented proxy
    if "tag" not in df.columns or df["tag"].isnull().all():
        # Concentration proxy: no real category column found in the dataset.
        # We assign synthetic categories seeded deterministically so the
        # simulation is reproducible. This is clearly a proxy, not ground truth.
        np.random.seed(config.get("random_seed", 42))
        df["tag"] = np.random.choice(
            ["Gaming", "Music", "Tech", "Comedy", "News", "Sports"],
            size=len(df),
        )
        st.sidebar.warning(
            "ℹ️ Dataset lacks explicit category labels. "
            "Using a deterministic synthetic proxy for tag concentration."
        )
    else:
        df["tag"] = df["tag"].fillna("Unknown")

    if "implicit_completion_ratio" not in df.columns:
        np.random.seed(config.get("random_seed", 42))
        df["implicit_completion_ratio"] = np.random.uniform(0, 1, len(df))

    return df


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("DiscoveryRank: Filter Bubble Simulator")
st.markdown(
    "This interactive simulation demonstrates how different ranking strategies "
    "affect user exposure diversity over repeated sessions. Optimizing purely "
    "for engagement (Popularity) can lead to a *filter bubble* — a high "
    "concentration of identical categories — while a Diversity-Aware strategy "
    "preserves broad discovery over time."
)

df = load_data()
if df.empty:
    st.stop()

# Session state ---
if "clicked_history" not in st.session_state:
    st.session_state.clicked_history = []
if "exposure_history" not in st.session_state:
    st.session_state.exposure_history = []
if "step" not in st.session_state:
    st.session_state.step = 0

strategy = st.sidebar.selectbox(
    "Ranking Strategy",
    ["Popularity Based", "Diversity Aware (Penalty)", "Random Baseline"],
)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Session Controls")

    if st.button("Simulate Next Session (Serve 5 Items)"):
        pool = df.sample(100, replace=False, random_state=None).copy()

        # Feedback-loop bias: boost engagement score for items matching
        # the user's historically-clicked tags
        if st.session_state.clicked_history:
            past_tags = [item["tag"] for item in st.session_state.clicked_history]
            pool.loc[
                pool["tag"].isin(past_tags), "implicit_completion_ratio"
            ] += 0.5

        # Rank
        if strategy == "Popularity Based":
            ranked = pool.sort_values("implicit_completion_ratio", ascending=False)
        elif strategy == "Diversity Aware (Penalty)":
            ranked = ranking_strategies.diversity_aware_rerank(pool)
        else:
            ranked = pool.sample(frac=1.0)

        top_5 = ranked.head(5)
        top_1 = top_5.iloc[0].to_dict()
        st.session_state.clicked_history.append(top_1)
        st.session_state.exposure_history.extend(top_5.to_dict("records"))
        st.session_state.step += 1

    if st.button("Reset Simulation"):
        st.session_state.clicked_history = []
        st.session_state.exposure_history = []
        st.session_state.step = 0
        st.rerun()

with col2:
    if st.session_state.step > 0:
        exp_df = pd.DataFrame(st.session_state.exposure_history)

        st.subheader("Aggregate User Exposure")
        st.write(
            f"Total Sessions: **{st.session_state.step}** | "
            f"Items Exposed: **{len(exp_df)}**"
        )

        unique_tags = exp_df["tag"].nunique()
        tag_ratio = unique_tags / max(len(exp_df), 1)

        m1, m2, _ = st.columns(3)
        m1.metric("Unique Categories Seen", unique_tags)
        m2.metric("Diversity Ratio", f"{tag_ratio:.2f}")

        st.markdown("**Exposure Concentration by Category**")
        tag_counts = exp_df["tag"].value_counts()
        st.bar_chart(tag_counts, height=250)

        st.subheader("Latest User Click (Top Recommended Item)")
        last_click = pd.DataFrame([st.session_state.clicked_history[-1]])
        display_cols = [c for c in ["video_id", "tag", "implicit_completion_ratio"] if c in last_click.columns]
        st.dataframe(last_click[display_cols])
    else:
        st.info("Click **Simulate Next Session** to begin serving recommendations.")
