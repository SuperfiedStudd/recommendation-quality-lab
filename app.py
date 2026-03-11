import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import yaml
import subprocess

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(__file__))
from src.config import PROJECT_ROOT
from src.history import get_run_index, get_run_details
from src.plotting import generate_comparison_plot

st.set_page_config(page_title="Recommendation Strategy Lab", layout="wide")

def load_yaml_names(folder: str):
    path = PROJECT_ROOT / "configs" / folder
    if not path.exists():
        return []
    files = [f.stem for f in path.glob("*.yaml")]
    return sorted(files)

def run_experiment(preset, strategy, custom_overrides):
    cmd = [sys.executable, "-m", "src.run_experiment", "--preset", preset, "--strategy", strategy]
    for k, v in custom_overrides.items():
        cmd.extend(["--override", f"{k}={v}"])
        
    with st.spinner(f"Running scenario '{preset}' with strategy '{strategy}'..."):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Run completed successfully!")
            return True
        else:
            st.error(f"Run failed:\n{result.stderr}")
            return False

st.title("🧪 Recommendation Strategy Lab")

tab_run, tab_history, tab_compare = st.tabs(["▶️ Run Lab", "📚 History", "⚖️ Compare Runs"])

with tab_run:
    st.markdown("Select a scenario and strategy to simulate user interactions and evaluate performance.")
    
    col1, col2 = st.columns(2)
    with col1:
        presets = load_yaml_names("presets")
        selected_preset = st.selectbox("Scenario Preset", presets, help="Defines environment, user behavior, and candidate pool parameters.")
        
    with col2:
        strategies = load_yaml_names("strategies")
        selected_strategy = st.selectbox("Ranking Strategy", strategies, help="Defines how candidates are scored and ranked.")
        
    with st.expander("Advanced Settings (Overrides)"):
        st.markdown("Override internal simulation config values (leave blank for defaults).")
        st.info("Examples: events=5000, top_k=10, eval_sessions=100")
        overrides_input = st.text_input("Overrides (comma separated key=value)", "")
        
    if st.button("Run Simulation", type="primary"):
        custom_overrides = {}
        if overrides_input.strip():
            pairs = overrides_input.split(",")
            for p in pairs:
                if "=" in p:
                    k, v = p.split("=", 1)
                    custom_overrides[k.strip()] = v.strip()
                    
        success = run_experiment(selected_preset, selected_strategy, custom_overrides)
        if success:
            st.balloons()
            # Try to show the latest result
            df_hist = get_run_index()
            if not df_hist.empty:
                latest_run_id = df_hist.iloc[-1]["run_id"]
                st.subheader("Latest Result")
                details = get_run_details(latest_run_id)
                st.markdown(details.get("summary", "Summary not generated."))
                
                plots_dir = PROJECT_ROOT / "outputs" / "runs" / latest_run_id / "plots"
                summary_plot = plots_dir / "metrics_summary.png"
                if summary_plot.exists():
                    st.image(str(summary_plot))

with tab_history:
    df_hist = get_run_index()
    if df_hist.empty:
        st.write("No experiment history found. Run a simulation first!")
    else:
        # Sort by timestamp descending
        df_hist = df_hist.sort_values("timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(df_hist)
        
        st.markdown("### View Specific Run")
        run_to_view = st.selectbox("Select Run ID", df_hist["run_id"].tolist())
        if run_to_view:
            details = get_run_details(run_to_view)
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(details.get("summary", "Summary not found."))
            with c2:
                plot_path = PROJECT_ROOT / "outputs" / "runs" / run_to_view / "plots" / "metrics_summary.png"
                if plot_path.exists():
                    st.image(str(plot_path))

with tab_compare:
    df_hist = get_run_index()
    if len(df_hist) < 2:
        st.write("Need at least 2 runs in history to compare.")
    else:
        run_ids = df_hist.sort_values("timestamp", ascending=False)["run_id"].tolist()
        col1, col2 = st.columns(2)
        with col1:
            run_1 = st.selectbox("Baseline Run", run_ids, index=1 if len(run_ids)>1 else 0)
        with col2:
            run_2 = st.selectbox("Comparison Run", run_ids, index=0)
            
        if st.button("Compare"):
            st.markdown(f"Comparing **{run_1}** and **{run_2}**")
            plot_path = generate_comparison_plot(run_1, run_2)
            if plot_path and plot_path.exists():
                st.image(str(plot_path))
                
            # Delta table
            d1 = get_run_details(run_1).get("metrics", {})
            d2 = get_run_details(run_2).get("metrics", {})
            
            deltas = []
            for k in d1.keys():
                v1 = d1.get(k, 0)
                v2 = d2.get(k, 0)
                if isinstance(v1, float):
                    diff = v2 - v1
                    deltas.append({"Metric": k, run_1: v1, run_2: v2, "Delta": diff})
                    
            if deltas:
                st.dataframe(pd.DataFrame(deltas))
