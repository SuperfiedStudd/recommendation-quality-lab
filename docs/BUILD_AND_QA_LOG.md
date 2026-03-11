# Build & QA Log — Recommendation Strategy Lab

## Session: 2026-03-11

### Entry 1 — 12:08 PM — Infrastructure Setup
- **Action:** Created `docs/screenshots/` directory structure and QA log files.
- **Result:** ✅ Directories created successfully.
- **Next Step:** Test CLI --help output.

### Entry 2 — 12:08 PM — CLI Help Validation
- **Action:** Ran `python -m src.run_experiment --help`.
- **Result:** ✅ Help text renders correctly. All flags listed: `--preset`, `--strategy`, `--override`, `--compare`, `--compare-last`, `--open-report`.
- **Next Step:** Run multi-scenario CLI experiments.

### Entry 3 — 12:08 PM — CLI Scenario Run: popularity_trap + balanced_discovery
- **Action:** `python -m src.run_experiment --preset popularity_trap --strategy balanced_discovery --open-report`
- **Result:** ✅ Completed successfully. CTR=0.4255, diversity=8.96, coverage=9.4%. Artifacts saved to `outputs/runs/`.
- **Next Step:** Run second scenario.

### Entry 4 — 12:08 PM — CLI Scenario Run: cold_start_catalog + freshness_boost
- **Action:** `python -m src.run_experiment --preset cold_start_catalog --strategy freshness_boost --compare-last --open-report`
- **Result:** ✅ Run completed. Comparison delta table printed. Comparison plot generated.
- **Next Step:** Verify artifact folder structure.

### Entry 5 — 12:09 PM — Artifact Verification
- **Action:** Listed all files in `outputs/runs/`.
- **Result:** ✅ Each run folder contains `config.yaml`, `metrics.csv`, `recommendations.csv`, `summary.md`, and `plots/metrics_summary.png`. Index CSV is kept in sync.
- **Next Step:** Launch Streamlit for UI testing.

### Entry 6 — 12:09 PM — Streamlit Launch & Tab Verification
- **Action:** Launched `streamlit run app.py` and browsed all 3 tabs.
- **Result:** ✅ Run Lab, History, and Compare Runs tabs all render correctly. Presets/strategies populate from YAML. History table shows all CLI experiments.
- **Weakness Found:** Compare Runs chart has overlapping x-axis labels (run IDs too long).
- **Screenshots:** `03_streamlit_home.png`, `04_history_view.png`, `05_compare_runs_view.png`

### Entry 7 — 12:13 PM — Bug Fix: Comparison Chart Labels
- **Action:** Refactored `src/plotting.py` to use short "Run A / Run B" labels with value annotations and a legend below the figure.
- **Result:** ✅ CLI comparison plots now have clean labels.
- **Bug Introduced:** NameError — `df` referenced before creation (DataFrame line was accidentally removed during refactor).

### Entry 8 — 12:13 PM — Enriched Summary Report
- **Action:** Updated `src/history.py` to add moderate-bracket interpretations for all 5 metrics (CTR, Watch Time, Diversity, Coverage, Creator Spread) and a Tradeoff Analysis section.
- **Result:** ✅ Summary now includes actionable product-oriented interpretations.
- **Weakness Fixed:** Previously, diversity values between 5-15 had no interpretation text.

### Entry 9 — 12:15 PM — UI Run Simulation Flow
- **Action:** Selected `sparse_user_history + creator_diversity` in-browser and clicked Run Simulation.
- **Result:** ✅ Simulation ran, balloons animation played, summary report rendered in-app with metrics table and charts.
- **Screenshot:** `06_run_completed_results.png`

### Entry 10 — 12:15 PM — Bug Found: NameError in Compare Runs
- **Action:** Clicked Compare in the Streamlit Compare Runs tab.
- **Result:** ❌ `NameError: name 'df' is not defined` in `plotting.py:70`.
- **Root Cause:** DataFrame creation line was deleted during the label refactor.
- **Screenshot:** `milestone_3_error.png`

### Entry 11 — 12:17 PM — Bug Fix: NameError
- **Action:** Re-added `df = pd.DataFrame([metrics_1, metrics_2])` before label assignment.
- **Result:** ✅ Comparison chart and delta table now render correctly.

### Entry 12 — 12:18 PM — Final Verification: Compare Runs Fixed
- **Action:** Re-tested Compare Runs in Streamlit after fix.
- **Result:** ✅ Clean Run A/B chart with value annotations. Delta table with all 5 metrics shown with precise deltas.
- **Screenshots:** `07_compare_runs_fixed.png`, `08_delta_table.png`

### Entry 13 — 12:20 PM — Test Suite
- **Action:** `pytest tests/ -v`
- **Result:** ✅ 7 passed, 2 pre-existing fixture errors in `test_online_loop.py` (unrelated to our changes).
- **Lab tests:** Both `test_config_loading` and `test_save_run` pass.

## Summary of Issues Found & Fixed

| # | Issue | Severity | Fixed? |
|---|-------|----------|--------|
| 1 | Comparison chart x-axis labels overlap due to long run IDs | Medium | ✅ Used Run A/B + legend |
| 2 | Summary.md missing interpretation for diversity 5-15 range | Low | ✅ Added moderate bracket |
| 3 | Summary.md missing watch time and creator spread interpretations | Low | ✅ Added all 5 metrics |
| 4 | NameError in plotting.py after chart label refactor | Critical | ✅ Re-added DataFrame creation |
| 5 | Em-dash encoding issue in summary.md Tradeoff Analysis | Low | ✅ Replaced with ASCII dashes |
| 6 | Pydantic missing from requirements.txt | Medium | ✅ Added |
