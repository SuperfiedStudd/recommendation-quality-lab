import pytest
import pandas as pd
import os
import sys
import shutil
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, ExperimentConfig
from src.history import save_run, get_run_index, OUTPUTS_DIR, INDEX_PATH

@pytest.fixture
def clean_outputs():
    """Ensure a clean state for tests that write to outputs/"""
    # Create temp backups of index so we don't nuke the user's real experiment history
    backup_index = None
    if INDEX_PATH.exists():
        backup_index = pd.read_csv(INDEX_PATH)
        INDEX_PATH.unlink()
        
    yield
    
    # Restore user's real experiment history
    if backup_index is not None:
        backup_index.to_csv(INDEX_PATH, index=False)
    elif INDEX_PATH.exists():
        INDEX_PATH.unlink()

def test_config_loading():
    """Test that presets and strategies load correctly into Pydantic models."""
    config = load_config("cold_start_catalog", "popularity_first", {"events": "500"})
    
    assert isinstance(config, ExperimentConfig)
    assert config.scenario_name == "cold_start_catalog"
    assert config.strategy.name == "popularity"
    assert config.simulation.events == 500  # Overridden value

def test_save_run(clean_outputs):
    """Smoke test for artifact persistence and index writing."""
    config = load_config("sparse_user_history", "freshness_boost")
    metrics = {
        "ctr_proxy": 0.45,
        "diversity": 16.0,
        "catalog_coverage": 0.05
    }
    recs = pd.DataFrame([
        {"user_id": 1, "session_id": "a", "video_id": 100, "rank": 1, "score": 2.5, "strategy": "mock"}
    ])
    
    run_id = save_run(config, metrics, recs)
    
    # Check folder creation
    run_dir = OUTPUTS_DIR / run_id
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "recommendations.csv").exists()
    assert (run_dir / "summary.md").exists()
    
    # Check index
    index_df = get_run_index()
    assert not index_df.empty
    assert len(index_df) == 1
    assert index_df.iloc[0]["run_id"] == run_id
    assert index_df.iloc[0]["preset"] == "sparse_user_history"
    assert index_df.iloc[0]["ctr_proxy"] == 0.45
    
    # Summary should be non-empty
    with open(run_dir / "summary.md", "r") as f:
        content = f.read()
        assert "Experiment Report" in content
        assert "High CTR Proxy" in content
        assert "High Diversity" in content
