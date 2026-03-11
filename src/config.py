import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class SimulationConfig(BaseModel):
    events: int = Field(10000, description="Number of historical events to replay for state warmup")
    top_k: int = Field(20, description="Number of recommendations to serve per session")
    eval_sessions: int = Field(50, description="Number of sessions to simulate for evaluation")
    base_click_prob: float = Field(0.3, description="Base probability of clicking an item")
    base_watch_prob: float = Field(0.5, description="Base probability of watching after clicking")
    like_prob: float = Field(0.05, description="Probability of liking after watching")
    save_prob: float = Field(0.03, description="Probability of saving after watching")
    history_pool_ratio: float = Field(0.8, description="Max proportion of candidate pool to fill with history-adjacent items")
    popular_pool_ratio: float = Field(0.5, description="Max proportion of candidate pool to fill with popular items")

class StrategyConfig(BaseModel):
    name: str = Field(description="Name of the ranking strategy (e.g., popularity, freshness_boosted)")
    strategy_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the strategy function")

class ExperimentConfig(BaseModel):
    scenario_name: str = Field(description="Name of the scenario or preset config")
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)

def load_config(scenario_name: str, strategy_name: str, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """
    Loads an ExperimentConfig by combining a scenario preset, a strategy preset, and manual overrides.
    """
    preset_path = PROJECT_ROOT / "configs" / "presets" / f"{scenario_name}.yaml"
    strategy_path = PROJECT_ROOT / "configs" / "strategies" / f"{strategy_name}.yaml"

    config_dict = {
        "scenario_name": scenario_name,
        "simulation": {},
        "strategy": {"name": strategy_name}
    }

    if preset_path.exists():
        with open(preset_path, "r") as f:
            preset_data = yaml.safe_load(f) or {}
            if "simulation" in preset_data:
                config_dict["simulation"].update(preset_data["simulation"])

    if strategy_path.exists():
        with open(strategy_path, "r") as f:
            strat_data = yaml.safe_load(f) or {}
            config_dict["strategy"].update(strat_data)

    # Apply flat overrides if provided (e.g. from CLI `key=value`)
    if overrides:
        for k, v in overrides.items():
            if k in SimulationConfig.model_fields:
                config_dict["simulation"][k] = float(v) if '.' in str(v) else int(v) if str(v).isdigit() else v
            else:
                config_dict["strategy"]["strategy_kwargs"][k] = float(v) if '.' in str(v) else int(v) if str(v).isdigit() else v

    return ExperimentConfig(**config_dict)

def resolve_path(relative_path: str) -> Path:
    """Resolves a path relative to project root."""
    return PROJECT_ROOT / relative_path
