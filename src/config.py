from pathlib import Path
import yaml

# Project root is always the parent of the directory containing this file (src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(config_path=None):
    """
    Loads configuration from a YAML file.
    Falls back to configs/default_config.yaml relative to the project root.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}. Using hardcoded defaults.")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def resolve_path(relative_path):
    """
    Resolves a path from the config (relative to project root) to an absolute path.
    """
    return PROJECT_ROOT / relative_path
