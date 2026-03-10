from pathlib import Path
import mlflow

# Portable project root — works regardless of where the script is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_mlflow(config, run_name="pipeline_run"):
    """
    Sets up MLflow tracking with a local file-backed store inside the project.
    """
    mlruns_dir = PROJECT_ROOT / "mlruns"
    # Use forward slashes in the URI for cross-platform compatibility
    mlflow_uri = mlruns_dir.as_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = config.get("experiment_name", "discovery_rank_experiment")
    mlflow.set_experiment(experiment_name)

    # End any active runs just in case
    if mlflow.active_run():
        mlflow.end_run()

    return mlflow.start_run(run_name=run_name)


def log_experiment_params(config):
    """
    Logs relevant configuration parameters to MLflow.
    """
    if "evaluation" in config:
        mlflow.log_params(config["evaluation"])

    if "data" in config:
        mlflow.log_param("data_pipeline_sample", config["data"].get("pipeline_sample"))


def log_metrics_safely(metrics_dict, step=None):
    """
    Helper to log metrics while ignoring non-numeric values.
    """
    clean_metrics = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float, complex)) and not isinstance(v, bool):
            clean_metrics[k] = v

    if clean_metrics:
        mlflow.log_metrics(clean_metrics, step=step)


def log_local_artifact(file_path):
    """
    Logs a local file as an artifact in MLflow.
    """
    if Path(file_path).exists():
        mlflow.log_artifact(str(file_path))
