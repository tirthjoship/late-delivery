"""MLflow experiment tracking adapter.

Implements ExperimentTrackerPort. Local tracking with Model Registry.
tracking_uri parameter is the seam for future remote MLflow (Decision D3).
"""

import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """Local MLflow tracker with Model Registry.

    Implements ExperimentTrackerPort.
    """

    def __init__(
        self,
        experiment_name: str = "late-delivery-risk",
        tracking_uri: str = "sqlite:///mlflow.db",
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run: mlflow.ActiveRun | None = None
        self._run_id: str | None = None

    def start_run(self, run_name: str) -> None:
        self._run = mlflow.start_run(run_name=run_name)
        self._run_id = self._run.info.run_id
        logger.info(
            "MLflow run started: {} (experiment: {})", run_name, self._experiment_name
        )

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str) -> None:
        mlflow.sklearn.log_model(model, artifact_path)

    def log_artifact(self, artifact: Any, artifact_path: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"{artifact_path}.joblib"
            joblib.dump(artifact, path)
            mlflow.log_artifact(str(path))

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)

    def load_artifact(self, run_id: str, artifact_path: str) -> Any:
        client = MlflowClient(self._tracking_uri)
        local_path = client.download_artifacts(run_id, artifact_path)
        return joblib.load(local_path)

    def register_model(self, model_name: str) -> None:
        if self._run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        model_uri = f"runs:/{self._run_id}/model"
        mlflow.register_model(model_uri, model_name)

    def end_run(self) -> None:
        logger.info("MLflow run ended: {}", self._run_id)
        mlflow.end_run()
        self._run = None

    @property
    def run_id(self) -> str | None:
        return self._run_id
