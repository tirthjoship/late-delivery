"""Tests for adapters/ml/mlflow_tracker.py.

Integration tests using tmp_path as tracking URI.
Real MLflow calls, auto-cleanup via pytest tmp_path fixture.
"""

from adapters.ml.mlflow_tracker import MLflowTracker


class TestMLflowTracker:
    def test_start_and_end_run_no_error(self, tmp_path) -> None:
        tracker = MLflowTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="test-run")
        tracker.end_run()

    def test_log_params(self, tmp_path) -> None:
        tracker = MLflowTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="test-params")
        tracker.log_params({"n_estimators": "100", "max_depth": "6"})
        tracker.end_run()

    def test_log_metrics(self, tmp_path) -> None:
        tracker = MLflowTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="test-metrics")
        tracker.log_metrics({"f1": 0.85, "precision": 0.82, "recall": 0.88})
        tracker.end_run()

    def test_log_model_creates_artifact(self, tmp_path) -> None:
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

        tracker = MLflowTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="test-model")
        tracker.log_model(model, "model")
        tracker.end_run()

    def test_full_workflow(self, tmp_path) -> None:
        """End-to-end: start -> params -> metrics -> model -> register -> end."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

        tracker = MLflowTracker(
            experiment_name="test-e2e",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="e2e-run")
        tracker.log_params({"solver": "lbfgs"})
        tracker.log_metrics({"f1": 0.90})
        tracker.log_model(model, "model")
        tracker.register_model("test-model")
        tracker.end_run()
