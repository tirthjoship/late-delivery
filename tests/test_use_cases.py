"""Tests for application/use_cases.py.

Integration tests: verify use cases wire domain + adapters correctly.
Uses synthetic_orders fixture (100 orders, never real CSV).
"""

import numpy as np
import pytest

from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.shap_explainer import ShapExplainer
from adapters.ml.sklearn_predictor import LogisticRegressionPredictor, XGBoostPredictor
from application.use_cases import PredictSingleOrderUseCase, TrainAndEvaluateUseCase
from domain.models import MetricsResult, Order, PredictionResult, TrainingResult


class FakeSalesDataRepository:
    """Fake data repository for testing."""

    def __init__(self, orders: list[Order]) -> None:
        self._orders = orders

    def get_orders(self) -> list[Order]:
        return self._orders

    def get_products(self) -> list:
        return []


class FakeTracker:
    """Fake experiment tracker that records calls without MLflow."""

    def __init__(self) -> None:
        self.params: dict = {}
        self.metrics: dict = {}
        self.models: list = []
        self.artifacts: list = []
        self.runs: list = []
        self.registered: list = []

    def start_run(self, run_name: str) -> None:
        self.runs.append(run_name)

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update(metrics)

    def log_model(self, model, artifact_path: str) -> None:
        self.models.append(artifact_path)

    def log_artifact(self, artifact, artifact_path: str) -> None:
        self.artifacts.append(artifact_path)

    def register_model(self, model_name: str) -> None:
        self.registered.append(model_name)

    def end_run(self) -> None:
        pass

    def load_model(self, model_name: str, stage: str = "Production"):
        pass

    def load_artifact(self, run_id: str, artifact_path: str):
        pass


class TestTrainAndEvaluateUseCase:
    def test_returns_training_result(self, synthetic_orders) -> None:
        repo = FakeSalesDataRepository(synthetic_orders)
        encoder = FeatureEncoder()
        model = XGBoostPredictor()
        tracker = FakeTracker()

        use_case = TrainAndEvaluateUseCase(
            data_repo=repo,
            feature_encoder=encoder,
            model=model,
            explainer_factory=lambda m, names: ShapExplainer(m.model, names),
            tracker=tracker,
        )
        result = use_case.execute()
        assert isinstance(result, TrainingResult)
        assert isinstance(result.metrics, MetricsResult)
        assert 0.0 <= result.metrics.f1 <= 1.0
        assert 0.0 <= result.metrics.auc_roc <= 1.0
        assert len(result.feature_importances) > 0

    def test_tracker_receives_metrics(self, synthetic_orders) -> None:
        repo = FakeSalesDataRepository(synthetic_orders)
        encoder = FeatureEncoder()
        model = XGBoostPredictor()
        tracker = FakeTracker()

        use_case = TrainAndEvaluateUseCase(
            data_repo=repo,
            feature_encoder=encoder,
            model=model,
            explainer_factory=lambda m, names: ShapExplainer(m.model, names),
            tracker=tracker,
        )
        use_case.execute()
        assert "f1" in tracker.metrics
        assert "precision" in tracker.metrics
        assert "recall" in tracker.metrics
        assert "auc_roc" in tracker.metrics

    def test_works_with_logreg(self, synthetic_orders) -> None:
        repo = FakeSalesDataRepository(synthetic_orders)
        encoder = FeatureEncoder()
        model = LogisticRegressionPredictor()
        tracker = FakeTracker()

        use_case = TrainAndEvaluateUseCase(
            data_repo=repo,
            feature_encoder=encoder,
            model=model,
            explainer_factory=lambda m, names: ShapExplainer(m.model, names),
            tracker=tracker,
        )
        result = use_case.execute()
        assert isinstance(result, TrainingResult)


def test_execute_passes_custom_run_name(synthetic_orders):
    """Verify run_name param is forwarded to tracker.start_run()."""
    from unittest.mock import MagicMock

    from adapters.ml.feature_encoder import FeatureEncoder
    from adapters.ml.shap_explainer import ShapExplainer
    from adapters.ml.sklearn_predictor import XGBoostPredictor
    from application.use_cases import TrainAndEvaluateUseCase

    mock_tracker = MagicMock()
    repo = MagicMock()
    repo.get_orders.return_value = synthetic_orders

    use_case = TrainAndEvaluateUseCase(
        data_repo=repo,
        feature_encoder=FeatureEncoder(),
        model=XGBoostPredictor(),
        explainer_factory=lambda m, names: ShapExplainer(m.model, names),
        tracker=mock_tracker,
    )
    use_case.execute(run_name="xgboost-shallow")
    mock_tracker.start_run.assert_called_once_with(run_name="xgboost-shallow")


def test_execute_logs_model_hyperparams(synthetic_orders):
    """Verify model hyperparams are logged to tracker."""
    from unittest.mock import MagicMock

    from adapters.ml.feature_encoder import FeatureEncoder
    from adapters.ml.shap_explainer import ShapExplainer
    from adapters.ml.sklearn_predictor import XGBoostPredictor
    from application.use_cases import TrainAndEvaluateUseCase

    mock_tracker = MagicMock()
    repo = MagicMock()
    repo.get_orders.return_value = synthetic_orders

    model = XGBoostPredictor(max_depth=3, n_estimators=200)
    use_case = TrainAndEvaluateUseCase(
        data_repo=repo,
        feature_encoder=FeatureEncoder(),
        model=model,
        explainer_factory=lambda m, names: ShapExplainer(m.model, names),
        tracker=mock_tracker,
    )
    use_case.execute()

    # log_params is called twice: once for split params, once for hp_ params
    all_calls = mock_tracker.log_params.call_args_list
    assert len(all_calls) == 2
    hp_params = all_calls[1][0][0]
    assert "hp_max_depth" in hp_params
    assert hp_params["hp_max_depth"] == "3"
    assert hp_params["hp_n_estimators"] == "200"


class TestEncoderFitOnTrainOnly:
    """CRITICAL: verify encoder is fit on training data only."""

    def test_encoder_mean_matches_train_distribution(
        self, synthetic_orders_distinctive_sales
    ) -> None:
        """Output-based test: train has sales=1000, test has sales=0.
        Scaler mean should be ~1000 (train), not ~500 (combined).
        """
        from domain.services import extract_features

        train_orders, test_orders = synthetic_orders_distinctive_sales
        train_raw = [extract_features(o) for o in train_orders]
        test_raw = [extract_features(o) for o in test_orders]

        encoder = FeatureEncoder()
        X_train = encoder.fit_transform(train_raw)
        X_test = encoder.transform(test_raw)

        # After StandardScaler, train mean is 0.0 (centered)
        # Test values should be negative (0 - 1000)/std
        assert X_train["sales_per_customer"].mean() == pytest.approx(0.0, abs=0.01)
        assert X_test["sales_per_customer"].mean() < -1.0  # far below train mean


class TestPredictSingleOrderUseCase:
    def test_returns_prediction_result(self, synthetic_orders) -> None:
        from domain.services import extract_features

        # Train a model first
        raw = [extract_features(o) for o in synthetic_orders]
        labels = [o.late_delivery_risk for o in synthetic_orders]
        encoder = FeatureEncoder()
        X = encoder.fit_transform(raw)
        model = XGBoostPredictor()
        model.train(X.values, np.array(labels))
        explainer = ShapExplainer(model.model, encoder.get_feature_names())

        use_case = PredictSingleOrderUseCase(
            feature_encoder=encoder,
            model=model,
            explainer=explainer,
        )
        result = use_case.execute(synthetic_orders[0])
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.probability <= 1.0
        assert isinstance(result.risk_label, bool)
        assert len(result.explanation) > 0
