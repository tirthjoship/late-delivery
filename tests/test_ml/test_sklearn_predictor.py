"""Tests for adapters/ml/sklearn_predictor.py.

Contract tests: verify both predictors implement ModelTrainerPort interface,
return valid predictions and probabilities.
"""

import numpy as np
import pytest

from adapters.ml.sklearn_predictor import LogisticRegressionPredictor, XGBoostPredictor


@pytest.fixture
def small_training_data():
    """50-row synthetic training data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    y = rng.integers(0, 2, size=50)
    return X, y


class TestLogisticRegressionPredictor:
    def test_train_no_error(self, small_training_data) -> None:
        X, y = small_training_data
        model = LogisticRegressionPredictor()
        model.train(X, y)

    def test_predict_returns_binary(self, small_training_data) -> None:
        X, y = small_training_data
        model = LogisticRegressionPredictor()
        model.train(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_returns_valid_probabilities(
        self, small_training_data
    ) -> None:
        X, y = small_training_data
        model = LogisticRegressionPredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert len(proba) == len(X)

    def test_predict_proba_shape(self, small_training_data) -> None:
        X, y = small_training_data
        model = LogisticRegressionPredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (50,)


class TestXGBoostPredictor:
    def test_train_no_error(self, small_training_data) -> None:
        X, y = small_training_data
        model = XGBoostPredictor()
        model.train(X, y)

    def test_predict_returns_binary(self, small_training_data) -> None:
        X, y = small_training_data
        model = XGBoostPredictor()
        model.train(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_returns_valid_probabilities(
        self, small_training_data
    ) -> None:
        X, y = small_training_data
        model = XGBoostPredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert len(proba) == len(X)

    def test_predict_proba_shape(self, small_training_data) -> None:
        X, y = small_training_data
        model = XGBoostPredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (50,)


class TestSameInterface:
    def test_both_have_train_predict_predict_proba(self) -> None:
        for cls in [LogisticRegressionPredictor, XGBoostPredictor]:
            instance = cls()
            assert hasattr(instance, "train")
            assert hasattr(instance, "predict")
            assert hasattr(instance, "predict_proba")

    def test_both_have_get_params(self) -> None:
        for cls in [LogisticRegressionPredictor, XGBoostPredictor]:
            instance = cls()
            params = instance.get_params()
            assert isinstance(params, dict)
            assert len(params) > 0

    def test_logreg_get_params_contains_expected_keys(self) -> None:
        model = LogisticRegressionPredictor(max_iter=500)
        params = model.get_params()
        assert params["max_iter"] == 500

    def test_xgboost_get_params_contains_expected_keys(self) -> None:
        model = XGBoostPredictor(max_depth=3, n_estimators=200, learning_rate=0.05)
        params = model.get_params()
        assert params["max_depth"] == 3
        assert params["n_estimators"] == 200
        assert params["learning_rate"] == 0.05
