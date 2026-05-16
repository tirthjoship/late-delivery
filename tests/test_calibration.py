"""Tests for calibration adapter."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from adapters.ml.calibration import CalibratedModelAdapter
from adapters.ml.sklearn_predictor import XGBoostPredictor


class TestCalibratedModelAdapter:
    def test_wraps_model_and_calibrates(self) -> None:
        """Calibrated adapter should produce probabilities in valid range."""
        X, y = make_classification(n_samples=500, random_state=42)
        X_train, X_cal, X_test = X[:300], X[300:400], X[400:]
        y_train, y_cal = y[:300], y[300:400]

        base_model = XGBoostPredictor()
        base_model.train(X_train, y_train)

        calibrated = CalibratedModelAdapter(base_model, method="isotonic")
        calibrated.calibrate(X_cal, y_cal)

        proba = calibrated.predict_proba(X_test)
        assert proba.shape == (100,)
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_predict_delegates_to_base(self) -> None:
        """predict() delegates to base model unchanged."""
        X, y = make_classification(n_samples=200, random_state=42)
        base_model = XGBoostPredictor()
        base_model.train(X[:150], y[:150])

        calibrated = CalibratedModelAdapter(base_model, method="isotonic")
        calibrated.calibrate(X[150:180], y[150:180])

        preds = calibrated.predict(X[180:])
        base_preds = base_model.predict(X[180:])
        np.testing.assert_array_equal(preds, base_preds)

    def test_get_params_includes_calibration_method(self) -> None:
        """get_params() includes calibration_method."""
        X, y = make_classification(n_samples=200, random_state=42)
        base_model = XGBoostPredictor()
        base_model.train(X[:150], y[:150])

        calibrated = CalibratedModelAdapter(base_model, method="isotonic")
        calibrated.calibrate(X[150:180], y[150:180])

        params = calibrated.get_params()
        assert params["calibration_method"] == "isotonic"

    def test_invalid_method_raises(self) -> None:
        """Invalid calibration method raises ValueError."""
        X, y = make_classification(n_samples=200, random_state=42)
        base_model = XGBoostPredictor()
        base_model.train(X[:150], y[:150])

        calibrated = CalibratedModelAdapter(base_model, method="invalid")
        with pytest.raises(ValueError, match="method"):
            calibrated.calibrate(X[150:180], y[150:180])

    def test_model_property_accesses_base(self) -> None:
        """model property accesses underlying model for SHAP compat."""
        X, y = make_classification(n_samples=100, random_state=42)
        base_model = XGBoostPredictor()
        base_model.train(X[:80], y[:80])

        calibrated = CalibratedModelAdapter(base_model, method="isotonic")
        assert calibrated.model is base_model.model
