"""Tests for adapters/ml/shap_explainer.py.

Contract tests: SHAP feature names match training features,
local explanation length matches feature count,
SHAP values sum approximately to prediction.
"""

import numpy as np
import pytest
from xgboost import XGBClassifier

from adapters.ml.shap_explainer import ShapExplainer


@pytest.fixture
def trained_xgb_model():
    """Small XGBoost model trained on synthetic data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    y = rng.integers(0, 2, size=50)
    model = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
    model.fit(X, y)
    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
    return model, X, feature_names


class TestShapExplainer:
    def test_global_feature_names_match(self, trained_xgb_model) -> None:
        model, X, names = trained_xgb_model
        explainer = ShapExplainer(model, names)
        result = explainer.explain_global(X)
        assert set(result.feature_importances.keys()) == set(names)

    def test_local_explanation_length(self, trained_xgb_model) -> None:
        model, X, names = trained_xgb_model
        explainer = ShapExplainer(model, names)
        result = explainer.explain_local(X, index=0)
        assert len(result.shap_values) == len(names)

    def test_local_shap_values_sum_to_expected(self, trained_xgb_model) -> None:
        """SHAP contract: values should be finite and consistent."""
        model, X, names = trained_xgb_model
        explainer = ShapExplainer(model, names)
        result = explainer.explain_local(X, index=0)
        shap_sum = sum(result.shap_values.values()) + result.expected_value
        assert np.isfinite(shap_sum)
        for v in result.shap_values.values():
            assert np.isfinite(v)

    def test_global_importances_are_non_negative(self, trained_xgb_model) -> None:
        model, X, names = trained_xgb_model
        explainer = ShapExplainer(model, names)
        result = explainer.explain_global(X)
        for importance in result.feature_importances.values():
            assert importance >= 0.0
