"""SHAP explainability adapter.

Implements ExplainerPort. Supports TreeExplainer (XGBoost)
and LinearExplainer (LogisticRegression). Auto-detects model type.
"""

from dataclasses import dataclass

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass
class ShapGlobalResult:
    """Global SHAP explanation result."""

    feature_importances: dict[str, float]
    """Mean absolute SHAP values per feature."""

    shap_values: np.ndarray
    """Raw SHAP values matrix (n_samples x n_features)."""


@dataclass
class ShapLocalResult:
    """Local SHAP explanation for a single prediction."""

    shap_values: dict[str, float]
    """SHAP value per feature for this prediction."""

    expected_value: float
    """Base value (average model output on training data)."""


class ShapExplainer:
    """SHAP wrapper for tree and linear models.

    Implements ExplainerPort. Auto-detects model type:
    - XGBClassifier -> shap.TreeExplainer
    - LogisticRegression -> shap.LinearExplainer
    """

    def __init__(
        self,
        model: XGBClassifier | LogisticRegression,
        feature_names: list[str],
    ) -> None:
        self._feature_names = feature_names
        if isinstance(model, XGBClassifier):
            self._explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            self._explainer = shap.LinearExplainer(
                model, np.zeros((1, len(feature_names)))
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    def explain_global(self, X: np.ndarray) -> ShapGlobalResult:
        """Compute global feature importance via mean absolute SHAP values."""
        shap_values = self._get_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        importances = {
            name: float(val) for name, val in zip(self._feature_names, mean_abs)
        }
        return ShapGlobalResult(
            feature_importances=importances,
            shap_values=shap_values,
        )

    def explain_local(self, X: np.ndarray, index: int) -> ShapLocalResult:
        """Compute SHAP explanation for a single prediction."""
        shap_values = self._get_shap_values(X)
        local_values = {
            name: float(val)
            for name, val in zip(self._feature_names, shap_values[index])
        }
        ev = self._explainer.expected_value
        # Binary classifiers may return array of expected values (one per class)
        expected = (
            float(ev[1])
            if isinstance(ev, (list, np.ndarray))
            and np.asarray(ev).ndim > 0
            and len(ev) > 1
            else float(ev)
        )
        return ShapLocalResult(shap_values=local_values, expected_value=expected)

    def _get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values, handling different explainer output formats."""
        sv = self._explainer.shap_values(X)
        # Newer SHAP versions may return Explanation objects
        if hasattr(sv, "values"):
            sv = sv.values
        # TreeExplainer for binary classification may return list of 2 arrays
        if isinstance(sv, list):
            result: np.ndarray = sv[1]  # positive class
            return result
        result_arr: np.ndarray = np.asarray(sv)
        return result_arr
