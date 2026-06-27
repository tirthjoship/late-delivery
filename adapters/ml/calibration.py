"""Calibration adapter - wraps a trained model with isotonic/Platt scaling.

Used when calibration diagnostic reveals poor probability calibration (Brier > 0.15).
Deep-dive notebook found Brier=0.2028, triggering this adapter's creation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratedModelAdapter:
    """Post-hoc calibration wrapper for any model with predict_proba.

    Fits a calibration function on held-out calibration set.
    predict() delegates to base model. predict_proba() is calibrated.
    """

    def __init__(self, base_model: Any, method: str = "isotonic") -> None:
        self._base = base_model
        self._method = method
        self._calibrator: IsotonicRegression | LogisticRegression | None = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Fit calibration function on calibration set."""
        raw_proba = self._base.predict_proba(X_cal)
        if self._method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_proba, y_cal)
        elif self._method == "platt":
            self._calibrator = LogisticRegression()
            self._calibrator.fit(raw_proba.reshape(-1, 1), y_cal)
        else:
            msg = f"method must be 'isotonic' or 'platt', got '{self._method}'"
            raise ValueError(msg)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (delegates to base model)."""
        result: np.ndarray = self._base.predict(X)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        raw_proba: np.ndarray = self._base.predict_proba(X)
        if self._calibrator is None:
            return raw_proba
        if self._method == "isotonic":
            result: np.ndarray = self._calibrator.predict(raw_proba)
        else:
            result = self._calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
        return result

    def get_params(self) -> dict[str, object]:
        """Return base model hyperparameters + calibration method."""
        params: dict[str, object] = self._base.get_params()
        params["calibration_method"] = self._method
        return params

    @property
    def model(self) -> object:
        """Access underlying model (for SHAP compatibility)."""
        return self._base.model
