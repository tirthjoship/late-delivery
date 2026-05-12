"""Scikit-learn and XGBoost model adapters.

Implements ModelTrainerPort for two classifiers:
- LogisticRegressionPredictor: interpretable baseline (odds ratios)
- XGBoostPredictor: production model (non-linear interactions)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class LogisticRegressionPredictor:
    """Interpretable baseline for late delivery risk prediction.

    Implements ModelTrainerPort. Wraps sklearn LogisticRegression.
    Exposes coef_ for odds ratio analysis after training.
    """

    def __init__(self, max_iter: int = 1000, random_state: int = 42) -> None:
        self._model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        result: np.ndarray = self._model.predict(X)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        result: np.ndarray = self._model.predict_proba(X)[:, 1]
        return result

    @property
    def coef_(self) -> np.ndarray:
        result: np.ndarray = self._model.coef_
        return result

    @property
    def model(self) -> LogisticRegression:
        return self._model


class XGBoostPredictor:
    """Production model for late delivery risk prediction.

    Implements ModelTrainerPort. Wraps XGBClassifier.
    Default hyperparams tuned for 55/45 class split.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = 0.82,
        random_state: int = 42,
    ) -> None:
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        result: np.ndarray = self._model.predict(X)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        result: np.ndarray = self._model.predict_proba(X)[:, 1]
        return result

    @property
    def model(self) -> XGBClassifier:
        return self._model
