"""Application-level workflows for Retail Supply Chain.

Orchestrates domain services and adapters for model training,
evaluation, and prediction. This is the composition root.

CRITICAL: Split BEFORE encode. Encoder fits on training data ONLY.
This prevents preprocessing leakage (see spec Section 5).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from adapters.ml.evaluation import compute_metrics
from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import Order, PredictionResult, TrainingResult
from domain.ports import ExperimentTrackerPort, ModelTrainerPort
from domain.services import extract_features


class TrainAndEvaluateUseCase:
    """Full training pipeline: load -> extract -> split -> encode -> train -> evaluate -> explain -> track.

    CRITICAL ordering: split raw features FIRST, then fit encoder on train only.
    """

    def __init__(
        self,
        data_repo: Any,
        feature_encoder: FeatureEncoder,
        model: ModelTrainerPort,
        explainer_factory: Callable[..., Any],
        tracker: ExperimentTrackerPort,
    ) -> None:
        self._data_repo = data_repo
        self._encoder = feature_encoder
        self._model = model
        self._explainer_factory = explainer_factory
        self._tracker = tracker

    def execute(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        risk_threshold: float = 0.5,
    ) -> TrainingResult:
        """Run full training pipeline and return results."""
        # 1. Load orders
        orders = self._data_repo.get_orders()

        # 2. Extract features (domain) — label kept separate
        raw_features = [extract_features(o) for o in orders]
        labels = np.array([o.late_delivery_risk for o in orders])

        # 3. SPLIT FIRST — before any encoding
        raw_train, raw_test, y_train, y_test = train_test_split(
            raw_features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # 4. Encode — fit on train ONLY
        X_train = self._encoder.fit_transform(raw_train)
        X_test = self._encoder.transform(raw_test)
        feature_names = self._encoder.get_feature_names()

        # 5. Track experiment
        model_name = type(self._model).__name__
        self._tracker.start_run(run_name=model_name)
        self._tracker.log_params(
            {
                "model": model_name,
                "test_size": str(test_size),
                "random_state": str(random_state),
                "n_features": str(len(feature_names)),
                "n_train": str(len(raw_train)),
                "n_test": str(len(raw_test)),
            }
        )

        # 6. Train
        self._model.train(X_train.values, y_train)

        # 7. Evaluate
        y_pred = self._model.predict(X_test.values)
        y_proba = self._model.predict_proba(X_test.values)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # 8. Log metrics
        self._tracker.log_metrics(
            {
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "auc_roc": metrics.auc_roc,
            }
        )

        # 9. Explain
        explainer = self._explainer_factory(self._model, feature_names)
        global_result = explainer.explain_global(X_test.values)

        # 10. Log artifacts
        self._tracker.log_model(self._model, "model")
        self._tracker.log_artifact(self._encoder, "encoder")
        self._tracker.register_model("late-delivery-risk")
        self._tracker.end_run()

        return TrainingResult(
            model_name=model_name,
            metrics=metrics,
            feature_importances=global_result.feature_importances,
        )


class PredictSingleOrderUseCase:
    """Predict and explain late delivery risk for a single order.

    For Phase 5 Streamlit dashboard integration.
    """

    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        model: ModelTrainerPort,
        explainer: Any,
        risk_threshold: float = 0.5,
    ) -> None:
        self._encoder = feature_encoder
        self._model = model
        self._explainer = explainer
        self._threshold = risk_threshold

    def execute(self, order: Order) -> PredictionResult:
        """Predict and explain risk for a single order."""
        # 1. Extract features (domain)
        raw = extract_features(order)

        # 2. Encode (fitted encoder)
        X = self._encoder.transform([raw])

        # 3. Predict
        probability = float(self._model.predict_proba(X.values)[0])

        # 4. Explain
        local_result = self._explainer.explain_local(X.values, index=0)

        return PredictionResult(
            probability=probability,
            risk_label=probability >= self._threshold,
            explanation=local_result.shap_values,
        )
