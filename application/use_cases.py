"""Application-level workflows for Retail Supply Chain.

Orchestrates domain services and adapters for model training,
evaluation, and prediction. This is the composition root.

CRITICAL: Split BEFORE encode. Encoder fits on training data ONLY.
This prevents preprocessing leakage (see spec Section 5).
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from adapters.ml.evaluation import compute_metrics
from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.kmeans_clusterer import KMeansClusterer
from domain.models import Order, PredictionResult, RiskCohort, TrainingResult
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
        run_name: str | None = None,
        split_strategy: str = "random",
    ) -> TrainingResult:
        """Run full training pipeline and return results.

        Args:
            split_strategy: "random" (stratified shuffle) or "temporal" (date-ordered).
        """
        if split_strategy not in ("random", "temporal"):
            msg = (
                f"split_strategy must be 'random' or 'temporal', got '{split_strategy}'"
            )
            raise ValueError(msg)

        # 1. Load orders
        orders = self._data_repo.get_orders()
        logger.info("Loaded {} orders from data repository", len(orders))

        # 2. Extract features (domain) — label kept separate
        raw_features = [extract_features(o) for o in orders]
        labels = np.array([o.late_delivery_risk for o in orders])

        # 3. SPLIT — strategy determines method
        if split_strategy == "temporal":
            # Sort by order_date, take first (1-test_size) as train
            sorted_indices = sorted(
                range(len(orders)), key=lambda i: orders[i].order_date
            )
            cutoff = int(len(sorted_indices) * (1 - test_size))
            train_idx = sorted_indices[:cutoff]
            test_idx = sorted_indices[cutoff:]
            raw_train = [raw_features[i] for i in train_idx]
            raw_test = [raw_features[i] for i in test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]
        else:
            # Original random stratified split
            raw_train, raw_test, y_train, y_test = train_test_split(
                raw_features,
                labels,
                test_size=test_size,
                random_state=random_state,
                stratify=labels,
            )
        logger.info(
            "Split ({}): {} train / {} test",
            split_strategy,
            len(raw_train),
            len(raw_test),
        )

        # 4. Encode — fit on train ONLY
        X_train = self._encoder.fit_transform(raw_train)
        X_test = self._encoder.transform(raw_test)
        feature_names = self._encoder.get_feature_names()

        # 5. Track experiment
        model_name = type(self._model).__name__
        self._tracker.start_run(run_name=run_name or model_name)
        self._tracker.log_params(
            {
                "model": model_name,
                "test_size": str(test_size),
                "random_state": str(random_state),
                "split_strategy": split_strategy,
                "n_features": str(len(feature_names)),
                "n_train": str(len(raw_train)),
                "n_test": str(len(raw_test)),
            }
        )

        # Log model hyperparameters
        model_params = self._model.get_params()
        self._tracker.log_params({f"hp_{k}": str(v) for k, v in model_params.items()})

        # 6. Train
        self._model.train(X_train.values, y_train)
        logger.info("Trained {}", model_name)

        # 7. Evaluate
        y_pred = self._model.predict(X_test.values)
        y_proba = self._model.predict_proba(X_test.values)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        logger.info(
            "Metrics — F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | AUC: {:.4f}",
            metrics.f1,
            metrics.precision,
            metrics.recall,
            metrics.auc_roc,
        )

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
        risk_threshold: float = 0.35,  # Cost-optimized: FN=3x FP
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
        logger.debug(
            "Prediction: probability={:.4f}, risk_label={}",
            probability,
            probability >= self._threshold,
        )

        # 4. Explain
        local_result = self._explainer.explain_local(X.values, index=0)

        return PredictionResult(
            probability=probability,
            risk_label=probability >= self._threshold,
            explanation=local_result.shap_values,
        )


CLUSTERING_NUMERIC_KEYS: list[str] = [
    "days_for_shipment_scheduled",
    "order_month",
    "order_day_of_week",
    "benefit_per_order",
    "sales_per_customer",
    "order_profit_per_order",
    "item_count",
    "total_quantity",
    "total_discount",
    "avg_unit_price",
]


@dataclass
class ClusterSelectionResult:
    """Result of K selection analysis (elbow + silhouette)."""

    optimal_k: int
    elbow_data: dict[int, float]
    silhouette_scores: dict[int, float]


class FitClustersUseCase:
    """Run elbow + silhouette analysis to find optimal K.

    Features: 10 numeric + shipping_mode (one-hot).
    order_region excluded from clustering input.
    """

    def __init__(
        self,
        data_repo: Any,
        feature_encoder: FeatureEncoder,
    ) -> None:
        self._data_repo = data_repo
        self._encoder = feature_encoder

    def execute(self, k_range: range = range(2, 11)) -> ClusterSelectionResult:
        """Run K selection analysis and return optimal K with scores."""
        orders = self._data_repo.get_orders()
        raw_features = [extract_features(o) for o in orders]

        # Encode full features (encoder requires both shipping_mode + order_region)
        X_full = self._encoder.fit_transform(raw_features)
        # Drop order_region columns — excluded from clustering per spec
        region_cols = [c for c in X_full.columns if c.startswith("order_region_")]
        X = X_full.drop(columns=region_cols)
        X_values = X.values

        elbow_data: dict[int, float] = {}
        silhouette_scores: dict[int, float] = {}

        for k in k_range:
            clusterer = KMeansClusterer(n_clusters=k)
            clusterer.fit(X_values)
            elbow_data[k] = clusterer.inertia
            silhouette_scores[k] = clusterer.compute_silhouette(X_values)

        logger.info("K selection — silhouette scores: {}", silhouette_scores)

        optimal_k = max(silhouette_scores, key=silhouette_scores.get)  # type: ignore[arg-type]

        return ClusterSelectionResult(
            optimal_k=optimal_k,
            elbow_data=elbow_data,
            silhouette_scores=silhouette_scores,
        )


class ProfileClustersUseCase:
    """Fit K-Means with chosen K and profile each cluster as a RiskCohort.

    Computes late_rate, dominant_shipping_mode, avg_scheduled_days,
    feature_centroid, and region_distribution per cluster.
    """

    def __init__(
        self,
        data_repo: Any,
        feature_encoder: FeatureEncoder,
        clusterer: KMeansClusterer,
    ) -> None:
        self._data_repo = data_repo
        self._encoder = feature_encoder
        self._clusterer = clusterer

    def execute(self) -> list[RiskCohort]:
        """Profile clusters and return list of RiskCohorts."""
        orders = self._data_repo.get_orders()
        raw_features = [extract_features(o) for o in orders]

        # Encode full features (encoder requires both shipping_mode + order_region)
        X_full = self._encoder.fit_transform(raw_features)
        # Drop order_region columns — excluded from clustering per spec
        region_cols = [c for c in X_full.columns if c.startswith("order_region_")]
        X = X_full.drop(columns=region_cols)
        X_values = X.values

        self._clusterer.fit(X_values)
        labels = self._clusterer.get_labels()
        centroids = self._clusterer.get_centroids()
        feature_names = list(X.columns)  # names after dropping order_region cols

        cohorts: list[RiskCohort] = []
        unique_labels = sorted(set(labels))

        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_orders = [o for o, m in zip(orders, mask) if m]
            _cluster_raw = [
                f for f, m in zip(raw_features, mask) if m
            ]  # for future use

            # Late rate (post-hoc, not used as clustering input)
            late_count = sum(1 for o in cluster_orders if o.late_delivery_risk == 1)
            late_rate = late_count / len(cluster_orders) if cluster_orders else 0.0

            # Dominant shipping mode
            mode_counts = Counter(o.shipping_mode for o in cluster_orders)
            dominant_mode = mode_counts.most_common(1)[0][0]

            # Avg scheduled days
            avg_days = (
                sum(o.days_for_shipment_scheduled for o in cluster_orders)
                / len(cluster_orders)
                if cluster_orders
                else 0.0
            )

            # Centroid as dict
            centroid_dict = {
                name: float(val)
                for name, val in zip(feature_names, centroids[cluster_id])
            }

            # Region distribution (post-hoc profiling)
            region_counts = Counter(o.order_region for o in cluster_orders)
            total = len(cluster_orders)
            region_dist = {
                region: count / total for region, count in region_counts.items()
            }

            # Auto-label by risk tier
            if late_rate >= 0.75:
                label = f"High-Risk {dominant_mode}"
            elif late_rate >= 0.50:
                label = f"Medium-Risk {dominant_mode}"
            else:
                label = f"Low-Risk {dominant_mode}"

            cohorts.append(
                RiskCohort(
                    cluster_id=int(cluster_id),
                    label=label,
                    size=len(cluster_orders),
                    late_rate=round(late_rate, 4),
                    dominant_shipping_mode=dominant_mode,
                    avg_scheduled_days=round(avg_days, 2),
                    feature_centroid=centroid_dict,
                    region_distribution=region_dist,
                )
            )

        logger.info(
            "Profiled {} cohorts: {}",
            len(cohorts),
            [(c.label, c.size, f"{c.late_rate:.1%}") for c in cohorts],
        )
        return cohorts
