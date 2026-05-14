"""Training script for late delivery risk prediction.

Runs the full ML pipeline: load data, train models, evaluate, explain,
and log to MLflow. Supports sample data for quick demos and multiple
model configurations for experiment comparison.

Usage:
    python scripts/train.py                              # full dataset, all models
    python scripts/train.py --sample                     # sample data, all models
    python scripts/train.py --model logreg               # single model
    python scripts/train.py --sample --model xgboost-default  # quick single test

MLflow UI:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    # open http://localhost:5000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from adapters.data.csv_repository import DataCoCSVRepository  # noqa: E402
from adapters.ml.feature_encoder import FeatureEncoder  # noqa: E402
from adapters.ml.mlflow_tracker import MLflowTracker  # noqa: E402
from adapters.ml.shap_explainer import ShapExplainer  # noqa: E402
from adapters.ml.sklearn_predictor import (  # noqa: E402
    LogisticRegressionPredictor,
    XGBoostPredictor,
)
from application.use_cases import TrainAndEvaluateUseCase  # noqa: E402
from domain.models import TrainingResult  # noqa: E402

# Model configurations: (run_name, model_instance)
MODEL_CONFIGS: dict[str, tuple[str, object]] = {
    "logreg": (
        "logreg-baseline",
        LogisticRegressionPredictor(),
    ),
    "xgboost-default": (
        "xgboost-default",
        XGBoostPredictor(n_estimators=100, max_depth=6, learning_rate=0.1),
    ),
    "xgboost-shallow": (
        "xgboost-shallow",
        XGBoostPredictor(n_estimators=200, max_depth=3, learning_rate=0.1),
    ),
    "xgboost-deep": (
        "xgboost-deep",
        XGBoostPredictor(n_estimators=100, max_depth=8, learning_rate=0.05),
    ),
}

SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "sample.csv"
FULL_PATH = PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train late delivery risk prediction models"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use 1000-row sample data instead of full dataset",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which model configuration to train (default: all)",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name",
        default="late-delivery-risk",
        help="MLflow experiment name (default: late-delivery-risk)",
    )
    return parser.parse_args()


def run_experiment(
    data_path: Path,
    run_name: str,
    model: object,
    tracking_uri: str,
    experiment_name: str,
) -> TrainingResult:
    """Run a single training experiment."""
    repo = DataCoCSVRepository(data_path)
    encoder = FeatureEncoder()
    tracker = MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    use_case = TrainAndEvaluateUseCase(
        data_repo=repo,
        feature_encoder=encoder,
        model=model,
        explainer_factory=lambda m, names: ShapExplainer(m.model, names),
        tracker=tracker,
    )
    return use_case.execute(run_name=run_name)


def log_result(result: TrainingResult) -> None:
    """Log training result summary."""
    m = result.metrics
    logger.info("  F1:        {:.4f}", m.f1)
    logger.info("  Precision: {:.4f}", m.precision)
    logger.info("  Recall:    {:.4f}", m.recall)
    logger.info("  AUC-ROC:   {:.4f}", m.auc_roc)
    logger.info("  Confusion Matrix: {}", m.confusion_matrix)

    top_features = sorted(
        result.feature_importances.items(), key=lambda x: abs(x[1]), reverse=True
    )[:5]
    logger.info("  Top 5 features (SHAP):")
    for name, importance in top_features:
        logger.info("    {}: {:.4f}", name, importance)


def main() -> None:
    args = parse_args()
    logger.info(
        "Training config — model: {} | sample: {} | tracking: {}",
        args.model,
        args.sample,
        args.tracking_uri,
    )

    # Select data path
    if args.sample:
        data_path = SAMPLE_PATH
        if not data_path.exists():
            logger.error("Sample data not found at {}", data_path)
            logger.error("Run: python scripts/generate_sample.py")
            raise SystemExit(1)
        logger.info("Using sample data: {}", data_path)
    else:
        data_path = FULL_PATH
        if not data_path.exists():
            logger.error("Full dataset not found at {}", data_path)
            logger.error("Download from Kaggle or use --sample for demo")
            raise SystemExit(1)
        logger.info("Using full dataset: {}", data_path)

    # Select models
    if args.model == "all":
        configs = list(MODEL_CONFIGS.items())
    else:
        key = args.model
        configs = [(key, MODEL_CONFIGS[key])]

    logger.info("Experiment: {}", args.experiment_name)
    logger.info("Tracking:   {}", args.tracking_uri)
    logger.info("Models:     {}", [c[0] for c in configs])
    logger.info("=" * 60)

    results: list[tuple[str, TrainingResult]] = []

    for config_name, (run_name, model) in configs:
        logger.info("--- {} ---", run_name)
        result = run_experiment(
            data_path=data_path,
            run_name=run_name,
            model=model,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
        log_result(result)
        results.append((run_name, result))
        logger.info("Training complete for {}", run_name)

    # Summary table
    if len(results) > 1:
        logger.info("=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info(
            "{:<25} {:>8} {:>10} {:>8} {:>8}",
            "Run",
            "F1",
            "Precision",
            "Recall",
            "AUC-ROC",
        )
        logger.info("-" * 60)
        for run_name, result in results:
            m = result.metrics
            logger.info(
                "{:<25} {:>8.4f} {:>10.4f} {:>8.4f} {:>8.4f}",
                run_name,
                m.f1,
                m.precision,
                m.recall,
                m.auc_roc,
            )
        logger.info("=" * 60)

    best = max(results, key=lambda x: x[1].metrics.f1)
    logger.success("Best model by F1: {} (F1={:.4f})", best[0], best[1].metrics.f1)
    logger.info("View experiments: mlflow ui --backend-store-uri {}", args.tracking_uri)


if __name__ == "__main__":
    main()
