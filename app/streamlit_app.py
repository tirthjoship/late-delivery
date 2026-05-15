"""Streamlit app entry point for the Supply Chain Late Delivery Risk Dashboard.

Loads the full pipeline once via @st.cache_resource, then renders 4 tabs:
  1. Predict & Explain — single-order risk prediction with SHAP waterfall
  2. Model Performance — LogReg vs XGBoost metrics, SHAP analysis
  3. Risk Cohorts — K-Means cluster scatter + profile cards
  4. Dataset — key stats and distribution charts

Run with:
    streamlit run app/streamlit_app.py
Or:
    make app
"""

import sys
from pathlib import Path
from typing import Any

# Add project root to path so adapters/domain/application are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from adapters.data.csv_repository import DataCoCSVRepository  # noqa: E402
from adapters.ml.feature_encoder import FeatureEncoder  # noqa: E402
from adapters.ml.kmeans_clusterer import KMeansClusterer  # noqa: E402
from adapters.ml.shap_explainer import ShapExplainer  # noqa: E402
from adapters.ml.sklearn_predictor import (  # noqa: E402
    LogisticRegressionPredictor,
    XGBoostPredictor,
)
from application.use_cases import (  # noqa: E402
    ProfileClustersUseCase,
    TrainAndEvaluateUseCase,
)
from domain.models import RiskCohort  # noqa: E402
from domain.ports import ExperimentTrackerPort  # noqa: E402


# ---------------------------------------------------------------------------
# No-op tracker — avoids MLflow file I/O overhead in the dashboard
# ---------------------------------------------------------------------------
class _FakeTracker(ExperimentTrackerPort):
    """No-op experiment tracker for dashboard use (no MLflow overhead)."""

    def start_run(self, run_name: str | None = None) -> None:
        pass

    def log_params(self, params: dict[str, str]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_model(self, model: Any, artifact_name: str) -> None:
        pass

    def log_artifact(self, obj: Any, artifact_name: str) -> None:
        pass

    def register_model(self, name: str) -> None:
        pass

    def end_run(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Cached pipeline loader — runs once per Streamlit session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training pipeline on sample data…")
def load_pipeline() -> dict[str, Any]:
    """Train XGBoost + LogReg on sample CSV, run clustering, compute PCA + SHAP.

    Returns a dict with all artifacts needed by the dashboard tabs:
        orders          — list[Order] (full sample set)
        xgb_model       — trained XGBoostPredictor
        logreg_model    — trained LogisticRegressionPredictor
        encoder         — fitted FeatureEncoder (from XGBoost run)
        xgb_result      — TrainingResult for XGBoost
        logreg_result   — TrainingResult for LogReg
        xgb_explainer   — ShapExplainer for XGBoost
        logreg_explainer— ShapExplainer for LogReg
        cohorts         — list[RiskCohort]
        X_pca           — (n, 2) PCA coordinates of full encoded feature set
        cluster_labels  — (n,) integer cluster assignments
        cohort_labels   — dict[int, str] cluster_id → label
        feature_names   — list[str] post-encoding feature names
        X_test_encoded  — pd.DataFrame encoded test split (for SHAP dependence)
        xgb_shap_values — np.ndarray SHAP values matrix for test split
    """
    sample_csv = Path(__file__).parent.parent / "data" / "sample" / "sample.csv"

    data_repo = DataCoCSVRepository(str(sample_csv))
    orders = data_repo.get_orders()
    tracker = _FakeTracker()

    # --- XGBoost ---
    xgb_encoder = FeatureEncoder()
    xgb_model = XGBoostPredictor()
    xgb_use_case = TrainAndEvaluateUseCase(
        data_repo=data_repo,
        feature_encoder=xgb_encoder,
        model=xgb_model,
        explainer_factory=ShapExplainer,
        tracker=tracker,
    )
    xgb_result = xgb_use_case.execute(run_name="xgb-dashboard")

    # --- LogReg ---
    lr_encoder = FeatureEncoder()
    logreg_model = LogisticRegressionPredictor()
    lr_use_case = TrainAndEvaluateUseCase(
        data_repo=data_repo,
        feature_encoder=lr_encoder,
        model=logreg_model,
        explainer_factory=ShapExplainer,
        tracker=tracker,
    )
    logreg_result = lr_use_case.execute(run_name="lr-dashboard")

    # --- SHAP explainers on full encoded data (for dependence plots) ---
    from domain.services import extract_features

    raw_features = [extract_features(o) for o in orders]
    X_full = xgb_encoder.transform(raw_features)
    feature_names = xgb_encoder.get_feature_names()

    xgb_explainer = ShapExplainer(xgb_model.model, feature_names)
    logreg_explainer = ShapExplainer(logreg_model.model, feature_names)

    xgb_global = xgb_explainer.explain_global(X_full.values)

    # --- PCA (2D) for cluster scatter ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_full.values)

    # --- Clustering ---
    cluster_encoder = FeatureEncoder()
    clusterer = KMeansClusterer(n_clusters=4, random_state=42)
    profile_use_case = ProfileClustersUseCase(
        data_repo=data_repo,
        feature_encoder=cluster_encoder,
        clusterer=clusterer,
    )
    cohorts: list[RiskCohort] = profile_use_case.execute()
    cluster_labels = clusterer.get_labels()

    # Cohort labels for scatter
    cohort_labels_map: dict[int, str] = {c.cluster_id: c.label for c in cohorts}

    return {
        "orders": orders,
        "xgb_model": xgb_model,
        "logreg_model": logreg_model,
        "encoder": xgb_encoder,
        "xgb_result": xgb_result,
        "logreg_result": logreg_result,
        "xgb_explainer": xgb_explainer,
        "logreg_explainer": logreg_explainer,
        "cohorts": cohorts,
        "X_pca": X_pca,
        "cluster_labels": cluster_labels,
        "cohort_labels": cohort_labels_map,
        "feature_names": feature_names,
        "X_full_encoded": X_full,
        "xgb_shap_values": xgb_global.shap_values,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Supply Chain Risk Dashboard",
        page_icon="🚚",
        layout="wide",
    )
    st.title("Supply Chain Late Delivery Risk Dashboard")
    st.caption(
        "DataCo Supply Chain dataset · 54.83% late delivery rate · "
        "XGBoost + LogReg · SHAP explanations · K-Means cohorts"
    )

    pipeline = load_pipeline()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Predict & Explain", "Model Performance", "Risk Cohorts", "Dataset"]
    )

    with tab1:
        from app.components.prediction import render_prediction_tab

        render_prediction_tab(pipeline)

    with tab2:
        from app.components.model_comparison import render_model_tab

        render_model_tab(pipeline)

    with tab3:
        from app.components.cohort_viz import render_cohorts_tab

        render_cohorts_tab(pipeline)

    with tab4:
        from app.components.dataset_overview import render_dataset_tab

        render_dataset_tab(pipeline)


if __name__ == "__main__":
    main()
