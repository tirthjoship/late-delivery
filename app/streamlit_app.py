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

import json
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
# Custom CSS for polished look
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
    /* Main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 600;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.8;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Info/disclaimer boxes */
    .disclaimer-box {
        background-color: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #4A9EFF;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.82rem;
        margin-bottom: 1rem;
        opacity: 0.85;
    }

    /* Section spacing */
    .section-gap {
        margin-top: 1.5rem;
    }

    /* Header subtitle */
    .header-subtitle {
        font-size: 1.05rem;
        opacity: 0.7;
        margin-top: -0.8rem;
        margin-bottom: 1.5rem;
    }
</style>
"""


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
# Load pre-computed full-dataset metrics
# ---------------------------------------------------------------------------
@st.cache_data
def load_full_metrics() -> dict[str, Any] | None:
    """Load pre-computed metrics from the full 180k dataset run."""
    metrics_path = _PROJECT_ROOT / "data" / "metrics" / "full_run_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            result: dict[str, Any] = json.load(f)
            return result
    return None


# ---------------------------------------------------------------------------
# Cached pipeline loader — runs once per Streamlit session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models on sample data (1,000 orders)…")
def load_pipeline() -> dict[str, Any]:
    """Train XGBoost + LogReg on sample CSV, run clustering, compute PCA + SHAP.

    Returns a dict with all artifacts needed by the dashboard tabs.
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

    # Inject custom CSS
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # --- Hero header ---
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'>"
        "Supply Chain Late Delivery Risk Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='header-subtitle' style='text-align: center;'>"
        "Predict &bull; Explain &bull; Compare &bull; Segment"
        "</p>",
        unsafe_allow_html=True,
    )

    # Key stats banner
    full_metrics = load_full_metrics()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "180,519 orders")
    col2.metric("Late Rate", "54.83%")
    col3.metric("Best F1 (180k)", "0.6543")
    col4.metric("Models Tracked", "4 variants")

    st.markdown("---")

    pipeline = load_pipeline()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔮 Predict & Explain",
            "📊 Model Performance",
            "🎯 Risk Cohorts",
            "📋 Dataset Overview",
        ]
    )

    with tab1:
        from app.components.prediction import render_prediction_tab

        render_prediction_tab(pipeline)

    with tab2:
        from app.components.model_comparison import render_model_tab

        render_model_tab(pipeline, full_metrics)

    with tab3:
        from app.components.cohort_viz import render_cohorts_tab

        render_cohorts_tab(pipeline)

    with tab4:
        from app.components.dataset_overview import render_dataset_tab

        render_dataset_tab(pipeline)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; opacity: 0.5; font-size: 0.8rem;'>"
        "Built with Streamlit &bull; XGBoost &bull; SHAP &bull; MLflow &bull; "
        "Hexagonal Architecture &bull; "
        "<a href='https://github.com/tirthjoship/supply-chain-optimization-ml' "
        "style='color: inherit;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
