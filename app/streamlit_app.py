"""Streamlit app — Supply Chain Late Delivery Risk Dashboard.

4 tabs: Risk Predictor, Model Results, Customer Segments, Data Explorer.
Sidebar toggle switches stats tabs between sample (1K) and full (180K) data.
Predict tab always uses live sample-trained model for interactivity.

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
# Custom CSS
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; font-weight: 600; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.8; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }

    /* Disclaimer boxes */
    .disclaimer-box {
        background-color: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #4A9EFF;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.82rem;
        margin-bottom: 1rem;
        opacity: 0.85;
    }
    .disclaimer-box.warn {
        border-left-color: #FFB84D;
    }

    /* Impact card */
    .impact-card {
        background: linear-gradient(135deg, rgba(74,158,255,0.08), rgba(255,107,107,0.08));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 1.5rem;
    }
    .impact-card h3 { margin-top: 0; margin-bottom: 8px; }

    /* Section spacing */
    .section-gap { margin-top: 1.5rem; }

    /* Sidebar styling */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.03);
    }
</style>
"""


# ---------------------------------------------------------------------------
# No-op tracker
# ---------------------------------------------------------------------------
class _FakeTracker(ExperimentTrackerPort):
    """No-op experiment tracker for dashboard use."""

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
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_full_metrics() -> dict[str, Any] | None:
    """Load pre-computed metrics from the full dataset."""
    path = _PROJECT_ROOT / "data" / "metrics" / "full_run_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_full_stats() -> dict[str, Any] | None:
    """Load pre-computed dataset stats and cohorts from the full dataset."""
    path = _PROJECT_ROOT / "data" / "metrics" / "full_dataset_stats.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource(
    show_spinner="🔄 Training XGBoost + LogReg on sample data — this takes ~5 seconds on first load…"
)
def load_pipeline() -> dict[str, Any]:
    """Train XGBoost + LogReg on sample CSV, run clustering, compute SHAP."""
    sample_csv = _PROJECT_ROOT / "data" / "sample" / "sample.csv"

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

    # --- SHAP ---
    from domain.services import extract_features

    raw_features = [extract_features(o) for o in orders]
    X_full = xgb_encoder.transform(raw_features)
    feature_names = xgb_encoder.get_feature_names()

    xgb_explainer = ShapExplainer(xgb_model.model, feature_names)
    xgb_global = xgb_explainer.explain_global(X_full.values)

    # --- PCA + Clustering ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_full.values)

    cluster_encoder = FeatureEncoder()
    clusterer = KMeansClusterer(n_clusters=4, random_state=42)
    profile_use_case = ProfileClustersUseCase(
        data_repo=data_repo,
        feature_encoder=cluster_encoder,
        clusterer=clusterer,
    )
    cohorts: list[RiskCohort] = profile_use_case.execute()
    cluster_labels = clusterer.get_labels()
    cohort_labels_map: dict[int, str] = {c.cluster_id: c.label for c in cohorts}

    return {
        "orders": orders,
        "xgb_model": xgb_model,
        "logreg_model": logreg_model,
        "encoder": xgb_encoder,
        "xgb_result": xgb_result,
        "logreg_result": logreg_result,
        "xgb_explainer": xgb_explainer,
        "cohorts": cohorts,
        "X_pca": X_pca,
        "cluster_labels": cluster_labels,
        "cohort_labels": cohort_labels_map,
        "feature_names": feature_names,
        "X_full_encoded": X_full,
        "xgb_shap_values": xgb_global.shap_values,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Supply Chain Risk Dashboard",
        page_icon="🚚",
        layout="wide",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        data_source = st.radio(
            "Statistics Source",
            options=["Full Dataset (180K)", "Sample (1K)"],
            index=0,
            help=(
                "Controls which data is shown in Model Results, "
                "Customer Segments, and Data Explorer tabs. "
                "Risk Predictor always uses the sample-trained model "
                "for live interactivity."
            ),
        )
        use_full = data_source == "Full Dataset (180K)"

        st.markdown("---")
        st.markdown("### 📌 Quick Facts")
        st.metric("Total Orders", "180,519 rows")
        st.metric("Unique Orders", "65,752")
        st.metric("Late Rate", "54.83%")
        st.metric("Top Signal", "Shipping Mode")

        st.markdown("---")
        st.markdown(
            "<div style='font-size: 0.75rem; opacity: 0.5;'>"
            "Built by Tirth Joshi<br>"
            "<a href='https://github.com/tirthjoship/supply-chain-optimization-ml' "
            "style='color: inherit;'>View on GitHub</a>"
            "</div>",
            unsafe_allow_html=True,
        )

    # --- Load data ---
    full_metrics = load_full_metrics()
    full_stats = load_full_stats()
    pipeline = load_pipeline()

    # --- Hero ---
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'>"
        "🚚 Supply Chain Late Delivery Risk</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; opacity: 0.6; margin-top: -0.5rem; "
        "margin-bottom: 1.2rem;'>"
        "Predict · Explain · Compare · Segment</p>",
        unsafe_allow_html=True,
    )

    # --- Impact section ---
    st.markdown(
        """<div class="impact-card">
<h3>📍 The Problem</h3>
<p style="margin-bottom: 12px;">
<strong>54.83% of e-commerce orders are delivered late</strong> — more than half.
First Class shipping, despite its premium branding, has a <strong>95.3% late rate</strong>.
This project predicts which orders will arrive late <em>before they ship</em>,
enabling logistics teams to reroute, prioritize, or proactively notify customers.
</p>
<h3>🎯 What I Built</h3>
<p style="margin-bottom: 0;">
An end-to-end ML pipeline with <strong>XGBoost + SHAP explainability</strong>,
<strong>MLflow experiment tracking</strong>, and <strong>hexagonal architecture</strong>
for production-grade code structure. The model achieves <strong>F1 = 0.6543</strong>
with <strong>84.5% precision</strong> — when it flags an order as late, it's right
5 out of 6 times. Shipping mode alone explains 85% of prediction variance.
</p>
</div>""",
        unsafe_allow_html=True,
    )

    # --- Architecture expander ---
    with st.expander("🏗️ How It Works — Architecture Overview", expanded=False):
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown(
                """
**Data Flow:**
1. Raw CSV → Leakage Shield (drops 3 post-shipment columns)
2. Domain entities → Feature extraction (12 pre-shipment features)
3. Train/test split FIRST → then encode (prevents preprocessing leakage)
4. XGBoost + Logistic Regression → evaluate with F1 (not accuracy)
5. SHAP TreeExplainer → global + local feature explanations
6. MLflow → log params, metrics, model artifacts, Model Registry

**Key Safeguard:** The model never sees `Days for shipping (real)`,
`Delivery Status`, or `shipping date` — these are post-shipment data
that would leak the answer.
                """
            )
        with col_b:
            st.markdown(
                """
**Tech Stack:**
| Layer | Tool |
|-------|------|
| ML Models | XGBoost, LogReg |
| Explainability | SHAP |
| Tracking | MLflow |
| Architecture | Hexagonal |
| Testing | 154 tests, 92% cov |
| Type Safety | mypy strict |
| Dashboard | Streamlit |
                """
            )

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔮 Risk Predictor",
            "📊 Model Results",
            "🎯 Customer Segments",
            "📋 Data Explorer",
        ]
    )

    with tab1:
        from app.components.prediction import render_prediction_tab

        render_prediction_tab(pipeline)

    with tab2:
        from app.components.model_comparison import render_model_tab

        render_model_tab(pipeline, full_metrics, use_full)

    with tab3:
        from app.components.cohort_viz import render_cohorts_tab

        render_cohorts_tab(pipeline, full_stats, use_full)

    with tab4:
        from app.components.dataset_overview import render_dataset_tab

        render_dataset_tab(pipeline, full_stats, use_full)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; opacity: 0.4; font-size: 0.78rem;'>"
        "Streamlit · XGBoost · SHAP · MLflow · Hexagonal Architecture · "
        "<a href='https://github.com/tirthjoship/supply-chain-optimization-ml' "
        "style='color: inherit;'>GitHub</a></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
