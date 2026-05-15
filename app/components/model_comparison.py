"""Model Performance tab component.

Strategy C: Shows pre-computed metrics from the full 180k dataset when available,
with sample-trained metrics for interactive SHAP exploration. Both are clearly labeled.
"""

from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import (
    confusion_matrix_heatmap,
    shap_dependence_scatter,
    shap_importance_bar,
)


def _render_full_metrics(full_metrics: dict[str, Any]) -> None:
    """Render metrics from the full 180k dataset (pre-computed JSON)."""
    st.subheader("Classification Metrics")
    st.markdown(
        "<div class='disclaimer-box'>"
        "📊 <strong>Full dataset results</strong> — trained on 144,415 orders, "
        "evaluated on 36,104 orders (80/20 stratified split)"
        "</div>",
        unsafe_allow_html=True,
    )

    lr_data = full_metrics["models"]["LogisticRegression"]
    xgb_data = full_metrics["models"]["XGBoost"]

    lr_col, xgb_col = st.columns(2)

    with lr_col:
        st.markdown("#### Logistic Regression *(baseline)*")
        c1, c2 = st.columns(2)
        c1.metric("F1 Score", f"{lr_data['f1']:.4f}")
        c2.metric("AUC-ROC", f"{lr_data['auc_roc']:.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{lr_data['precision']:.4f}")
        c4.metric("Recall", f"{lr_data['recall']:.4f}")

    with xgb_col:
        st.markdown("#### XGBoost *(production)*")
        c1, c2 = st.columns(2)
        f1_delta = f"{xgb_data['f1'] - lr_data['f1']:+.4f}"
        auc_delta = f"{xgb_data['auc_roc'] - lr_data['auc_roc']:+.4f}"
        c1.metric("F1 Score", f"{xgb_data['f1']:.4f}", delta=f1_delta)
        c2.metric("AUC-ROC", f"{xgb_data['auc_roc']:.4f}", delta=auc_delta)
        prec_delta = f"{xgb_data['precision'] - lr_data['precision']:+.4f}"
        rec_delta = f"{xgb_data['recall'] - lr_data['recall']:+.4f}"
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{xgb_data['precision']:.4f}", delta=prec_delta)
        c4.metric("Recall", f"{xgb_data['recall']:.4f}", delta=rec_delta)

    # Confusion matrices from full run
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.subheader("Confusion Matrices")
    cm_lr_col, cm_xgb_col = st.columns(2)

    with cm_lr_col:
        st.markdown("#### Logistic Regression")
        cm = tuple(tuple(row) for row in lr_data["confusion_matrix"])
        st.plotly_chart(confusion_matrix_heatmap(cm), use_container_width=True)

    with cm_xgb_col:
        st.markdown("#### XGBoost")
        cm = tuple(tuple(row) for row in xgb_data["confusion_matrix"])
        st.plotly_chart(confusion_matrix_heatmap(cm), use_container_width=True)


def _render_sample_metrics(pipeline: dict[str, Any]) -> None:
    """Fallback: render metrics from the sample-trained models."""
    st.subheader("Classification Metrics")
    st.markdown(
        "<div class='disclaimer-box'>"
        "⚡ <strong>Sample data</strong> — trained on 1,000-row sample for demo speed. "
        "See README for full 180k dataset results."
        "</div>",
        unsafe_allow_html=True,
    )

    logreg_result = pipeline["logreg_result"]
    xgb_result = pipeline["xgb_result"]

    lr_col, xgb_col = st.columns(2)

    with lr_col:
        st.markdown("#### Logistic Regression *(baseline)*")
        m = logreg_result.metrics
        c1, c2 = st.columns(2)
        c1.metric("F1 Score", f"{m.f1:.4f}")
        c2.metric("AUC-ROC", f"{m.auc_roc:.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{m.precision:.4f}")
        c4.metric("Recall", f"{m.recall:.4f}")

    with xgb_col:
        st.markdown("#### XGBoost *(production)*")
        m = xgb_result.metrics
        c1, c2 = st.columns(2)
        f1_delta = f"{xgb_result.metrics.f1 - logreg_result.metrics.f1:+.4f}"
        auc_delta = f"{xgb_result.metrics.auc_roc - logreg_result.metrics.auc_roc:+.4f}"
        c1.metric("F1 Score", f"{m.f1:.4f}", delta=f1_delta)
        c2.metric("AUC-ROC", f"{m.auc_roc:.4f}", delta=auc_delta)
        prec_delta = (
            f"{xgb_result.metrics.precision - logreg_result.metrics.precision:+.4f}"
        )
        rec_delta = f"{xgb_result.metrics.recall - logreg_result.metrics.recall:+.4f}"
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{m.precision:.4f}", delta=prec_delta)
        c4.metric("Recall", f"{m.recall:.4f}", delta=rec_delta)

    # Confusion matrices
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.subheader("Confusion Matrices")
    cm_lr_col, cm_xgb_col = st.columns(2)

    with cm_lr_col:
        st.markdown("#### Logistic Regression")
        st.plotly_chart(
            confusion_matrix_heatmap(logreg_result.metrics.confusion_matrix),
            use_container_width=True,
        )

    with cm_xgb_col:
        st.markdown("#### XGBoost")
        st.plotly_chart(
            confusion_matrix_heatmap(xgb_result.metrics.confusion_matrix),
            use_container_width=True,
        )


def render_model_tab(
    pipeline: dict[str, Any], full_metrics: dict[str, Any] | None = None
) -> None:
    """Render the Model Performance tab.

    Uses pre-computed full-dataset metrics when available (Strategy C),
    falls back to sample metrics otherwise.
    """
    st.header("Model Performance Comparison")
    st.caption(
        "Logistic Regression (interpretable baseline) vs XGBoost (production model)"
    )

    # --- Metrics section (full or sample) ---
    if full_metrics:
        _render_full_metrics(full_metrics)
    else:
        _render_sample_metrics(pipeline)

    st.divider()

    # --- Global SHAP importance (XGBoost) — always from sample (interactive) ---
    st.subheader("Global SHAP Feature Importance — XGBoost")
    st.markdown(
        "<div class='disclaimer-box'>"
        "🔍 <strong>Interactive exploration</strong> — SHAP values computed on "
        "1,000-row sample for real-time interaction"
        "</div>",
        unsafe_allow_html=True,
    )

    # Show full-dataset importances if available, else sample
    if full_metrics:
        importance_data = full_metrics["models"]["XGBoost"]["feature_importances"]
    else:
        importance_data = pipeline["xgb_result"].feature_importances

    importance_fig = shap_importance_bar(importance_data)
    st.plotly_chart(importance_fig, use_container_width=True)

    st.divider()

    # --- SHAP dependence scatter (always sample — needs live model) ---
    st.subheader("SHAP Dependence Plot — XGBoost")
    feature_names = pipeline["feature_names"]
    xgb_shap_values = pipeline["xgb_shap_values"]
    X_full = pipeline["X_full_encoded"]

    selected_feature = st.selectbox(
        "Select feature to explore",
        options=feature_names,
        index=0,
        help="See how this feature's values relate to its SHAP contribution.",
    )

    feat_idx = feature_names.index(selected_feature)
    feature_vals = X_full[selected_feature].tolist()
    shap_vals = xgb_shap_values[:, feat_idx].tolist()

    dep_fig = shap_dependence_scatter(selected_feature, feature_vals, shap_vals)
    st.plotly_chart(dep_fig, use_container_width=True)
