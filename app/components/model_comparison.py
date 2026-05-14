"""Model Performance tab component.

Renders side-by-side LogReg vs XGBoost metrics, confusion matrices,
global SHAP importance bar, and SHAP dependence scatter.
"""

from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import (
    confusion_matrix_heatmap,
    shap_dependence_scatter,
    shap_importance_bar,
)


def render_model_tab(pipeline: dict[str, Any]) -> None:
    """Render the Model Performance tab.

    Args:
        pipeline: Dict of artifacts returned by load_pipeline().
    """
    st.header("Model Performance Comparison")
    st.caption("LogisticRegression (interpretable baseline) vs XGBoost (production model)")

    logreg_result = pipeline["logreg_result"]
    xgb_result = pipeline["xgb_result"]

    # --- Metrics cards ---
    st.subheader("Classification Metrics")
    lr_col, xgb_col = st.columns(2)

    with lr_col:
        st.markdown("#### LogisticRegression")
        m = logreg_result.metrics
        c1, c2 = st.columns(2)
        c1.metric("F1 Score", f"{m.f1:.4f}")
        c2.metric("AUC-ROC", f"{m.auc_roc:.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{m.precision:.4f}")
        c4.metric("Recall", f"{m.recall:.4f}")

    with xgb_col:
        st.markdown("#### XGBoost")
        m = xgb_result.metrics
        c1, c2 = st.columns(2)
        f1_delta = f"{xgb_result.metrics.f1 - logreg_result.metrics.f1:+.4f}"
        auc_delta = f"{xgb_result.metrics.auc_roc - logreg_result.metrics.auc_roc:+.4f}"
        c1.metric("F1 Score", f"{m.f1:.4f}", delta=f1_delta)
        c2.metric("AUC-ROC", f"{m.auc_roc:.4f}", delta=auc_delta)
        prec_delta = f"{xgb_result.metrics.precision - logreg_result.metrics.precision:+.4f}"
        rec_delta = f"{xgb_result.metrics.recall - logreg_result.metrics.recall:+.4f}"
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{m.precision:.4f}", delta=prec_delta)
        c4.metric("Recall", f"{m.recall:.4f}", delta=rec_delta)

    st.divider()

    # --- Confusion matrices ---
    st.subheader("Confusion Matrices")
    cm_lr_col, cm_xgb_col = st.columns(2)

    with cm_lr_col:
        st.markdown("#### LogisticRegression")
        cm_fig = confusion_matrix_heatmap(logreg_result.metrics.confusion_matrix)
        st.plotly_chart(cm_fig, use_container_width=True)

    with cm_xgb_col:
        st.markdown("#### XGBoost")
        cm_fig = confusion_matrix_heatmap(xgb_result.metrics.confusion_matrix)
        st.plotly_chart(cm_fig, use_container_width=True)

    st.divider()

    # --- Global SHAP importance (XGBoost) ---
    st.subheader("Global SHAP Feature Importance (XGBoost)")
    importance_fig = shap_importance_bar(xgb_result.feature_importances)
    st.plotly_chart(importance_fig, use_container_width=True)

    st.divider()

    # --- SHAP dependence scatter ---
    st.subheader("SHAP Dependence Plot (XGBoost)")
    feature_names = pipeline["feature_names"]
    xgb_shap_values = pipeline["xgb_shap_values"]
    X_full = pipeline["X_full_encoded"]

    selected_feature = st.selectbox(
        "Select Feature",
        options=feature_names,
        index=0,
        help="Plot how this feature's values relate to its SHAP contribution.",
    )

    feat_idx = feature_names.index(selected_feature)
    feature_vals = X_full[selected_feature].tolist()
    shap_vals = xgb_shap_values[:, feat_idx].tolist()

    dep_fig = shap_dependence_scatter(selected_feature, feature_vals, shap_vals)
    st.plotly_chart(dep_fig, use_container_width=True)
