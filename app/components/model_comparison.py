"""Model Results tab — Strategy C with sidebar toggle.

Shows full-dataset or sample metrics based on sidebar selection.
SHAP importance always from full dataset when available.
SHAP dependence plot removed — replaced with key insight callout.
"""

from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import (
    confusion_matrix_heatmap,
    shap_importance_bar,
)


def _source_label(use_full: bool) -> str:
    return "Full Dataset (180K)" if use_full else "Sample (1K)"


def render_model_tab(
    pipeline: dict[str, Any],
    full_metrics: dict[str, Any] | None = None,
    use_full: bool = True,
) -> None:
    """Render the Model Results tab."""
    st.header("Model Results")

    source = _source_label(use_full)
    if use_full and full_metrics:
        st.markdown(
            "<div class='disclaimer-box'>"
            f"📊 <strong>Showing: {source}</strong> — trained on 144,415 orders, "
            "evaluated on 36,104 orders (80/20 stratified split)"
            "</div>",
            unsafe_allow_html=True,
        )
        lr = full_metrics["models"]["LogisticRegression"]
        xgb = full_metrics["models"]["XGBoost"]
        lr_f1, lr_auc = lr["f1"], lr["auc_roc"]
        lr_prec, lr_rec = lr["precision"], lr["recall"]
        xgb_f1, xgb_auc = xgb["f1"], xgb["auc_roc"]
        xgb_prec, xgb_rec = xgb["precision"], xgb["recall"]
        lr_cm = tuple(tuple(r) for r in lr["confusion_matrix"])
        xgb_cm = tuple(tuple(r) for r in xgb["confusion_matrix"])
        importance_data = xgb["feature_importances"]
    else:
        st.markdown(
            "<div class='disclaimer-box warn'>"
            f"⚡ <strong>Showing: {source}</strong> — trained on 1,000-row sample "
            "for demo speed. Toggle to Full Dataset in sidebar for production metrics."
            "</div>",
            unsafe_allow_html=True,
        )
        lr_m = pipeline["logreg_result"].metrics
        xgb_m = pipeline["xgb_result"].metrics
        lr_f1, lr_auc = lr_m.f1, lr_m.auc_roc
        lr_prec, lr_rec = lr_m.precision, lr_m.recall
        xgb_f1, xgb_auc = xgb_m.f1, xgb_m.auc_roc
        xgb_prec, xgb_rec = xgb_m.precision, xgb_m.recall
        lr_cm = lr_m.confusion_matrix
        xgb_cm = xgb_m.confusion_matrix
        importance_data = pipeline["xgb_result"].feature_importances

    # --- Metrics cards ---
    st.subheader("Classification Metrics")
    lr_col, xgb_col = st.columns(2)

    with lr_col:
        st.markdown("#### Logistic Regression *(baseline)*")
        c1, c2 = st.columns(2)
        c1.metric("F1 Score", f"{lr_f1:.4f}")
        c2.metric("AUC-ROC", f"{lr_auc:.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{lr_prec:.4f}")
        c4.metric("Recall", f"{lr_rec:.4f}")

    with xgb_col:
        st.markdown("#### XGBoost *(production)*")
        c1, c2 = st.columns(2)
        c1.metric("F1 Score", f"{xgb_f1:.4f}", delta=f"{xgb_f1 - lr_f1:+.4f}")
        c2.metric("AUC-ROC", f"{xgb_auc:.4f}", delta=f"{xgb_auc - lr_auc:+.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{xgb_prec:.4f}", delta=f"{xgb_prec - lr_prec:+.4f}")
        c4.metric("Recall", f"{xgb_rec:.4f}", delta=f"{xgb_rec - lr_rec:+.4f}")

    st.divider()

    # --- Confusion Matrices ---
    st.subheader("Confusion Matrices")
    cm_lr_col, cm_xgb_col = st.columns(2)
    with cm_lr_col:
        st.markdown("#### Logistic Regression")
        st.plotly_chart(confusion_matrix_heatmap(lr_cm), use_container_width=True)
    with cm_xgb_col:
        st.markdown("#### XGBoost")
        st.plotly_chart(confusion_matrix_heatmap(xgb_cm), use_container_width=True)

    st.divider()

    # --- SHAP importance ---
    st.subheader("Feature Importance — SHAP Values (XGBoost)")
    importance_fig = shap_importance_bar(importance_data)
    st.plotly_chart(importance_fig, use_container_width=True)

    # --- Key insight callout (replaces SHAP dependence plot) ---
    st.info(
        "**Key Insight:** Shipping mode accounts for ~73% of mean |SHAP| on the "
        "temporal test set. First Class and Standard Class dummies dominate; "
        "non-shipping features like `sales_per_customer` and `total_discount` "
        "contribute marginally — the late delivery problem is fundamentally "
        "operational, not transactional."
    )

    # --- Split comparison section ---
    if use_full and full_metrics and "random_split_comparison" in full_metrics:
        st.divider()
        st.subheader("Temporal vs Random Split")
        st.markdown(
            "<div class='disclaimer-box'>"
            "&#x1f4c5; <strong>Temporal split</strong> trains on 2015-2017 orders, "
            "tests on 2018 orders. More realistic than random stratified split."
            "</div>",
            unsafe_allow_html=True,
        )
        random_metrics = full_metrics["random_split_comparison"]
        if "XGBoost" in full_metrics.get("models", {}) and "XGBoost" in random_metrics:
            temporal_f1 = full_metrics["models"]["XGBoost"]["f1"]
            random_f1 = random_metrics["XGBoost"]["f1"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Temporal F1 (realistic)", f"{temporal_f1:.4f}")
            with col2:
                st.metric(
                    "Random F1 (optimistic)",
                    f"{random_f1:.4f}",
                    delta=f"{temporal_f1 - random_f1:.4f}",
                )
