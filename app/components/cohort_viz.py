"""Risk Cohorts tab component.

Renders PCA cluster scatter and cohort profile cards with
color indicators based on late rate tiers.
"""

from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import cluster_scatter_2d


def _risk_indicator(late_rate: float) -> tuple[str, str]:
    """Return (emoji, color_class) based on late rate threshold."""
    if late_rate >= 0.70:
        return "🔴", "High"
    elif late_rate >= 0.40:
        return "🟡", "Medium"
    else:
        return "🟢", "Low"


def render_cohorts_tab(pipeline: dict[str, Any]) -> None:
    """Render the Risk Cohorts tab."""
    st.header("Risk Cohorts — K-Means Segmentation")
    st.markdown(
        "<div class='disclaimer-box'>"
        "🎯 <strong>Unsupervised clustering</strong> — 4 clusters identified via "
        "K-Means on 10 numeric + shipping mode features. order_region excluded from "
        "clustering (used in post-hoc profiling only)."
        "</div>",
        unsafe_allow_html=True,
    )

    cohorts = pipeline["cohorts"]
    X_pca = pipeline["X_pca"]
    cluster_labels = pipeline["cluster_labels"]
    cohort_labels = pipeline["cohort_labels"]

    scatter_col, cards_col = st.columns([3, 2])

    with scatter_col:
        st.markdown("##### Cluster Visualization (PCA 2D)")
        scatter_fig = cluster_scatter_2d(
            X_pca, cluster_labels, cohort_labels=cohort_labels
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    with cards_col:
        st.markdown("##### Cohort Profiles")
        sorted_cohorts = sorted(cohorts, key=lambda c: c.late_rate, reverse=True)

        for cohort in sorted_cohorts:
            emoji, risk_tier = _risk_indicator(cohort.late_rate)
            header = f"{emoji} {cohort.label} — {cohort.size:,} orders"

            with st.expander(header, expanded=cohort.late_rate >= 0.70):
                c1, c2, c3 = st.columns(3)
                c1.metric("Late Rate", f"{cohort.late_rate:.1%}")
                c2.metric("Avg Days", f"{cohort.avg_scheduled_days:.1f}")
                c3.metric("Risk Tier", risk_tier)

                st.markdown(f"**Shipping Mode:** {cohort.dominant_shipping_mode}")

                if cohort.region_distribution:
                    top_regions = sorted(
                        cohort.region_distribution.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:3]
                    region_str = " · ".join(f"{r} ({v:.0%})" for r, v in top_regions)
                    st.markdown(f"**Top Regions:** {region_str}")

                if cohort.late_rate >= 0.70:
                    st.error("Proactive intervention recommended")
                elif cohort.late_rate >= 0.40:
                    st.warning("Monitor closely")
                else:
                    st.success("Performing within tolerance")
