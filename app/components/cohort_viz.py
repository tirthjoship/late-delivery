"""Risk Cohorts tab component.

Renders PCA cluster scatter on the left and expandable cohort profile
cards on the right, with color emoji indicators based on late rate.
"""

from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import cluster_scatter_2d


def _risk_emoji(late_rate: float) -> str:
    """Return a color emoji indicator based on late rate threshold."""
    if late_rate >= 0.70:
        return "🔴"
    elif late_rate >= 0.40:
        return "🟡"
    else:
        return "🟢"


def render_cohorts_tab(pipeline: dict[str, Any]) -> None:
    """Render the Risk Cohorts tab.

    Left panel: PCA scatter plot with cluster_scatter_2d.
    Right panel: Expandable profile cards per cohort.

    Args:
        pipeline: Dict of artifacts returned by load_pipeline().
    """
    st.header("Risk Cohorts — K-Means Segmentation")
    st.caption(
        "4 clusters identified via K-Means on 10 numeric + shipping mode features. "
        "order_region excluded from clustering per spec (post-hoc profiling only)."
    )

    cohorts = pipeline["cohorts"]
    X_pca = pipeline["X_pca"]
    cluster_labels = pipeline["cluster_labels"]
    cohort_labels = pipeline["cohort_labels"]

    scatter_col, cards_col = st.columns([3, 2])

    with scatter_col:
        st.subheader("PCA Cluster Scatter")
        scatter_fig = cluster_scatter_2d(
            X_pca, cluster_labels, cohort_labels=cohort_labels
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    with cards_col:
        st.subheader("Cohort Profiles")
        # Sort by late_rate descending so highest risk appears first
        sorted_cohorts = sorted(cohorts, key=lambda c: c.late_rate, reverse=True)

        for cohort in sorted_cohorts:
            emoji = _risk_emoji(cohort.late_rate)
            header = f"{emoji} {cohort.label} (n={cohort.size})"

            with st.expander(header, expanded=cohort.late_rate >= 0.70):
                c1, c2 = st.columns(2)
                c1.metric("Late Rate", f"{cohort.late_rate:.1%}")
                c2.metric("Avg Scheduled Days", f"{cohort.avg_scheduled_days:.1f}")

                st.markdown(
                    f"**Dominant Shipping Mode:** {cohort.dominant_shipping_mode}"
                )

                # Top regions (up to 3)
                if cohort.region_distribution:
                    top_regions = sorted(
                        cohort.region_distribution.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:3]
                    region_str = ", ".join(f"{r} ({v:.0%})" for r, v in top_regions)
                    st.markdown(f"**Top Regions:** {region_str}")

                # Risk interpretation
                if cohort.late_rate >= 0.70:
                    st.error("High-risk cohort — proactive intervention recommended.")
                elif cohort.late_rate >= 0.40:
                    st.warning("Medium-risk cohort — monitor closely.")
                else:
                    st.success("Low-risk cohort — performing well.")
