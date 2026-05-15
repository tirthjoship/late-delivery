"""Customer Segments tab — K-Means risk cohorts with data source toggle."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from adapters.visualization.plotly_charts import cluster_scatter_2d

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_data
def _load_full_pca() -> dict[str, Any] | None:
    """Load pre-computed PCA cluster coordinates for full dataset."""
    path = _PROJECT_ROOT / "data" / "metrics" / "full_pca_clusters.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _risk_indicator(late_rate: float) -> tuple[str, str]:
    """Return (emoji, tier_label) based on late rate."""
    if late_rate >= 0.70:
        return "🔴", "High"
    elif late_rate >= 0.40:
        return "🟡", "Medium"
    else:
        return "🟢", "Low"


def _render_cohort_card(
    label: str,
    size: int,
    late_rate: float,
    avg_days: float,
    dominant_mode: str,
    top_regions: dict[str, float],
    expanded: bool = False,
) -> None:
    """Render a single cohort profile card."""
    emoji, risk_tier = _risk_indicator(late_rate)
    header = f"{emoji} {label} — {size:,} orders"

    with st.expander(header, expanded=expanded):
        c1, c2, c3 = st.columns(3)
        c1.metric("Late Rate", f"{late_rate:.1%}")
        c2.metric("Avg Days", f"{avg_days:.1f}")
        c3.metric("Risk Tier", risk_tier)

        st.markdown(f"**Shipping Mode:** {dominant_mode}")

        if top_regions:
            region_str = " · ".join(f"{r} ({v:.0%})" for r, v in top_regions.items())
            st.markdown(f"**Top Regions:** {region_str}")

        if late_rate >= 0.70:
            st.error("Proactive intervention recommended")
        elif late_rate >= 0.40:
            st.warning("Monitor closely")
        else:
            st.success("Performing within tolerance")


def render_cohorts_tab(
    pipeline: dict[str, Any],
    full_stats: dict[str, Any] | None = None,
    use_full: bool = True,
) -> None:
    """Render the Customer Segments tab."""
    st.header("Customer Segments")

    if use_full and full_stats:
        st.markdown(
            "<div class='disclaimer-box'>"
            "🎯 <strong>Showing: Full Dataset (180K)</strong> — 4 clusters from "
            "65,752 unique orders via K-Means on 10 numeric + shipping mode features"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='disclaimer-box warn'>"
            "⚡ <strong>Showing: Sample (1K)</strong> — clusters from 1,000-order "
            "sample. Toggle to Full Dataset in sidebar for production segmentation."
            "</div>",
            unsafe_allow_html=True,
        )

    if use_full and full_stats:
        cohorts_data = sorted(
            full_stats["cohorts"], key=lambda c: c["late_rate"], reverse=True
        )
        cohort_label_map = {c["cluster_id"]: c["label"] for c in full_stats["cohorts"]}

        # Try to load pre-computed PCA
        full_pca = _load_full_pca()

        if full_pca:
            scatter_col, cards_col = st.columns([3, 2])
            with scatter_col:
                st.markdown("##### Cluster Visualization (PCA 2D)")
                st.caption(
                    f"2,000-point subsample from {full_pca['total_orders']:,} orders"
                )
                X_pca = np.column_stack([full_pca["pc1"], full_pca["pc2"]])
                cluster_labels = np.array(full_pca["cluster"])
                scatter_fig = cluster_scatter_2d(
                    X_pca, cluster_labels, cohort_labels=cohort_label_map
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
            with cards_col:
                st.markdown("##### Cohort Profiles (65,752 orders)")
                for c in cohorts_data:
                    _render_cohort_card(
                        label=c["label"],
                        size=c["size"],
                        late_rate=c["late_rate"],
                        avg_days=c["avg_scheduled_days"],
                        dominant_mode=c["dominant_shipping_mode"],
                        top_regions=c.get("top_regions", {}),
                        expanded=c["late_rate"] >= 0.70,
                    )
        else:
            st.markdown("##### Cohort Profiles (65,752 orders)")
            for c in cohorts_data:
                _render_cohort_card(
                    label=c["label"],
                    size=c["size"],
                    late_rate=c["late_rate"],
                    avg_days=c["avg_scheduled_days"],
                    dominant_mode=c["dominant_shipping_mode"],
                    top_regions=c.get("top_regions", {}),
                    expanded=c["late_rate"] >= 0.70,
                )
    else:
        # Sample — has live PCA scatter
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
                top_regions = dict(
                    sorted(
                        cohort.region_distribution.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:3]
                )
                _render_cohort_card(
                    label=cohort.label,
                    size=cohort.size,
                    late_rate=cohort.late_rate,
                    avg_days=cohort.avg_scheduled_days,
                    dominant_mode=cohort.dominant_shipping_mode,
                    top_regions=top_regions,
                    expanded=cohort.late_rate >= 0.70,
                )
