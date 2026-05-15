"""Dataset Overview tab component.

Key stats, shipping mode distribution, region distribution,
and data leakage documentation.
"""

from collections import defaultdict
from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import (
    late_rate_by_region_bar,
    shipping_mode_bar,
)
from domain.models import Order


def _compute_late_rates_by_mode(orders: list[Order]) -> dict[str, float]:
    """Compute late delivery rate per shipping mode."""
    counts: dict[str, int] = defaultdict(int)
    late_counts: dict[str, int] = defaultdict(int)
    for order in orders:
        mode = order.shipping_mode
        counts[mode] += 1
        if order.late_delivery_risk == 1:
            late_counts[mode] += 1
    return {
        mode: late_counts[mode] / counts[mode] for mode in counts if counts[mode] > 0
    }


def _compute_late_rates_by_region(orders: list[Order]) -> dict[str, float]:
    """Compute late delivery rate per order region."""
    counts: dict[str, int] = defaultdict(int)
    late_counts: dict[str, int] = defaultdict(int)
    for order in orders:
        region = order.order_region
        counts[region] += 1
        if order.late_delivery_risk == 1:
            late_counts[region] += 1
    return {
        region: late_counts[region] / counts[region]
        for region in counts
        if counts[region] > 0
    }


def render_dataset_tab(pipeline: dict[str, Any]) -> None:
    """Render the Dataset Overview tab."""
    st.header("Dataset Overview")
    st.markdown(
        "<div class='disclaimer-box'>"
        "📋 <strong>DataCo Supply Chain</strong> — statistics shown from the "
        "1,000-row stratified sample. Full dataset contains 180,519 orders across "
        "multiple product categories, shipping modes, and customer segments."
        "</div>",
        unsafe_allow_html=True,
    )

    orders = pipeline["orders"]

    # --- Key stats ---
    total_orders = len(orders)
    late_orders = sum(1 for o in orders if o.late_delivery_risk == 1)
    overall_late_rate = late_orders / total_orders if total_orders > 0 else 0.0
    shipping_modes = sorted({o.shipping_mode for o in orders})
    regions = sorted({o.order_region for o in orders})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample Orders", f"{total_orders:,}")
    col2.metric("Late Rate", f"{overall_late_rate:.1%}")
    col3.metric("Shipping Modes", str(len(shipping_modes)))
    col4.metric("Regions", str(len(regions)))

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # --- Shipping mode distribution ---
    mode_late_rates = _compute_late_rates_by_mode(orders)
    st.markdown("##### Late Delivery Rate by Shipping Mode")
    mode_fig = shipping_mode_bar(mode_late_rates)
    st.plotly_chart(mode_fig, use_container_width=True)

    st.caption(
        "First Class and Second Class are encoded as HIGH_RISK_SHIPPING_MODES "
        "in the domain layer — they consistently show elevated late rates."
    )

    st.divider()

    # --- Region distribution ---
    region_late_rates = _compute_late_rates_by_region(orders)
    st.markdown("##### Late Delivery Rate by Region")
    region_fig = late_rate_by_region_bar(region_late_rates)
    st.plotly_chart(region_fig, use_container_width=True)

    st.divider()

    # --- Data integrity section ---
    st.markdown("##### Data Integrity & Leakage Protection")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
**3 Columns Excluded (Leakage Shield):**

| Column | Reason |
|--------|--------|
| `Days for shipping (real)` | Only known after delivery |
| `Delivery Status` | Directly encodes the target |
| `shipping date (DateOrders)` | Future information |
            """
        )

    with col_b:
        st.markdown(
            """
**Pipeline Safeguards:**

- ✅ Leakage columns dropped at CSV adapter boundary
- ✅ Train/test split BEFORE encoding
- ✅ Encoder fitted on training data ONLY
- ✅ Property-based tests verify no leakage keywords in features
            """
        )

    # --- Dataset source ---
    with st.expander("Dataset Source & License"):
        st.markdown(
            f"""
**Source:** [DataCo SMART Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Kaggle)

**Full dataset:** 180,519 orders · 53 columns · ~95 MB

**Sample:** {total_orders:,} orders (stratified, PII stripped) · committed to repo

**Class Distribution (sample):**
- Late (1): {late_orders:,} ({late_orders / total_orders:.1%})
- On-Time (0): {total_orders - late_orders:,} ({(total_orders - late_orders) / total_orders:.1%})
            """
        )
