"""Dataset Overview tab component.

Renders key stats and distribution charts:
  - Key metrics row: total orders, late rate, shipping modes count, regions count
  - Shipping mode late rate bar chart
  - Region late rate bar chart
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
    """Compute late delivery rate per shipping mode.

    Args:
        orders: List of Order domain objects with late_delivery_risk labels.

    Returns:
        Dict mapping shipping mode → late rate (0.0–1.0).
    """
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
    """Compute late delivery rate per order region.

    Args:
        orders: List of Order domain objects with late_delivery_risk labels.

    Returns:
        Dict mapping region → late rate (0.0–1.0).
    """
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
    """Render the Dataset Overview tab.

    Args:
        pipeline: Dict of artifacts returned by load_pipeline().
    """
    st.header("Dataset Overview")
    st.caption(
        "DataCo Supply Chain sample — summary statistics and delivery risk distributions."
    )

    orders = pipeline["orders"]

    # --- Key stats ---
    total_orders = len(orders)
    late_orders = sum(1 for o in orders if o.late_delivery_risk == 1)
    overall_late_rate = late_orders / total_orders if total_orders > 0 else 0.0
    shipping_modes = sorted({o.shipping_mode for o in orders})
    regions = sorted({o.order_region for o in orders})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Overall Late Rate", f"{overall_late_rate:.1%}")
    col3.metric("Shipping Modes", str(len(shipping_modes)))
    col4.metric("Regions", str(len(regions)))

    st.divider()

    # --- Shipping mode distribution ---
    mode_late_rates = _compute_late_rates_by_mode(orders)
    st.subheader("Late Rate by Shipping Mode")
    mode_fig = shipping_mode_bar(mode_late_rates)
    st.plotly_chart(mode_fig, use_container_width=True)

    st.caption(
        "First Class (95%) and Second Class (77%) are systematically high-risk modes. "
        "These are encoded as `HIGH_RISK_SHIPPING_MODES` in the domain layer."
    )

    st.divider()

    # --- Region distribution ---
    region_late_rates = _compute_late_rates_by_region(orders)
    st.subheader("Late Rate by Region")
    region_fig = late_rate_by_region_bar(region_late_rates)
    st.plotly_chart(region_fig, use_container_width=True)

    # --- Dataset details expander ---
    with st.expander("Dataset Details", expanded=False):
        st.markdown(
            f"""
**Source:** DataCo Supply Chain Dataset (sample of {total_orders:,} orders)

**Shipping Modes:** {", ".join(shipping_modes)}

**Regions:** {", ".join(regions)}

**Class Distribution:**
- Late (1): {late_orders:,} ({late_orders / total_orders:.1%})
- On-Time (0): {total_orders - late_orders:,} ({(total_orders - late_orders) / total_orders:.1%})

**Note on data leakage:** Three columns are excluded from all feature pipelines:
`Days for shipping (real)`, `Delivery Status`, `shipping date (DateOrders)`.
            """
        )
