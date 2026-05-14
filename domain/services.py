"""Business logic and use cases for Retail Supply Chain.

Includes Late Delivery Risk logic refined for Phase 2: high-risk
shipping modes and rule-based baseline (no leakage features).
Pure Python only — no pandas, numpy, or ML libraries.
"""

from .models import Order, RiskCohort

# Shipping modes identified in EDA (DataCo): First Class, Same Day, Second Class, Standard Class.
# Per LOCAL_PROJECT_STORY.md: First Class (95.3% late), Second Class (76.6% late), Standard (38% late).
# First Class and Second Class are HIGH-RISK, not Standard Class.
HIGH_RISK_SHIPPING_MODES: frozenset[str] = frozenset({"First Class", "Second Class"})


def is_high_risk_shipping_mode(shipping_mode: str) -> bool:
    """Return True if the shipping mode is considered high-risk for late delivery.

    Based on EDA: First Class (95.3% late rate) and Second Class (76.6% late rate)
    are treated as higher risk than Standard Class (38.0% late rate) and Same Day.

    Args:
        shipping_mode: The Shipping Mode value (e.g. 'First Class').

    Returns:
        True if the mode is in the high-risk set, False otherwise.

    Example:
        >>> is_high_risk_shipping_mode("First Class")
        True
        >>> is_high_risk_shipping_mode("Same Day")
        False
    """
    return shipping_mode.strip() in HIGH_RISK_SHIPPING_MODES


def order_has_high_risk_shipping_mode(order: Order) -> bool:
    """Return True if the order uses a high-risk shipping mode.

    Args:
        order: Domain Order entity.

    Returns:
        True if order.shipping_mode is high-risk, False otherwise.
    """
    return is_high_risk_shipping_mode(order.shipping_mode)


def baseline_late_delivery_risk_flag(
    order: Order,
    *,
    use_scheduled_days_threshold: bool = True,
    scheduled_days_threshold: int = 4,
) -> bool:
    """Rule-based baseline for late delivery risk (Phase 2, no leakage).

    Uses only pre-shipment features: shipping mode and optionally
    scheduled days for shipment. Does not use actual shipping days,
    delivery status, or shipping date (those are leakage).

    Args:
        order: Domain Order entity.
        use_scheduled_days_threshold: If True, also treat orders with
            scheduled shipment days above threshold as at-risk.
        scheduled_days_threshold: Threshold in days (e.g. 4).
            Orders with days_for_shipment_scheduled > this are
            considered at-risk when use_scheduled_days_threshold is True.

    Returns:
        True if the order is classified as at-risk for late delivery,
        False otherwise.

    Example:
        >>> from domain.models import Order
        >>> from datetime import datetime
        >>> o = Order(order_id=1, order_date=datetime(2024,1,1), ...,
        ...           shipping_mode="Standard Class", days_for_shipment_scheduled=5, ...)
        >>> baseline_late_delivery_risk_flag(o)
        True
    """
    if is_high_risk_shipping_mode(order.shipping_mode):
        return True
    if (
        use_scheduled_days_threshold
        and order.days_for_shipment_scheduled > scheduled_days_threshold
    ):
        return True
    return False


def extract_features(order: Order) -> dict[str, float | str]:
    """Extract ML features from an order using only pre-shipment data.

    Returns a raw feature dict. Encoding (one-hot, scaling) is the
    adapter's job. The label (late_delivery_risk) is NOT included —
    it is extracted separately in the application layer to prevent leakage.
    """
    return {
        "shipping_mode": order.shipping_mode,
        "days_for_shipment_scheduled": order.days_for_shipment_scheduled,
        "order_month": order.order_date.month,
        "order_day_of_week": order.order_date.weekday(),
        "order_region": order.order_region,
        "benefit_per_order": order.benefit_per_order,
        "sales_per_customer": order.sales_per_customer,
        "order_profit_per_order": order.order_profit_per_order,
        "item_count": len(order.items),
        "total_quantity": sum(item.quantity for item in order.items),
        "total_discount": sum(item.discount for item in order.items),
        "avg_unit_price": (
            sum(item.unit_price for item in order.items) / len(order.items)
            if order.items
            else 0.0
        ),
    }


def validate_cohort_separation(cohorts: list[RiskCohort]) -> bool:
    """Check that cohorts have meaningfully different late delivery rates.

    Requires at least 2 cohorts with a >5 percentage point gap between
    the highest and lowest late_rate. A single cohort or cohorts with
    similar late rates indicate K-Means did not find useful segmentation.

    Args:
        cohorts: List of RiskCohort objects from clustering.

    Returns:
        True if cohorts are meaningfully separated, False otherwise.
    """
    if len(cohorts) < 2:
        return False
    rates = [c.late_rate for c in cohorts]
    gap = max(rates) - min(rates)
    # Round to 2 decimals (100ths of a percent) to avoid floating-point
    # precision artifacts, then check if strictly > 5pp
    return round(gap, 2) > 0.05
