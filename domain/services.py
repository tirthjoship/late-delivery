"""Business logic and use cases for Retail Supply Chain.

Includes Late Delivery Risk logic refined for Phase 2: high-risk
shipping modes and rule-based baseline (no leakage features).
Pure Python only — no pandas, numpy, or ML libraries.
"""

from .models import Order

# Shipping modes identified in EDA (DataCo): First Class, Same Day, Second Class, Standard Class.
# Per LOCAL_PROJECT_STORY.md: First Class (95.3% late), Second Class (76.6% late), Standard (38% late).
# First Class and Second Class are HIGH-RISK, not Standard Class.
HIGH_RISK_SHIPPING_MODES: frozenset[str] = frozenset(
    {"First Class", "Second Class"}
)


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
    if use_scheduled_days_threshold and order.days_for_shipment_scheduled > scheduled_days_threshold:
        return True
    return False
