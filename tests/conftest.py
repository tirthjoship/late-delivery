"""Shared test fixtures for supply-chain-optimization-ml.

All fixtures use small synthetic data. Never load the real 180k-row CSV.
"""

from datetime import datetime

import numpy as np
import pytest

from domain.models import Order, OrderItem


def _make_order(
    order_id: int,
    shipping_mode: str,
    scheduled_days: int,
    late_risk: int,
    region: str = "North America",
    month: int = 3,
    sales: float = 200.0,
    benefit: float = 50.0,
    profit: float = 30.0,
) -> Order:
    """Factory for synthetic Order objects."""
    item = OrderItem(
        order_item_id=order_id * 10,
        product_card_id=f"PROD-{order_id:04d}",
        quantity=2,
        unit_price=100.0,
        discount=10.0,
        discount_rate=0.05,
        profit_ratio=0.15,
        sales=190.0,
        item_total=180.0,
    )
    return Order(
        order_id=order_id,
        order_date=datetime(2024, month, 15),
        order_customer_id=order_id + 1000,
        order_region=region,
        order_country="USA",
        order_state="CA",
        order_status="Complete",
        order_city="San Francisco",
        order_zipcode="94102",
        shipping_mode=shipping_mode,
        days_for_shipment_scheduled=scheduled_days,
        benefit_per_order=benefit,
        sales_per_customer=sales,
        order_profit_per_order=profit,
        items=[item],
        late_delivery_risk=late_risk,
    )


@pytest.fixture
def synthetic_orders() -> list[Order]:
    """100 synthetic Orders with realistic distributions.

    ~55% late_delivery_risk=1 (matches real dataset).
    Mix of shipping modes and regions.
    """
    rng = np.random.default_rng(42)
    modes = ["First Class", "Second Class", "Standard Class", "Same Day"]
    mode_weights = [0.15, 0.20, 0.50, 0.15]
    regions = ["North America", "Europe", "Asia", "South America"]

    orders = []
    for i in range(100):
        mode = rng.choice(modes, p=mode_weights)
        region = rng.choice(regions)
        scheduled = int(rng.integers(1, 8))
        month = int(rng.integers(1, 13))

        # Late risk correlated with shipping mode (mirrors real data)
        if mode == "First Class":
            late = 1 if rng.random() < 0.95 else 0
        elif mode == "Second Class":
            late = 1 if rng.random() < 0.77 else 0
        elif mode == "Standard Class":
            late = 1 if rng.random() < 0.38 else 0
        else:  # Same Day
            late = 1 if rng.random() < 0.15 else 0

        orders.append(
            _make_order(
                order_id=i + 1,
                shipping_mode=mode,
                scheduled_days=scheduled,
                late_risk=late,
                region=region,
                month=month,
                sales=float(rng.uniform(10, 5000)),
                benefit=float(rng.uniform(-50, 500)),
                profit=float(rng.uniform(-100, 300)),
            )
        )
    return orders


@pytest.fixture
def synthetic_orders_distinctive_sales() -> tuple[list[Order], list[Order]]:
    """Two sets of orders with distinctive sales_per_customer distributions.

    Train set: sales_per_customer = 1000.0 (all)
    Test set: sales_per_customer = 0.0 (all)

    Used to verify encoder is fit on train only (output-based test).
    """
    train = [_make_order(i, "Standard Class", 3, 1, sales=1000.0) for i in range(50)]
    test = [_make_order(i + 50, "Standard Class", 3, 0, sales=0.0) for i in range(50)]
    return train, test


@pytest.fixture
def clustering_feature_dicts(synthetic_orders: list[Order]) -> list[dict]:
    """Raw feature dicts from synthetic orders for clustering tests."""
    from domain.services import extract_features

    return [extract_features(o) for o in synthetic_orders]


@pytest.fixture
def clustering_numeric_keys() -> list[str]:
    """Numeric feature keys used for clustering (excludes order_region)."""
    return [
        "days_for_shipment_scheduled",
        "order_month",
        "order_day_of_week",
        "benefit_per_order",
        "sales_per_customer",
        "order_profit_per_order",
        "item_count",
        "total_quantity",
        "total_discount",
        "avg_unit_price",
    ]
