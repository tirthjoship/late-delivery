"""Domain entities and value objects for Retail Supply Chain.

Reflects the DataCo dataset structure from EDA (notebooks/eda_initial.ipynb).
Pure Python only — no pandas, numpy, or other external dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Product:
    """A product (SKU) from the DataCo catalog.

    Maps to Product Card Id, Product Category Id, Category Name,
    Product Name, Product Price, Product Status in the dataset.
    Steering: SKU, category, price, lead time.
    """

    product_card_id: str
    """Unique product identifier (SKU)."""
    product_category_id: int
    """Category identifier."""
    category_name: str
    """Human-readable category name."""
    product_name: str
    """Display name of the product."""
    product_price: float
    """Unit price."""
    product_status: str
    """e.g. 'Active', 'Inactive'."""
    lead_time_days: int = 0
    """Optional lead time in days for replenishment (default 0)."""


@dataclass
class OrderItem:
    """A single line item within an order.

    Maps to Order Item Id, Order Item Quantity, Order Item Product Price,
    Order Item Discount, Order Item Profit Ratio, Sales, Order Item Total.
    """

    order_item_id: int
    """Unique line item id within the dataset."""
    product_card_id: str
    """References Product.product_card_id."""
    quantity: int
    """Ordered quantity."""
    unit_price: float
    """Price per unit at time of order."""
    discount: float
    """Discount amount."""
    discount_rate: float
    """Discount rate (e.g. 0.1 for 10%)."""
    profit_ratio: float
    """Profit ratio for this line (e.g. 0.15)."""
    sales: float
    """Total sales (revenue) for this line."""
    item_total: float
    """Total for this line after discount."""


@dataclass
class Order:
    """An order from the DataCo dataset.

    One row in the CSV is one order *item*; multiple rows can share
    the same Order Id. This entity represents the order-level fields
    plus a list of line items.
    Maps to Order Id, order date (DateOrders), Customer Id, Order Region,
    Order Country, Order State, Order Status, Shipping Mode,
    Days for shipment (scheduled), Late_delivery_risk, etc.
    """

    order_id: int
    """Unique order identifier."""
    order_date: datetime
    """Order placement date (from 'order date (DateOrders)')."""
    order_customer_id: int
    """Customer Id."""
    order_region: str
    """Order Region."""
    order_country: str
    """Order Country."""
    order_state: str
    """Order State."""
    order_status: str
    """e.g. 'Complete', 'Pending', 'Canceled' (Order Status)."""
    order_city: str
    """Order City."""
    order_zipcode: str
    """Order Zipcode."""
    shipping_mode: str
    """One of: 'First Class', 'Same Day', 'Second Class', 'Standard Class'."""
    days_for_shipment_scheduled: int
    """Scheduled days for shipment (no leakage)."""
    benefit_per_order: float
    """Benefit per order."""
    sales_per_customer: float
    """Sales per customer."""
    order_profit_per_order: float
    """Order-level profit."""
    items: list[OrderItem] = field(default_factory=list)
    """Line items for this order."""
    late_delivery_risk: Optional[int] = None
    """Binary target (0/1) when labels are present; None if unknown."""


@dataclass(frozen=True)
class MetricsResult:
    """Evaluation metrics for a trained model.

    F1 is the primary metric — accuracy is misleading with 55/45 class split.
    """

    f1: float
    precision: float
    recall: float
    auc_roc: float
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TrainingResult:
    """Output of a model training run."""

    model_name: str
    metrics: MetricsResult
    feature_importances: dict[str, float]


@dataclass(frozen=True)
class PredictionResult:
    """Output of a single-order prediction with explanation."""

    probability: float
    risk_label: bool
    explanation: dict[str, float]


@dataclass(frozen=True)
class RiskCohort:
    """A cluster of orders sharing similar delivery risk characteristics.

    Created by K-Means clustering on order features. Labels assigned
    post-hoc based on late delivery rate and dominant shipping mode.
    """

    cluster_id: int
    label: str
    size: int
    late_rate: float
    dominant_shipping_mode: str
    avg_scheduled_days: float
    feature_centroid: dict[str, float]
    region_distribution: dict[str, float]
