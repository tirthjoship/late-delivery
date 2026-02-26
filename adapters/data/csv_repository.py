"""CSV data repository adapter.

Loads the DataCo supply chain CSV and returns domain entities.
Excludes leakage columns as identified in EDA (notebooks/eda_initial.ipynb).
"""

from datetime import datetime
from pathlib import Path

from domain.exceptions import DomainError
from domain.models import Order, OrderItem, Product


class CSVValidationError(DomainError):
    """Raised when CSV file is missing required columns or is invalid."""

    pass

# Columns that leak the target (Late_delivery_risk) — must not be used for modeling.
# EDA: Days for shipping (real), Delivery Status, shipping date (DateOrders).
LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "shipping date (DateOrders)",
]

# Required columns for loading orders (non-leakage features only).
REQUIRED_ORDER_COLUMNS = [
    "Order Id",
    "order date (DateOrders)",
    "Order Customer Id",
    "Order Region",
    "Order Country",
    "Order State",
    "Order Status",
    "Order City",
    "Order Zipcode",
    "Shipping Mode",
    "Days for shipment (scheduled)",
]

# Required columns for loading products.
REQUIRED_PRODUCT_COLUMNS = [
    "Product Category Id",
    "Category Name",
    "Product Name",
    "Product Price",
    "Product Status",
]


def _parse_order_date(value: object) -> datetime:
    """Parse order date from CSV value; require datetime for type checker."""
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    s = str(value).strip()
    return datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T"))


def _safe_int(val: object, default: int = 0) -> int:
    """Coerce value to int; return default on failure."""
    if val is None or (isinstance(val, float) and (val != val or val == float("inf"))):
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_float(val: object, default: float = 0.0) -> float:
    """Coerce value to float; return default on failure."""
    if val is None or (isinstance(val, float) and (val != val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_str(val: object, default: str = "") -> str:
    """Coerce value to str; return default if empty."""
    s = str(val).strip() if val is not None else ""
    return s if s else default


class DataCoCSVRepository:
    """Load orders and products from the DataCo supply chain CSV.

    Implements the SalesDataRepository port. Reads the file at construction
    or on first get_orders/get_products call; drops leakage columns before
    building domain entities.
    """

    def __init__(
        self,
        file_path: Path | str,
        *,
        encoding: str = "latin-1",
        drop_leakage: bool = True,
    ) -> None:
        """Initialize repository with path to the DataCo CSV.

        Args:
            file_path: Path to DataCoSupplyChainDataset.csv.
            encoding: File encoding (DataCo uses latin-1).
            drop_leakage: If True, do not load leakage columns (recommended).
        """
        self._path = Path(file_path)
        self._encoding = encoding
        self._drop_leakage = drop_leakage
        self._orders: list[Order] | None = None
        self._products: list[Product] | None = None

    def get_orders(self) -> list[Order]:
        """Return all orders with line items; excludes leakage columns.

        Returns:
            List of domain Order entities. Multiple CSV rows with the same
            Order Id are merged into one Order with multiple OrderItems.

        Raises:
            CSVValidationError: If CSV file does not exist, cannot be read,
                or is missing required columns.
        """
        if self._orders is not None:
            return self._orders
        import pandas as pd

        # Validate file exists
        if not self._path.exists():
            raise CSVValidationError(
                f"CSV file not found: {self._path}"
            )

        # Load CSV with error handling
        try:
            df = pd.read_csv(self._path, encoding=self._encoding)
        except Exception as e:
            raise CSVValidationError(
                f"Failed to read CSV file {self._path}: {e}"
            ) from e

        # Validate required columns exist (before dropping leakage columns)
        missing_cols = [c for c in REQUIRED_ORDER_COLUMNS if c not in df.columns]
        if missing_cols:
            raise CSVValidationError(
                f"CSV missing required order columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        if self._drop_leakage and to_drop:
            df = df.drop(columns=to_drop)

        # Parse order date once
        date_col = "order date (DateOrders)"
        if date_col in df.columns:
            df["_order_date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df["_order_date"] = pd.NaT

        orders_dict: dict[int, list[OrderItem]] = {}
        order_meta: dict[int, dict] = {}

        for _, row in df.iterrows():
            oid = _safe_int(row.get("Order Id"))
            if oid == 0:
                continue
            dt = row.get("_order_date")
            order_dt = _parse_order_date(dt) if pd.notna(dt) else _parse_order_date("2000-01-01")

            if oid not in order_meta:
                order_meta[oid] = {
                    "order_id": oid,
                    "order_date": order_dt,
                    "order_customer_id": _safe_int(row.get("Order Customer Id")),
                    "order_region": _safe_str(row.get("Order Region")),
                    "order_country": _safe_str(row.get("Order Country")),
                    "order_state": _safe_str(row.get("Order State")),
                    "order_status": _safe_str(row.get("Order Status")),
                    "order_city": _safe_str(row.get("Order City")),
                    "order_zipcode": _safe_str(row.get("Order Zipcode")),
                    "shipping_mode": _safe_str(row.get("Shipping Mode")),
                    "days_for_shipment_scheduled": _safe_int(
                        row.get("Days for shipment (scheduled)")
                    ),
                    "benefit_per_order": _safe_float(row.get("Benefit per order")),
                    "sales_per_customer": _safe_float(row.get("Sales per customer")),
                    "order_profit_per_order": _safe_float(
                        row.get("Order Profit Per Order")
                    ),
                    "late_delivery_risk": None,
                }
            if "Late_delivery_risk" in df.columns and order_meta[oid]["late_delivery_risk"] is None:
                val = row.get("Late_delivery_risk")
                if val is not None and str(val).strip() not in ("", "nan"):
                    v = _safe_int(val, default=-1)
                    order_meta[oid]["late_delivery_risk"] = v if v in (0, 1) else None

            item = OrderItem(
                order_item_id=_safe_int(row.get("Order Item Id")),
                product_card_id=_safe_str(row.get("Order Item Cardprod Id"))
                or _safe_str(row.get("Product Card Id")),
                quantity=_safe_int(row.get("Order Item Quantity"), default=1),
                unit_price=_safe_float(row.get("Order Item Product Price")),
                discount=_safe_float(row.get("Order Item Discount")),
                discount_rate=_safe_float(row.get("Order Item Discount Rate")),
                profit_ratio=_safe_float(row.get("Order Item Profit Ratio")),
                sales=_safe_float(row.get("Sales")),
                item_total=_safe_float(row.get("Order Item Total")),
            )
            orders_dict.setdefault(oid, []).append(item)

        self._orders = []
        for oid, items in orders_dict.items():
            meta = order_meta[oid]
            late = meta.get("late_delivery_risk")
            self._orders.append(
                Order(
                    order_id=meta["order_id"],
                    order_date=meta["order_date"],
                    order_customer_id=meta["order_customer_id"],
                    order_region=meta["order_region"],
                    order_country=meta["order_country"],
                    order_state=meta["order_state"],
                    order_status=meta["order_status"],
                    order_city=meta["order_city"],
                    order_zipcode=meta["order_zipcode"],
                    shipping_mode=meta["shipping_mode"],
                    days_for_shipment_scheduled=meta["days_for_shipment_scheduled"],
                    benefit_per_order=meta["benefit_per_order"],
                    sales_per_customer=meta["sales_per_customer"],
                    order_profit_per_order=meta["order_profit_per_order"],
                    items=items,
                    late_delivery_risk=late if late is not None else None,
                )
            )
        return self._orders

    def get_products(self) -> list[Product]:
        """Return distinct products from the CSV.

        Returns:
            List of domain Product entities (one per Product Card Id).

        Raises:
            CSVValidationError: If CSV file does not exist, cannot be read,
                or is missing required columns.
        """
        if self._products is not None:
            return self._products
        import pandas as pd

        # Validate file exists
        if not self._path.exists():
            raise CSVValidationError(
                f"CSV file not found: {self._path}"
            )

        # Load CSV with error handling
        try:
            df = pd.read_csv(self._path, encoding=self._encoding)
        except Exception as e:
            raise CSVValidationError(
                f"Failed to read CSV file {self._path}: {e}"
            ) from e

        # Validate required columns exist
        missing_cols = [c for c in REQUIRED_PRODUCT_COLUMNS if c not in df.columns]
        if missing_cols:
            raise CSVValidationError(
                f"CSV missing required product columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        if self._drop_leakage and to_drop:
            df = df.drop(columns=to_drop)

        # One row per product (use Product Card Id as key; take first row per product)
        key = "Product Card Id"
        if key not in df.columns:
            key = "Order Item Cardprod Id"
        if key not in df.columns:
            self._products = []
            return self._products

        subset = df.drop_duplicates(subset=[key], keep="first")
        self._products = []
        for _, row in subset.iterrows():
            pcid = _safe_str(row.get(key))
            if not pcid:
                continue
            self._products.append(
                Product(
                    product_card_id=pcid,
                    product_category_id=_safe_int(row.get("Product Category Id")),
                    category_name=_safe_str(row.get("Category Name")),
                    product_name=_safe_str(row.get("Product Name")),
                    product_price=_safe_float(row.get("Product Price")),
                    product_status=_safe_str(row.get("Product Status")),
                    lead_time_days=0,
                )
            )
        return self._products
