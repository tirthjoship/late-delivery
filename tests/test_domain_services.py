"""Unit tests for domain/services.py.

Tests the business logic for late delivery risk prediction.
Follows TDD principles with clear Arrange-Act-Assert structure.
"""

from datetime import datetime

import pytest

from domain.models import Order, OrderItem
from domain.services import (
    HIGH_RISK_SHIPPING_MODES,
    baseline_late_delivery_risk_flag,
    extract_features,
    is_high_risk_shipping_mode,
    order_has_high_risk_shipping_mode,
)


class TestHighRiskShippingMode:
    """Tests for is_high_risk_shipping_mode function."""

    def test_first_class_is_high_risk(self) -> None:
        """Test that First Class is correctly identified as high-risk."""
        # Arrange & Act
        result = is_high_risk_shipping_mode("First Class")

        # Assert
        assert result is True

    def test_second_class_is_high_risk(self) -> None:
        """Test that Second Class is correctly identified as high-risk."""
        # Arrange & Act
        result = is_high_risk_shipping_mode("Second Class")

        # Assert
        assert result is True

    def test_standard_class_is_not_high_risk(self) -> None:
        """Test that Standard Class is NOT high-risk (38% late rate)."""
        # Arrange & Act
        result = is_high_risk_shipping_mode("Standard Class")

        # Assert
        assert result is False

    def test_same_day_is_not_high_risk(self) -> None:
        """Test that Same Day is NOT high-risk."""
        # Arrange & Act
        result = is_high_risk_shipping_mode("Same Day")

        # Assert
        assert result is False

    def test_strips_whitespace(self) -> None:
        """Test that function handles whitespace correctly."""
        # Arrange & Act
        result = is_high_risk_shipping_mode("  First Class  ")

        # Assert
        assert result is True

    def test_unknown_shipping_mode_returns_false(self) -> None:
        """Test that unknown shipping modes return False."""
        # Arrange & Act
        result = is_high_risk_shipping_mode("Overnight Express")

        # Assert
        assert result is False


class TestOrderHasHighRiskShippingMode:
    """Tests for order_has_high_risk_shipping_mode function."""

    def _create_order(self, shipping_mode: str) -> Order:
        """Helper to create a minimal Order for testing."""
        return Order(
            order_id=1,
            order_date=datetime(2024, 1, 1),
            order_customer_id=100,
            order_region="North America",
            order_country="USA",
            order_state="CA",
            order_status="Complete",
            order_city="San Francisco",
            order_zipcode="94102",
            shipping_mode=shipping_mode,
            days_for_shipment_scheduled=3,
            benefit_per_order=50.0,
            sales_per_customer=200.0,
            order_profit_per_order=30.0,
        )

    def test_order_with_first_class_is_high_risk(self) -> None:
        """Test that order with First Class shipping is flagged as high-risk."""
        # Arrange
        order = self._create_order("First Class")

        # Act
        result = order_has_high_risk_shipping_mode(order)

        # Assert
        assert result is True

    def test_order_with_standard_class_is_not_high_risk(self) -> None:
        """Test that order with Standard Class is NOT flagged as high-risk."""
        # Arrange
        order = self._create_order("Standard Class")

        # Act
        result = order_has_high_risk_shipping_mode(order)

        # Assert
        assert result is False


class TestBaselineLateDeliveryRiskFlag:
    """Tests for baseline_late_delivery_risk_flag function."""

    def _create_order(self, shipping_mode: str, scheduled_days: int) -> Order:
        """Helper to create a minimal Order for testing."""
        return Order(
            order_id=1,
            order_date=datetime(2024, 1, 1),
            order_customer_id=100,
            order_region="North America",
            order_country="USA",
            order_state="CA",
            order_status="Complete",
            order_city="San Francisco",
            order_zipcode="94102",
            shipping_mode=shipping_mode,
            days_for_shipment_scheduled=scheduled_days,
            benefit_per_order=50.0,
            sales_per_customer=200.0,
            order_profit_per_order=30.0,
        )

    def test_first_class_flagged_regardless_of_days(self) -> None:
        """Test that First Class is always flagged as high-risk."""
        # Arrange
        order = self._create_order("First Class", scheduled_days=2)

        # Act
        result = baseline_late_delivery_risk_flag(order)

        # Assert
        assert result is True

    def test_second_class_flagged_regardless_of_days(self) -> None:
        """Test that Second Class is always flagged as high-risk."""
        # Arrange
        order = self._create_order("Second Class", scheduled_days=1)

        # Act
        result = baseline_late_delivery_risk_flag(order)

        # Assert
        assert result is True

    def test_standard_class_with_short_schedule_not_flagged(self) -> None:
        """Test Standard Class with 3 days scheduled is NOT flagged."""
        # Arrange
        order = self._create_order("Standard Class", scheduled_days=3)

        # Act
        result = baseline_late_delivery_risk_flag(order)

        # Assert
        assert result is False

    def test_standard_class_with_long_schedule_flagged(self) -> None:
        """Test Standard Class with 5+ days is flagged as high-risk."""
        # Arrange
        order = self._create_order("Standard Class", scheduled_days=5)

        # Act
        result = baseline_late_delivery_risk_flag(
            order, use_scheduled_days_threshold=True, scheduled_days_threshold=4
        )

        # Assert
        assert result is True

    def test_same_day_with_short_schedule_not_flagged(self) -> None:
        """Test Same Day with short schedule is NOT flagged."""
        # Arrange
        order = self._create_order("Same Day", scheduled_days=1)

        # Act
        result = baseline_late_delivery_risk_flag(order)

        # Assert
        assert result is False

    def test_threshold_disabled_ignores_scheduled_days(self) -> None:
        """Test that disabling threshold ignores scheduled days."""
        # Arrange
        order = self._create_order("Standard Class", scheduled_days=10)

        # Act
        result = baseline_late_delivery_risk_flag(
            order, use_scheduled_days_threshold=False
        )

        # Assert
        assert result is False

    def test_custom_threshold_works(self) -> None:
        """Test that custom threshold is applied correctly."""
        # Arrange
        order = self._create_order("Same Day", scheduled_days=6)

        # Act
        result = baseline_late_delivery_risk_flag(
            order, use_scheduled_days_threshold=True, scheduled_days_threshold=5
        )

        # Assert
        assert result is True


class TestHighRiskShippingModesConstant:
    """Tests for the HIGH_RISK_SHIPPING_MODES constant."""

    def test_contains_first_class(self) -> None:
        """Verify First Class is in high-risk modes."""
        assert "First Class" in HIGH_RISK_SHIPPING_MODES

    def test_contains_second_class(self) -> None:
        """Verify Second Class is in high-risk modes."""
        assert "Second Class" in HIGH_RISK_SHIPPING_MODES

    def test_does_not_contain_standard_class(self) -> None:
        """Verify Standard Class is NOT in high-risk modes."""
        assert "Standard Class" not in HIGH_RISK_SHIPPING_MODES

    def test_does_not_contain_same_day(self) -> None:
        """Verify Same Day is NOT in high-risk modes."""
        assert "Same Day" not in HIGH_RISK_SHIPPING_MODES

    def test_is_frozen(self) -> None:
        """Verify HIGH_RISK_SHIPPING_MODES is immutable."""
        # Assert - attempting to add should raise error
        with pytest.raises(AttributeError):
            HIGH_RISK_SHIPPING_MODES.add("Express")  # type: ignore


class TestExtractFeatures:
    """Tests for extract_features function."""

    def _create_order_with_items(
        self,
        shipping_mode: str = "Standard Class",
        scheduled_days: int = 3,
        late_risk: int = 1,
    ) -> Order:
        """Helper: order with 2 items."""
        item1 = OrderItem(
            order_item_id=1,
            product_card_id="PROD-001",
            quantity=2,
            unit_price=100.0,
            discount=10.0,
            discount_rate=0.05,
            profit_ratio=0.15,
            sales=190.0,
            item_total=190.0,
        )
        item2 = OrderItem(
            order_item_id=2,
            product_card_id="PROD-002",
            quantity=3,
            unit_price=50.0,
            discount=5.0,
            discount_rate=0.03,
            profit_ratio=0.10,
            sales=145.0,
            item_total=145.0,
        )
        return Order(
            order_id=1,
            order_date=datetime(2024, 3, 15),
            order_customer_id=100,
            order_region="North America",
            order_country="USA",
            order_state="CA",
            order_status="Complete",
            order_city="San Francisco",
            order_zipcode="94102",
            shipping_mode=shipping_mode,
            days_for_shipment_scheduled=scheduled_days,
            benefit_per_order=50.0,
            sales_per_customer=200.0,
            order_profit_per_order=30.0,
            items=[item1, item2],
            late_delivery_risk=late_risk,
        )

    def test_extract_features_returns_correct_keys(self) -> None:
        order = self._create_order_with_items()
        features = extract_features(order)
        expected_keys = {
            "shipping_mode",
            "days_for_shipment_scheduled",
            "order_month",
            "order_day_of_week",
            "order_region",
            "benefit_per_order",
            "sales_per_customer",
            "order_profit_per_order",
            "item_count",
            "total_quantity",
            "total_discount",
            "avg_unit_price",
        }
        assert set(features.keys()) == expected_keys

    def test_extract_features_no_leakage_columns(self) -> None:
        """CRITICAL: no key should match any leakage column name."""
        from adapters.data.csv_repository import LEAKAGE_COLUMNS

        order = self._create_order_with_items()
        features = extract_features(order)
        for key in features:
            assert key not in LEAKAGE_COLUMNS, f"Leakage column found: {key}"

    def test_extract_features_label_not_included(self) -> None:
        order = self._create_order_with_items(late_risk=1)
        features = extract_features(order)
        assert "late_delivery_risk" not in features

    def test_extract_features_correct_values(self) -> None:
        order = self._create_order_with_items()
        features = extract_features(order)
        assert features["shipping_mode"] == "Standard Class"
        assert features["days_for_shipment_scheduled"] == 3
        assert features["order_month"] == 3  # March
        assert features["order_day_of_week"] == 4  # Friday (2024-03-15)
        assert features["order_region"] == "North America"
        assert features["item_count"] == 2
        assert features["total_quantity"] == 5  # 2 + 3
        assert features["total_discount"] == pytest.approx(15.0)  # 10 + 5
        assert features["avg_unit_price"] == pytest.approx(75.0)  # (100+50)/2

    def test_extract_features_empty_items(self) -> None:
        order = Order(
            order_id=1,
            order_date=datetime(2024, 1, 1),
            order_customer_id=100,
            order_region="Europe",
            order_country="Germany",
            order_state="Bavaria",
            order_status="Complete",
            order_city="Munich",
            order_zipcode="80331",
            shipping_mode="First Class",
            days_for_shipment_scheduled=2,
            benefit_per_order=10.0,
            sales_per_customer=50.0,
            order_profit_per_order=5.0,
        )
        features = extract_features(order)
        assert features["item_count"] == 0
        assert features["total_quantity"] == 0
        assert features["total_discount"] == 0.0
        assert features["avg_unit_price"] == 0.0

    def test_extract_features_returns_correct_types(self) -> None:
        order = self._create_order_with_items()
        features = extract_features(order)
        assert isinstance(features["shipping_mode"], str)
        assert isinstance(features["order_region"], str)
        assert isinstance(features["days_for_shipment_scheduled"], int)
        assert isinstance(features["order_month"], int)
        assert isinstance(features["benefit_per_order"], float)
