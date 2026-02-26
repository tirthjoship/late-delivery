"""Unit tests for domain/services.py.

Tests the business logic for late delivery risk prediction.
Follows TDD principles with clear Arrange-Act-Assert structure.
"""

from datetime import datetime

import pytest

from domain.models import Order
from domain.services import (
    HIGH_RISK_SHIPPING_MODES,
    baseline_late_delivery_risk_flag,
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

    def _create_order(
        self, shipping_mode: str, scheduled_days: int
    ) -> Order:
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
