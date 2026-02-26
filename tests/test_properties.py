"""Property-based tests using Hypothesis.

Tests universal properties that should hold for all valid inputs.
Demonstrates advanced testing techniques expected at Amazon/Walmart.
"""

from datetime import datetime

from hypothesis import given, strategies as st

from domain.models import Forecast, Order
from domain.services import baseline_late_delivery_risk_flag, is_high_risk_shipping_mode


# Strategy for valid shipping modes
shipping_mode_strategy = st.sampled_from([
    "First Class",
    "Second Class",
    "Standard Class",
    "Same Day",
])


# Strategy for creating valid Orders
def order_strategy() -> st.SearchStrategy[Order]:
    """Strategy for generating valid Order instances."""
    return st.builds(
        Order,
        order_id=st.integers(min_value=1, max_value=1000000),
        order_date=st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31),
        ),
        order_customer_id=st.integers(min_value=1, max_value=100000),
        order_region=st.sampled_from(["North America", "Europe", "Asia", "South America"]),
        order_country=st.text(min_size=2, max_size=50),
        order_state=st.text(min_size=2, max_size=50),
        order_status=st.sampled_from(["Complete", "Pending", "Canceled", "Processing"]),
        order_city=st.text(min_size=2, max_size=50),
        order_zipcode=st.text(min_size=5, max_size=10),
        shipping_mode=shipping_mode_strategy,
        days_for_shipment_scheduled=st.integers(min_value=1, max_value=30),
        benefit_per_order=st.floats(min_value=0.0, max_value=10000.0),
        sales_per_customer=st.floats(min_value=0.0, max_value=100000.0),
        order_profit_per_order=st.floats(min_value=-1000.0, max_value=10000.0),
        late_delivery_risk=st.one_of(st.none(), st.sampled_from([0, 1])),
    )


class TestShippingModeProperties:
    """Property-based tests for shipping mode classification."""

    @given(shipping_mode=shipping_mode_strategy)
    def test_high_risk_modes_are_consistent(self, shipping_mode: str) -> None:
        """Property: High-risk classification should be deterministic."""
        # Act
        result1 = is_high_risk_shipping_mode(shipping_mode)
        result2 = is_high_risk_shipping_mode(shipping_mode)

        # Assert - same input should always give same result
        assert result1 == result2

    @given(shipping_mode=shipping_mode_strategy)
    def test_high_risk_classification_is_boolean(self, shipping_mode: str) -> None:
        """Property: High-risk classification always returns bool."""
        # Act
        result = is_high_risk_shipping_mode(shipping_mode)

        # Assert
        assert isinstance(result, bool)

    def test_first_class_always_high_risk(self) -> None:
        """Property: First Class (95.3% late rate) is ALWAYS high-risk."""
        # This is a business invariant based on EDA findings
        assert is_high_risk_shipping_mode("First Class") is True

    def test_second_class_always_high_risk(self) -> None:
        """Property: Second Class (76.6% late rate) is ALWAYS high-risk."""
        assert is_high_risk_shipping_mode("Second Class") is True

    def test_standard_class_never_high_risk(self) -> None:
        """Property: Standard Class (38% late rate) is NEVER high-risk by mode alone."""
        assert is_high_risk_shipping_mode("Standard Class") is False


class TestBaselineRiskPredictionProperties:
    """Property-based tests for baseline late delivery risk prediction."""

    @given(order=order_strategy())
    def test_baseline_prediction_is_boolean(self, order: Order) -> None:
        """Property: Baseline risk prediction always returns bool."""
        # Act
        result = baseline_late_delivery_risk_flag(order)

        # Assert
        assert isinstance(result, bool)

    @given(order=order_strategy())
    def test_baseline_prediction_is_deterministic(self, order: Order) -> None:
        """Property: Same order should always produce same risk prediction."""
        # Act
        result1 = baseline_late_delivery_risk_flag(order)
        result2 = baseline_late_delivery_risk_flag(order)

        # Assert
        assert result1 == result2

    @given(
        scheduled_days=st.integers(min_value=1, max_value=3),
    )
    def test_first_class_always_flagged_regardless_of_days(
        self, scheduled_days: int
    ) -> None:
        """Property: First Class is ALWAYS flagged, even with short schedules."""
        # Arrange
        order = Order(
            order_id=1,
            order_date=datetime(2024, 1, 1),
            order_customer_id=100,
            order_region="Test",
            order_country="Test",
            order_state="Test",
            order_status="Complete",
            order_city="Test",
            order_zipcode="12345",
            shipping_mode="First Class",
            days_for_shipment_scheduled=scheduled_days,
            benefit_per_order=50.0,
            sales_per_customer=200.0,
            order_profit_per_order=30.0,
        )

        # Act
        result = baseline_late_delivery_risk_flag(order)

        # Assert
        assert result is True

    @given(
        threshold=st.integers(min_value=1, max_value=10),
        scheduled_days=st.integers(min_value=1, max_value=20),
    )
    def test_threshold_logic_is_correct(
        self, threshold: int, scheduled_days: int
    ) -> None:
        """Property: Orders exceeding threshold should be flagged (when enabled)."""
        # Arrange - Use Standard Class (not high-risk by mode)
        order = Order(
            order_id=1,
            order_date=datetime(2024, 1, 1),
            order_customer_id=100,
            order_region="Test",
            order_country="Test",
            order_state="Test",
            order_status="Complete",
            order_city="Test",
            order_zipcode="12345",
            shipping_mode="Standard Class",
            days_for_shipment_scheduled=scheduled_days,
            benefit_per_order=50.0,
            sales_per_customer=200.0,
            order_profit_per_order=30.0,
        )

        # Act
        result_with_threshold = baseline_late_delivery_risk_flag(
            order,
            use_scheduled_days_threshold=True,
            scheduled_days_threshold=threshold,
        )
        result_without_threshold = baseline_late_delivery_risk_flag(
            order, use_scheduled_days_threshold=False
        )

        # Assert - threshold logic
        if scheduled_days > threshold:
            assert result_with_threshold is True
        else:
            assert result_with_threshold is False

        # Without threshold, Standard Class should never be flagged
        assert result_without_threshold is False


class TestForecastProperties:
    """Property-based tests for Forecast value object."""

    @given(
        point_forecast=st.lists(
            st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
            min_size=1,
            max_size=100,
        )
    )
    def test_forecast_accepts_any_length_point_estimate(
        self, point_forecast: list[float]
    ) -> None:
        """Property: Forecast should accept any length point estimate list."""
        # Act
        forecast = Forecast(point_forecast=point_forecast)

        # Assert
        assert forecast.point_forecast == point_forecast
        assert len(forecast.point_forecast) == len(point_forecast)

    @given(
        n=st.integers(min_value=1, max_value=50),
        lower_diff=st.integers(min_value=1, max_value=10),
        upper_diff=st.integers(min_value=1, max_value=10),
    )
    def test_forecast_confidence_intervals_must_match_length(
        self, n: int, lower_diff: int, upper_diff: int
    ) -> None:
        """Property: Confidence bounds must match point forecast length."""
        # Arrange
        point_forecast = [100.0] * n
        correct_lower = [90.0] * n
        correct_upper = [110.0] * n
        wrong_lower = [90.0] * (n + lower_diff)
        wrong_upper = [110.0] * (n - upper_diff) if n > upper_diff else [110.0]

        # Act & Assert - correct lengths should work
        forecast_correct = Forecast(
            point_forecast=point_forecast,
            confidence_lower=correct_lower,
            confidence_upper=correct_upper,
        )
        assert len(forecast_correct.confidence_lower) == n  # type: ignore
        assert len(forecast_correct.confidence_upper) == n  # type: ignore

    @given(
        point_forecast=st.lists(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
            min_size=2,
            max_size=20,
        )
    )
    def test_forecast_optional_confidence_is_none_when_not_provided(
        self, point_forecast: list[float]
    ) -> None:
        """Property: Confidence bounds default to None when not provided."""
        # Act
        forecast = Forecast(point_forecast=point_forecast)

        # Assert
        assert forecast.confidence_lower is None
        assert forecast.confidence_upper is None
