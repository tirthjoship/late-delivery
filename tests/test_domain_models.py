"""Unit tests for domain/models.py.

Tests domain entities and value objects.
Validates data integrity constraints and domain logic.
"""

from datetime import datetime

import pytest

from domain.models import Forecast, Order, OrderItem, Product


class TestProduct:
    """Tests for Product entity."""

    def test_product_creation(self) -> None:
        """Test that Product can be created with valid data."""
        # Arrange & Act
        product = Product(
            product_card_id="PROD-001",
            product_category_id=10,
            category_name="Electronics",
            product_name="Laptop",
            product_price=999.99,
            product_status="Active",
            lead_time_days=5,
        )

        # Assert
        assert product.product_card_id == "PROD-001"
        assert product.product_category_id == 10
        assert product.category_name == "Electronics"
        assert product.product_name == "Laptop"
        assert product.product_price == 999.99
        assert product.product_status == "Active"
        assert product.lead_time_days == 5

    def test_product_default_lead_time(self) -> None:
        """Test that Product has default lead_time_days of 0."""
        # Arrange & Act
        product = Product(
            product_card_id="PROD-002",
            product_category_id=20,
            category_name="Books",
            product_name="Python Guide",
            product_price=49.99,
            product_status="Active",
        )

        # Assert
        assert product.lead_time_days == 0

    def test_product_is_frozen(self) -> None:
        """Test that Product is immutable (frozen dataclass)."""
        # Arrange
        product = Product(
            product_card_id="PROD-003",
            product_category_id=30,
            category_name="Apparel",
            product_name="T-Shirt",
            product_price=19.99,
            product_status="Active",
        )

        # Act & Assert
        with pytest.raises(AttributeError):
            product.product_price = 29.99  # type: ignore


class TestOrderItem:
    """Tests for OrderItem entity."""

    def test_order_item_creation(self) -> None:
        """Test that OrderItem can be created with valid data."""
        # Arrange & Act
        item = OrderItem(
            order_item_id=1,
            product_card_id="PROD-001",
            quantity=2,
            unit_price=999.99,
            discount=50.0,
            discount_rate=0.05,
            profit_ratio=0.15,
            sales=1949.98,
            item_total=1899.98,
        )

        # Assert
        assert item.order_item_id == 1
        assert item.product_card_id == "PROD-001"
        assert item.quantity == 2
        assert item.unit_price == 999.99
        assert item.discount == 50.0
        assert item.discount_rate == 0.05
        assert item.profit_ratio == 0.15
        assert item.sales == 1949.98
        assert item.item_total == 1899.98


class TestOrder:
    """Tests for Order entity."""

    def test_order_creation_minimal(self) -> None:
        """Test Order creation with minimal required fields."""
        # Arrange & Act
        order = Order(
            order_id=100,
            order_date=datetime(2024, 1, 15, 10, 30),
            order_customer_id=5000,
            order_region="North America",
            order_country="USA",
            order_state="CA",
            order_status="Complete",
            order_city="San Francisco",
            order_zipcode="94102",
            shipping_mode="First Class",
            days_for_shipment_scheduled=3,
            benefit_per_order=50.0,
            sales_per_customer=200.0,
            order_profit_per_order=30.0,
        )

        # Assert
        assert order.order_id == 100
        assert order.order_date == datetime(2024, 1, 15, 10, 30)
        assert order.order_customer_id == 5000
        assert order.order_region == "North America"
        assert order.shipping_mode == "First Class"
        assert order.items == []  # default empty list
        assert order.late_delivery_risk is None  # default None

    def test_order_with_items(self) -> None:
        """Test Order with multiple OrderItems."""
        # Arrange
        item1 = OrderItem(
            order_item_id=1,
            product_card_id="PROD-001",
            quantity=1,
            unit_price=100.0,
            discount=0.0,
            discount_rate=0.0,
            profit_ratio=0.2,
            sales=100.0,
            item_total=100.0,
        )
        item2 = OrderItem(
            order_item_id=2,
            product_card_id="PROD-002",
            quantity=2,
            unit_price=50.0,
            discount=5.0,
            discount_rate=0.05,
            profit_ratio=0.15,
            sales=100.0,
            item_total=95.0,
        )

        # Act
        order = Order(
            order_id=200,
            order_date=datetime(2024, 2, 1),
            order_customer_id=5001,
            order_region="Europe",
            order_country="Germany",
            order_state="Bavaria",
            order_status="Pending",
            order_city="Munich",
            order_zipcode="80331",
            shipping_mode="Standard Class",
            days_for_shipment_scheduled=5,
            benefit_per_order=25.0,
            sales_per_customer=195.0,
            order_profit_per_order=30.0,
            items=[item1, item2],
        )

        # Assert
        assert len(order.items) == 2
        assert order.items[0].product_card_id == "PROD-001"
        assert order.items[1].product_card_id == "PROD-002"

    def test_order_with_late_delivery_risk(self) -> None:
        """Test Order with late_delivery_risk label."""
        # Arrange & Act
        order = Order(
            order_id=300,
            order_date=datetime(2024, 3, 1),
            order_customer_id=5002,
            order_region="Asia",
            order_country="Japan",
            order_state="Tokyo",
            order_status="Complete",
            order_city="Tokyo",
            order_zipcode="100-0001",
            shipping_mode="Second Class",
            days_for_shipment_scheduled=4,
            benefit_per_order=40.0,
            sales_per_customer=300.0,
            order_profit_per_order=50.0,
            late_delivery_risk=1,
        )

        # Assert
        assert order.late_delivery_risk == 1


class TestForecast:
    """Tests for Forecast value object."""

    def test_forecast_creation_with_point_only(self) -> None:
        """Test Forecast with only point estimates."""
        # Arrange & Act
        forecast = Forecast(point_forecast=[100.0, 110.0, 105.0, 115.0])

        # Assert
        assert forecast.point_forecast == [100.0, 110.0, 105.0, 115.0]
        assert forecast.confidence_lower is None
        assert forecast.confidence_upper is None

    def test_forecast_with_confidence_intervals(self) -> None:
        """Test Forecast with confidence intervals."""
        # Arrange & Act
        forecast = Forecast(
            point_forecast=[100.0, 110.0, 105.0],
            confidence_lower=[90.0, 100.0, 95.0],
            confidence_upper=[110.0, 120.0, 115.0],
        )

        # Assert
        assert forecast.point_forecast == [100.0, 110.0, 105.0]
        assert forecast.confidence_lower == [90.0, 100.0, 95.0]
        assert forecast.confidence_upper == [110.0, 120.0, 115.0]

    def test_forecast_rejects_mismatched_lower_bound(self) -> None:
        """Test Forecast raises error when confidence_lower length mismatches."""
        # Act & Assert
        with pytest.raises(ValueError, match="confidence_lower length must match"):
            Forecast(
                point_forecast=[100.0, 110.0, 105.0],
                confidence_lower=[90.0, 100.0],  # Wrong length
            )

    def test_forecast_rejects_mismatched_upper_bound(self) -> None:
        """Test Forecast raises error when confidence_upper length mismatches."""
        # Act & Assert
        with pytest.raises(ValueError, match="confidence_upper length must match"):
            Forecast(
                point_forecast=[100.0, 110.0],
                confidence_upper=[110.0, 120.0, 115.0],  # Wrong length
            )

    def test_forecast_allows_partial_confidence(self) -> None:
        """Test Forecast allows only lower OR only upper confidence bound."""
        # Arrange & Act
        forecast_lower_only = Forecast(
            point_forecast=[100.0, 110.0],
            confidence_lower=[90.0, 100.0],
        )
        forecast_upper_only = Forecast(
            point_forecast=[100.0, 110.0],
            confidence_upper=[110.0, 120.0],
        )

        # Assert
        assert forecast_lower_only.confidence_lower == [90.0, 100.0]
        assert forecast_lower_only.confidence_upper is None
        assert forecast_upper_only.confidence_lower is None
        assert forecast_upper_only.confidence_upper == [110.0, 120.0]


class TestMetricsResult:
    """Tests for MetricsResult value object."""

    def test_metrics_result_creation(self) -> None:
        from domain.models import MetricsResult

        metrics = MetricsResult(
            f1=0.85,
            precision=0.82,
            recall=0.88,
            auc_roc=0.91,
            confusion_matrix=((100, 20), (15, 85)),
        )
        assert metrics.f1 == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.88
        assert metrics.auc_roc == 0.91
        assert metrics.confusion_matrix == ((100, 20), (15, 85))

    def test_metrics_result_is_frozen(self) -> None:
        from domain.models import MetricsResult

        metrics = MetricsResult(
            f1=0.85,
            precision=0.82,
            recall=0.88,
            auc_roc=0.91,
            confusion_matrix=((100, 20), (15, 85)),
        )
        with pytest.raises(AttributeError):
            metrics.f1 = 0.99  # type: ignore


class TestTrainingResult:
    """Tests for TrainingResult value object."""

    def test_training_result_creation(self) -> None:
        from domain.models import MetricsResult, TrainingResult

        metrics = MetricsResult(
            f1=0.85,
            precision=0.82,
            recall=0.88,
            auc_roc=0.91,
            confusion_matrix=((100, 20), (15, 85)),
        )
        result = TrainingResult(
            model_name="xgboost",
            metrics=metrics,
            feature_importances={
                "shipping_mode_First Class": 0.45,
                "days_for_shipment_scheduled": 0.30,
            },
        )
        assert result.model_name == "xgboost"
        assert result.metrics.f1 == 0.85
        assert result.feature_importances["shipping_mode_First Class"] == 0.45

    def test_training_result_is_frozen(self) -> None:
        from domain.models import MetricsResult, TrainingResult

        metrics = MetricsResult(
            f1=0.85,
            precision=0.82,
            recall=0.88,
            auc_roc=0.91,
            confusion_matrix=((100, 20), (15, 85)),
        )
        result = TrainingResult(
            model_name="xgboost", metrics=metrics, feature_importances={}
        )
        with pytest.raises(AttributeError):
            result.model_name = "logreg"  # type: ignore


class TestPredictionResult:
    """Tests for PredictionResult value object."""

    def test_prediction_result_creation(self) -> None:
        from domain.models import PredictionResult

        result = PredictionResult(
            probability=0.87,
            risk_label=True,
            explanation={
                "shipping_mode_First Class": 0.35,
                "days_for_shipment_scheduled": 0.20,
            },
        )
        assert result.probability == 0.87
        assert result.risk_label is True
        assert result.explanation["shipping_mode_First Class"] == 0.35

    def test_prediction_result_is_frozen(self) -> None:
        from domain.models import PredictionResult

        result = PredictionResult(probability=0.87, risk_label=True, explanation={})
        with pytest.raises(AttributeError):
            result.probability = 0.50  # type: ignore


class TestRiskCohort:
    """Tests for RiskCohort value object."""

    def test_risk_cohort_creation(self) -> None:
        from domain.models import RiskCohort

        cohort = RiskCohort(
            cluster_id=0,
            label="High-Risk Express",
            size=500,
            late_rate=0.92,
            dominant_shipping_mode="First Class",
            avg_scheduled_days=2.5,
            feature_centroid={
                "days_for_shipment_scheduled": 2.5,
                "benefit_per_order": 120.0,
            },
            region_distribution={"North America": 0.45, "Europe": 0.30, "Asia": 0.25},
        )
        assert cohort.cluster_id == 0
        assert cohort.label == "High-Risk Express"
        assert cohort.size == 500
        assert cohort.late_rate == 0.92
        assert cohort.dominant_shipping_mode == "First Class"
        assert cohort.avg_scheduled_days == 2.5
        assert cohort.feature_centroid["benefit_per_order"] == 120.0
        assert cohort.region_distribution["North America"] == 0.45

    def test_risk_cohort_is_frozen(self) -> None:
        from domain.models import RiskCohort

        cohort = RiskCohort(
            cluster_id=0,
            label="Test",
            size=100,
            late_rate=0.5,
            dominant_shipping_mode="Standard Class",
            avg_scheduled_days=3.0,
            feature_centroid={},
            region_distribution={},
        )
        with pytest.raises(AttributeError):
            cohort.late_rate = 0.99  # type: ignore

    def test_risk_cohort_late_rate_range(self) -> None:
        from domain.models import RiskCohort

        cohort = RiskCohort(
            cluster_id=1,
            label="Low-Risk Standard",
            size=200,
            late_rate=0.0,
            dominant_shipping_mode="Same Day",
            avg_scheduled_days=1.0,
            feature_centroid={},
            region_distribution={},
        )
        assert 0.0 <= cohort.late_rate <= 1.0
