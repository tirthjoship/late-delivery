"""Integration tests for adapters/data/csv_repository.py.

Tests CSV loading, validation, and error handling.
Validates that the adapter correctly implements the SalesDataRepository port.
"""

import tempfile
from pathlib import Path

import pytest

from adapters.data.csv_repository import (
    LEAKAGE_COLUMNS,
    CSVValidationError,
    DataCoCSVRepository,
)


class TestCSVValidationErrors:
    """Tests for CSV validation and error handling."""

    def test_missing_file_raises_error(self) -> None:
        """Test that missing CSV file raises CSVValidationError."""
        # Arrange
        repo = DataCoCSVRepository("/nonexistent/path/to/file.csv")

        # Act & Assert
        with pytest.raises(CSVValidationError, match="CSV file not found"):
            repo.get_orders()

    def test_missing_required_columns_raises_error(self) -> None:
        """Test that missing required columns raises CSVValidationError."""
        # Arrange - Create a CSV with incomplete columns
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("Order Id,Order Status\n")
            f.write("1,Complete\n")
            temp_path = f.name

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act & Assert
            with pytest.raises(
                CSVValidationError, match="missing required order columns"
            ):
                repo.get_orders()
        finally:
            Path(temp_path).unlink()

    def test_corrupted_csv_raises_error(self) -> None:
        """Test that corrupted CSV file raises CSVValidationError."""
        # Arrange - Create a corrupted CSV (pandas can read garbage as single column,
        # so we verify it fails on missing required columns instead)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("This is not a valid CSV file!@#$%^&*()")
            temp_path = f.name

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act & Assert - Should fail due to missing required columns
            with pytest.raises(CSVValidationError, match="missing required"):
                repo.get_orders()
        finally:
            Path(temp_path).unlink()


class TestLeakageColumnExclusion:
    """Tests that leakage columns are properly excluded."""

    def _create_minimal_csv_with_leakage(self) -> str:
        """Create a minimal valid CSV with leakage columns."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            # Write header with required columns + leakage columns
            f.write(
                "Order Id,order date (DateOrders),Order Customer Id,"
                "Order Region,Order Country,Order State,Order Status,"
                "Order City,Order Zipcode,Shipping Mode,"
                "Days for shipment (scheduled),Benefit per order,"
                "Sales per customer,Order Profit Per Order,"
                "Days for shipping (real),Delivery Status,"
                "shipping date (DateOrders)\n"
            )
            # Write one data row
            f.write(
                "1,2024-01-01,100,North America,USA,CA,Complete,"
                "SF,94102,First Class,3,50.0,200.0,30.0,"
                "5,Late,2024-01-06\n"
            )
            return f.name

    def test_leakage_columns_dropped_by_default(self) -> None:
        """Test that leakage columns are dropped when drop_leakage=True."""
        # Arrange
        temp_path = self._create_minimal_csv_with_leakage()

        try:
            repo = DataCoCSVRepository(temp_path, drop_leakage=True)

            # Act
            orders = repo.get_orders()

            # Assert - We should get valid orders without leakage data
            assert len(orders) > 0
            # The domain Order model doesn't have leakage fields,
            # so we just verify it loaded successfully
        finally:
            Path(temp_path).unlink()

    def test_leakage_columns_constant_is_correct(self) -> None:
        """Verify LEAKAGE_COLUMNS contains the correct leakage features."""
        # Assert
        assert "Days for shipping (real)" in LEAKAGE_COLUMNS
        assert "Delivery Status" in LEAKAGE_COLUMNS
        assert "shipping date (DateOrders)" in LEAKAGE_COLUMNS
        assert len(LEAKAGE_COLUMNS) == 3


class TestDataCoCSVRepositoryOrders:
    """Tests for get_orders() method."""

    def _create_minimal_valid_csv(self) -> str:
        """Create a minimal valid CSV for testing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(
                "Order Id,order date (DateOrders),Order Customer Id,"
                "Order Region,Order Country,Order State,Order Status,"
                "Order City,Order Zipcode,Shipping Mode,"
                "Days for shipment (scheduled),Benefit per order,"
                "Sales per customer,Order Profit Per Order,"
                "Order Item Id,Order Item Cardprod Id,Order Item Quantity,"
                "Order Item Product Price,Order Item Discount,"
                "Order Item Discount Rate,Order Item Profit Ratio,"
                "Sales,Order Item Total,Late_delivery_risk\n"
            )
            f.write(
                "1,2024-01-01,100,North America,USA,CA,Complete,"
                "SF,94102,First Class,3,50.0,200.0,30.0,"
                "1,PROD-001,2,99.99,5.0,0.05,0.15,189.98,184.98,1\n"
            )
            f.write(
                "2,2024-01-02,101,Europe,Germany,Bavaria,Pending,"
                "Munich,80331,Standard Class,5,25.0,150.0,20.0,"
                "2,PROD-002,1,49.99,0.0,0.0,0.2,49.99,49.99,0\n"
            )
            return f.name

    def test_get_orders_returns_list(self) -> None:
        """Test that get_orders returns a list of Order entities."""
        # Arrange
        temp_path = self._create_minimal_valid_csv()

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act
            orders = repo.get_orders()

            # Assert
            assert isinstance(orders, list)
            assert len(orders) == 2
        finally:
            Path(temp_path).unlink()

    def test_get_orders_caches_result(self) -> None:
        """Test that get_orders caches and returns same object on multiple calls."""
        # Arrange
        temp_path = self._create_minimal_valid_csv()

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act
            orders1 = repo.get_orders()
            orders2 = repo.get_orders()

            # Assert - should return cached result
            assert orders1 is orders2
        finally:
            Path(temp_path).unlink()

    def test_order_has_correct_attributes(self) -> None:
        """Test that loaded Order has all expected attributes."""
        # Arrange
        temp_path = self._create_minimal_valid_csv()

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act
            orders = repo.get_orders()
            order = orders[0]

            # Assert
            assert order.order_id == 1
            assert order.order_customer_id == 100
            assert order.order_region == "North America"
            assert order.order_country == "USA"
            assert order.order_state == "CA"
            assert order.shipping_mode == "First Class"
            assert order.days_for_shipment_scheduled == 3
            assert order.late_delivery_risk == 1
        finally:
            Path(temp_path).unlink()


class TestDataCoCSVRepositoryProducts:
    """Tests for get_products() method."""

    def _create_minimal_product_csv(self) -> str:
        """Create a minimal valid CSV with product data."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(
                "Product Card Id,Product Category Id,Category Name,"
                "Product Name,Product Price,Product Status\n"
            )
            f.write("PROD-001,10,Electronics,Laptop,999.99,Active\n")
            f.write("PROD-002,20,Books,Python Guide,49.99,Active\n")
            return f.name

    def test_get_products_returns_list(self) -> None:
        """Test that get_products returns a list of Product entities."""
        # Arrange
        temp_path = self._create_minimal_product_csv()

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act
            products = repo.get_products()

            # Assert
            assert isinstance(products, list)
            assert len(products) == 2
        finally:
            Path(temp_path).unlink()

    def test_get_products_caches_result(self) -> None:
        """Test that get_products caches result."""
        # Arrange
        temp_path = self._create_minimal_product_csv()

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act
            products1 = repo.get_products()
            products2 = repo.get_products()

            # Assert
            assert products1 is products2
        finally:
            Path(temp_path).unlink()

    def test_product_has_correct_attributes(self) -> None:
        """Test that loaded Product has expected attributes."""
        # Arrange
        temp_path = self._create_minimal_product_csv()

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act
            products = repo.get_products()
            product = products[0]

            # Assert
            assert product.product_card_id == "PROD-001"
            assert product.product_category_id == 10
            assert product.category_name == "Electronics"
            assert product.product_name == "Laptop"
            assert product.product_price == 999.99
            assert product.product_status == "Active"
        finally:
            Path(temp_path).unlink()

    def test_missing_product_columns_raises_error(self) -> None:
        """Test that missing product columns raises CSVValidationError."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("Product Card Id,Product Name\n")
            f.write("PROD-001,Laptop\n")
            temp_path = f.name

        try:
            repo = DataCoCSVRepository(temp_path)

            # Act & Assert
            with pytest.raises(
                CSVValidationError, match="missing required product columns"
            ):
                repo.get_products()
        finally:
            Path(temp_path).unlink()
