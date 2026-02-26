---
inclusion: always
---

# ML Portfolio Architecture & Coding Standards

This steering file enforces unified coding standards across all three ML portfolio projects (Retail Supply Chain, Healthcare Readmission, Stock Recommendation System). These standards align with UBC MDS best practices and industry-standard software engineering principles.

## Core Principles

### 1. Strict Type Hinting (Python 3.12+)

All functions, methods, and class attributes MUST include type hints.

```python
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

def calculate_demand_forecast(
    sales_data: pd.DataFrame,
    forecast_horizon: int,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate demand forecast with confidence intervals.
    
    Args:
        sales_data: Historical sales data with datetime index
        forecast_horizon: Number of periods to forecast
        confidence_level: Confidence level for prediction intervals
        
    Returns:
        Tuple of (point_forecast, confidence_intervals)
    """
    pass
```

### 2. Hexagonal Architecture (Ports and Adapters)

All projects MUST follow hexagonal architecture to separate business logic from infrastructure concerns.

**Directory Structure:**
```
project_name/
├── domain/              # Core business logic (pure Python, no dependencies)
│   ├── models.py       # Domain entities and value objects
│   ├── services.py     # Business logic and use cases
│   └── ports.py        # Abstract interfaces (protocols)
├── adapters/           # Infrastructure implementations
│   ├── data/          # Data source adapters
│   │   ├── csv_repository.py
│   │   ├── api_client.py
│   │   └── database_repository.py
│   ├── ml/            # ML model adapters
│   │   ├── sklearn_predictor.py
│   │   └── pytorch_predictor.py
│   └── visualization/ # Visualization adapters
│       └── plotly_charts.py
├── application/        # Application services (orchestration)
│   └── use_cases.py   # Application-level workflows
└── tests/             # Test suite
    ├── unit/          # Unit tests for domain logic
    ├── integration/   # Integration tests for adapters
    └── properties/    # Property-based tests
```

**Key Rules:**
- Domain layer has NO external dependencies (no pandas, sklearn, etc.)
- Adapters implement port interfaces defined in domain
- Application layer orchestrates domain services and adapters
- All dependencies point inward (adapters depend on domain, not vice versa)

### 3. Test-Driven Development (TDD)

TDD is the PRIMARY workflow. Write tests BEFORE implementation.

**TDD Cycle:**
1. Write a failing test (RED)
2. Write minimal code to pass the test (GREEN)
3. Refactor while keeping tests green (REFACTOR)

**Testing Requirements:**
- Minimum 80% code coverage for domain logic
- Both unit tests AND property-based tests required
- Tests MUST be co-located with source files using `.test.py` suffix
- Use pytest as the testing framework
- Use Hypothesis for property-based testing

**Example Test Structure:**
```python
# domain/services.test.py
import pytest
from hypothesis import given, strategies as st
from .services import DemandForecaster
from .models import SalesRecord

class TestDemandForecaster:
    """Unit tests for demand forecasting service."""
    
    def test_forecast_returns_correct_shape(self):
        """Test that forecast returns expected array shape."""
        # Arrange
        forecaster = DemandForecaster()
        sales_data = [SalesRecord(date="2024-01-01", quantity=100)]
        
        # Act
        forecast = forecaster.forecast(sales_data, horizon=7)
        
        # Assert
        assert len(forecast) == 7
    
    @given(st.lists(st.integers(min_value=0, max_value=1000), min_size=30))
    def test_forecast_never_negative(self, sales_quantities: List[int]):
        """Property: Forecasts should never be negative."""
        # Arrange
        forecaster = DemandForecaster()
        sales_data = [
            SalesRecord(date=f"2024-01-{i:02d}", quantity=q)
            for i, q in enumerate(sales_quantities, 1)
        ]
        
        # Act
        forecast = forecaster.forecast(sales_data, horizon=7)
        
        # Assert
        assert all(f >= 0 for f in forecast)
```

### 4. Documentation Standards (Google-Style Docstrings)

All modules, classes, and functions MUST have Google-style docstrings.

```python
def optimize_inventory(
    demand_forecast: np.ndarray,
    lead_time: int,
    service_level: float,
    holding_cost: float,
    ordering_cost: float
) -> Dict[str, float]:
    """Calculate optimal inventory policy using EOQ model.
    
    This function implements the Economic Order Quantity (EOQ) model
    with safety stock calculations based on demand variability and
    desired service level.
    
    Args:
        demand_forecast: Array of forecasted demand values
        lead_time: Lead time in days for replenishment
        service_level: Target service level (e.g., 0.95 for 95%)
        holding_cost: Cost per unit per period to hold inventory
        ordering_cost: Fixed cost per order placement
        
    Returns:
        Dictionary containing:
            - 'reorder_point': When to place an order
            - 'order_quantity': How much to order
            - 'safety_stock': Buffer inventory level
            - 'expected_cost': Total expected inventory cost
            
    Raises:
        ValueError: If service_level not in range (0, 1)
        ValueError: If costs are negative
        
    Example:
        >>> forecast = np.array([100, 110, 105, 115])
        >>> policy = optimize_inventory(
        ...     demand_forecast=forecast,
        ...     lead_time=7,
        ...     service_level=0.95,
        ...     holding_cost=2.0,
        ...     ordering_cost=50.0
        ... )
        >>> print(policy['reorder_point'])
        150.5
    """
    pass
```

## Code Quality Standards

### Linting and Formatting

All code MUST pass the following tools:

```bash
# Format code with black
black --line-length 88 .

# Sort imports with isort
isort --profile black .

# Type checking with mypy
mypy --strict .

# Linting with ruff
ruff check .
```

**Pre-commit Configuration:**
Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.12
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--strict]
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.15
    hooks:
      - id: ruff
```

### Error Handling

- Use custom exception classes for domain errors
- Never use bare `except:` clauses
- Always log exceptions with context
- Fail fast with clear error messages

```python
# domain/exceptions.py
class DomainError(Exception):
    """Base exception for domain errors."""
    pass

class InvalidForecastHorizonError(DomainError):
    """Raised when forecast horizon is invalid."""
    pass

# Usage
def forecast(self, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise InvalidForecastHorizonError(
            f"Forecast horizon must be positive, got {horizon}"
        )
    # ... implementation
```

## ML-Specific Standards

### Model Development

1. **Baseline First**: Always implement a simple baseline model before complex models
2. **Reproducibility**: Set random seeds, version data, track experiments
3. **Evaluation**: Use appropriate metrics for the problem (not just accuracy)
4. **Interpretability**: Provide model explanations (SHAP, feature importance)

### Experiment Tracking

Use MLflow or similar for experiment tracking:

```python
import mlflow

def train_model(X_train, y_train, params: Dict[str, Any]) -> Model:
    """Train model with experiment tracking."""
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Log metrics
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_r2", train_score)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
    return model
```

### Data Validation

Use Pydantic for data validation:

```python
from pydantic import BaseModel, Field, validator
from datetime import datetime

class SalesRecord(BaseModel):
    """Validated sales record."""
    
    order_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
    order_date: datetime
    
    @validator('order_date')
    def date_not_future(cls, v):
        if v > datetime.now():
            raise ValueError('Order date cannot be in the future')
        return v
```

## Project-Specific Guidelines

### Project 1: Retail Supply Chain Optimization

**Domain Concepts:**
- `Product`: SKU, category, price, lead time
- `Order`: Order ID, products, quantities, dates
- `Forecast`: Demand predictions with confidence intervals
- `InventoryPolicy`: Reorder point, order quantity, safety stock

**Key Ports:**
- `SalesDataRepository`: Load historical sales data
- `DemandForecaster`: Generate demand forecasts
- `InventoryOptimizer`: Calculate optimal inventory policies
- `PerformanceEvaluator`: Evaluate forecast accuracy and inventory metrics

### Project 2: Healthcare Readmission Prediction

**Domain Concepts:**
- `Patient`: Demographics, medical history
- `Encounter`: Hospital visit with diagnoses, procedures, medications
- `RiskScore`: Readmission probability with explanation
- `ClinicalRecommendation`: Actionable insights for care teams

**Key Ports:**
- `PatientDataRepository`: Load patient encounter data
- `RiskPredictor`: Predict readmission risk
- `ModelExplainer`: Generate SHAP explanations
- `FairnessAnalyzer`: Evaluate model fairness across demographics

### Project 3: Stock Recommendation System

**Domain Concepts:**
- `Stock`: Ticker, price history, fundamentals
- `NewsArticle`: Headline, content, sentiment score
- `PriceForecast`: Predicted prices with confidence intervals
- `Recommendation`: Buy/Hold/Sell signal with rationale

**Key Ports:**
- `StockDataRepository`: Load price data (Yahoo Finance)
- `NewsRepository`: Scrape financial news
- `SentimentAnalyzer`: Analyze news sentiment
- `PriceForecaster`: Predict stock prices
- `RecommendationEngine`: Generate buy/sell signals
- `BacktestingEngine`: Simulate trading strategy

## Testing Strategy

### Unit Tests (pytest)

Test domain logic in isolation:

```python
def test_safety_stock_increases_with_variability():
    """Test that higher demand variability requires more safety stock."""
    # Arrange
    optimizer = InventoryOptimizer()
    low_variability_demand = [100, 101, 99, 100, 101]
    high_variability_demand = [50, 150, 75, 125, 100]
    
    # Act
    low_safety_stock = optimizer.calculate_safety_stock(
        low_variability_demand, service_level=0.95
    )
    high_safety_stock = optimizer.calculate_safety_stock(
        high_variability_demand, service_level=0.95
    )
    
    # Assert
    assert high_safety_stock > low_safety_stock
```

### Property-Based Tests (Hypothesis)

Test universal properties:

```python
from hypothesis import given, strategies as st

@given(
    demand=st.lists(st.floats(min_value=0, max_value=1000), min_size=10),
    service_level=st.floats(min_value=0.5, max_value=0.99)
)
def test_higher_service_level_increases_safety_stock(
    demand: List[float],
    service_level: float
):
    """Property: Higher service level should require more safety stock."""
    optimizer = InventoryOptimizer()
    
    lower_service = service_level - 0.05
    higher_service = service_level
    
    safety_stock_lower = optimizer.calculate_safety_stock(demand, lower_service)
    safety_stock_higher = optimizer.calculate_safety_stock(demand, higher_service)
    
    assert safety_stock_higher >= safety_stock_lower
```

### Integration Tests

Test adapter implementations:

```python
def test_csv_repository_loads_sales_data():
    """Test that CSV adapter correctly loads sales data."""
    # Arrange
    repo = CSVSalesRepository(file_path="test_data.csv")
    
    # Act
    sales_records = repo.get_sales_by_product("PROD-001")
    
    # Assert
    assert len(sales_records) > 0
    assert all(isinstance(r, SalesRecord) for r in sales_records)
```

## Performance Guidelines

- Profile code before optimizing (use `cProfile`, `line_profiler`)
- Vectorize operations with NumPy/Pandas instead of loops
- Use appropriate data structures (sets for membership, dicts for lookups)
- Cache expensive computations with `functools.lru_cache`
- Consider memory usage for large datasets (use chunking, generators)

## Git Workflow

- Use feature branches: `feature/demand-forecasting`
- Write descriptive commit messages: "Add EOQ calculation to inventory optimizer"
- Keep commits atomic (one logical change per commit)
- Run tests before committing
- Use conventional commits format:
  - `feat:` New feature
  - `fix:` Bug fix
  - `test:` Add or update tests
  - `refactor:` Code refactoring
  - `docs:` Documentation changes

## Documentation Requirements

Each project MUST include:

1. **README.md**: Project overview, setup instructions, usage examples
2. **ARCHITECTURE.md**: System design, component diagram, data flow
3. **API.md**: Public API documentation (if applicable)
4. **RESULTS.md**: Model performance, evaluation metrics, business impact
5. **Inline docstrings**: All public functions and classes

## Continuous Integration

Set up GitHub Actions for automated testing:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov hypothesis
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
      - name: Type check
        run: mypy --strict .
      - name: Lint
        run: ruff check .
```

## Summary

These standards ensure:
- **Maintainability**: Clean architecture, clear separation of concerns
- **Reliability**: Comprehensive testing, type safety
- **Professionalism**: Industry-standard practices, proper documentation
- **Reproducibility**: Consistent code style, experiment tracking
- **Scalability**: Modular design, performance considerations

All three ML projects MUST adhere to these standards to demonstrate production-ready software engineering skills to potential employers.
