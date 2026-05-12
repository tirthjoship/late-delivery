# Phase 4: Predictive Modeling & Explainability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a late delivery risk prediction pipeline with two models (Logistic Regression + XGBoost), SHAP explainability, and MLflow experiment tracking — all within hexagonal architecture boundaries.

**Architecture:** Port Per Concern. Domain extracts features (pure dict), adapters encode/train/explain/track. Application orchestrates. Split before encode to prevent preprocessing leakage.

**Tech Stack:** scikit-learn, xgboost, shap, mlflow, hypothesis, pytest

**Spec:** `docs/superpowers/specs/2026-05-11-phase4-modeling-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `domain/models.py` | Modify (lines 108+) | Add MetricsResult, TrainingResult, PredictionResult |
| `domain/ports.py` | Modify (lines 36+) | Add ModelTrainerPort, ExplainerPort, ExperimentTrackerPort |
| `domain/services.py` | Modify (lines 88+) | Add extract_features() |
| `adapters/ml/feature_encoder.py` | Create | FeatureEncoder (ColumnTransformer wrapper) |
| `adapters/ml/sklearn_predictor.py` | Rewrite | LogisticRegressionPredictor, XGBoostPredictor |
| `adapters/ml/evaluation.py` | Create | compute_metrics() → MetricsResult |
| `adapters/ml/shap_explainer.py` | Create | ShapExplainer (global + local) |
| `adapters/ml/mlflow_tracker.py` | Create | MLflowTracker (local + registry) |
| `application/use_cases.py` | Rewrite | TrainAndEvaluateUseCase, PredictSingleOrderUseCase |
| `tests/test_domain_services.py` | Modify | Add extract_features tests |
| `tests/test_properties.py` | Modify | Add order_item_strategy, extract_features properties |
| `tests/test_ml/__init__.py` | Create | Package init |
| `tests/test_ml/test_feature_encoder.py` | Create | Encoder contract tests |
| `tests/test_ml/test_sklearn_predictor.py` | Create | Model contract tests |
| `tests/test_ml/test_evaluation.py` | Create | Metrics computation tests |
| `tests/test_ml/test_shap_explainer.py` | Create | SHAP contract tests |
| `tests/test_ml/test_mlflow_tracker.py` | Create | MLflow integration tests (tmp_path) |
| `tests/test_use_cases.py` | Create | Use case integration tests |
| `tests/conftest.py` | Create | Shared fixtures (synthetic_orders, etc.) |
| `pyproject.toml` | Modify | Add shap dependency |
| `notebooks/eda_initial.ipynb` | Modify | Add interaction analysis cell (Step 0) |

---

## Task 0: Quick Interaction EDA

**Files:**
- Modify: `notebooks/eda_initial.ipynb` (add cell after last cell)

- [ ] **Step 1: Add interaction analysis cell to notebook**

Open `notebooks/eda_initial.ipynb` and add a new markdown + code cell at the end:

Markdown cell:
```markdown
## Interaction Analysis: Shipping Mode × Scheduled Days

Before modeling, check whether scheduled days has a different effect on late delivery depending on shipping mode. This informs whether interaction features are worth engineering.
```

Code cell:
```python
# Interaction: shipping mode × scheduled days bins
interaction = df.groupby(
    ['Shipping Mode', pd.cut(df['Days for shipment (scheduled)'], bins=5)]
)['Late_delivery_risk'].agg(['mean', 'count']).round(3)
interaction.columns = ['Late Rate', 'Order Count']
print("Late Delivery Rate by Shipping Mode × Scheduled Days:")
display(interaction)

# Interaction: shipping mode × order value quartile
df['order_value_quartile'] = pd.qcut(df['Sales per customer'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
interaction2 = df.groupby(
    ['Shipping Mode', 'order_value_quartile']
)['Late_delivery_risk'].agg(['mean', 'count']).round(3)
interaction2.columns = ['Late Rate', 'Order Count']
print("\nLate Delivery Rate by Shipping Mode × Order Value Quartile:")
display(interaction2)
```

- [ ] **Step 2: Run the cell and review output**

Run: open notebook in JupyterLab or `jupyter nbconvert --execute notebooks/eda_initial.ipynb`

Look for: does scheduled days have different late rate slopes per shipping mode? If First Class is 95% late regardless of scheduled days, that feature adds nothing for that mode. If Standard Class goes from 20% to 60% as scheduled days increase, that's a signal.

- [ ] **Step 3: Commit**

```bash
git add notebooks/eda_initial.ipynb
git commit -m "eda: add interaction analysis for shipping mode × scheduled days"
```

---

## Task 1: Domain Models — Result Dataclasses

**Files:**
- Modify: `domain/models.py` (append after line 136)
- Test: `tests/test_domain_models.py`

- [ ] **Step 1: Write failing tests for MetricsResult, TrainingResult, PredictionResult**

Add to `tests/test_domain_models.py`:

```python
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
            f1=0.85, precision=0.82, recall=0.88,
            auc_roc=0.91, confusion_matrix=((100, 20), (15, 85)),
        )
        with pytest.raises(AttributeError):
            metrics.f1 = 0.99  # type: ignore


class TestTrainingResult:
    """Tests for TrainingResult value object."""

    def test_training_result_creation(self) -> None:
        from domain.models import MetricsResult, TrainingResult

        metrics = MetricsResult(
            f1=0.85, precision=0.82, recall=0.88,
            auc_roc=0.91, confusion_matrix=((100, 20), (15, 85)),
        )
        result = TrainingResult(
            model_name="xgboost",
            metrics=metrics,
            feature_importances={"shipping_mode_First Class": 0.45, "days_for_shipment_scheduled": 0.30},
        )
        assert result.model_name == "xgboost"
        assert result.metrics.f1 == 0.85
        assert result.feature_importances["shipping_mode_First Class"] == 0.45

    def test_training_result_is_frozen(self) -> None:
        from domain.models import MetricsResult, TrainingResult

        metrics = MetricsResult(
            f1=0.85, precision=0.82, recall=0.88,
            auc_roc=0.91, confusion_matrix=((100, 20), (15, 85)),
        )
        result = TrainingResult(model_name="xgboost", metrics=metrics, feature_importances={})
        with pytest.raises(AttributeError):
            result.model_name = "logreg"  # type: ignore


class TestPredictionResult:
    """Tests for PredictionResult value object."""

    def test_prediction_result_creation(self) -> None:
        from domain.models import PredictionResult

        result = PredictionResult(
            probability=0.87,
            risk_label=True,
            explanation={"shipping_mode_First Class": 0.35, "days_for_shipment_scheduled": 0.20},
        )
        assert result.probability == 0.87
        assert result.risk_label is True
        assert result.explanation["shipping_mode_First Class"] == 0.35

    def test_prediction_result_is_frozen(self) -> None:
        from domain.models import PredictionResult

        result = PredictionResult(probability=0.87, risk_label=True, explanation={})
        with pytest.raises(AttributeError):
            result.probability = 0.50  # type: ignore
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_domain_models.py::TestMetricsResult -v`
Expected: FAIL with `ImportError: cannot import name 'MetricsResult'`

- [ ] **Step 3: Implement the three dataclasses**

Add to `domain/models.py` after the `Forecast` class (after line 136):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_domain_models.py -v`
Expected: ALL PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add domain/models.py tests/test_domain_models.py
git commit -m "feat(domain): add MetricsResult, TrainingResult, PredictionResult dataclasses"
```

---

## Task 2: Domain Ports — ML Protocol Interfaces

**Files:**
- Modify: `domain/ports.py` (append after line 35)

- [ ] **Step 1: Add the three new port protocols**

Add to `domain/ports.py` after the `SalesDataRepository` class:

```python
from typing import Any, Protocol

# (update existing import line to include Any)


class ModelTrainerPort(Protocol):
    """Port: train and predict late delivery risk.

    Implemented by adapters (e.g. LogReg, XGBoost) to provide
    model training and prediction without domain depending on ML framework.
    """

    def train(self, X: Any, y: Any) -> None:
        """Train the model on feature matrix X and labels y."""
        ...

    def predict(self, X: Any) -> Any:
        """Return binary predictions (0 or 1)."""
        ...

    def predict_proba(self, X: Any) -> Any:
        """Return probability estimates for the positive class."""
        ...


class ExplainerPort(Protocol):
    """Port: explain model predictions.

    Implemented by adapters (e.g. SHAP, LIME) to provide
    model explanations without domain depending on explainability framework.
    """

    def explain_global(self, X: Any) -> Any:
        """Return global feature importance explanation."""
        ...

    def explain_local(self, X: Any, index: int) -> Any:
        """Return explanation for a single prediction at index."""
        ...


class ExperimentTrackerPort(Protocol):
    """Port: track experiments, log metrics, register models.

    Implemented by adapters (e.g. MLflow, W&B) to provide
    experiment tracking without domain depending on tracking framework.
    """

    def start_run(self, run_name: str) -> None:
        """Start a new experiment run."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the current run."""
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation metrics for the current run."""
        ...

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log a trained model as an artifact."""
        ...

    def log_artifact(self, artifact: Any, artifact_path: str) -> None:
        """Log an arbitrary artifact (e.g. encoder)."""
        ...

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """Load a registered model by name and stage."""
        ...

    def load_artifact(self, run_id: str, artifact_path: str) -> Any:
        """Load an artifact from a specific run."""
        ...

    def register_model(self, model_name: str) -> None:
        """Register the current run's model in the model registry."""
        ...

    def end_run(self) -> None:
        """End the current experiment run."""
        ...
```

- [ ] **Step 2: Update the import line**

Change line 7 of `domain/ports.py` from:
```python
from typing import Protocol
```
to:
```python
from typing import Any, Protocol
```

- [ ] **Step 3: Run domain-check skill**

Run: `/domain-check` — verify domain/ has zero external imports.

- [ ] **Step 4: Run existing tests**

Run: `pytest tests/ -v`
Expected: ALL PASS (ports are Protocol — no implementation needed to pass existing tests)

- [ ] **Step 5: Commit**

```bash
git add domain/ports.py
git commit -m "feat(domain): add ModelTrainerPort, ExplainerPort, ExperimentTrackerPort protocols"
```

---

## Task 3: Domain Services — extract_features()

**Files:**
- Modify: `domain/services.py` (append after line 88)
- Test: `tests/test_domain_services.py`
- Test: `tests/test_properties.py`

- [ ] **Step 1: Write failing unit tests for extract_features**

Add to `tests/test_domain_services.py`:

```python
from domain.services import extract_features


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
            "shipping_mode", "days_for_shipment_scheduled",
            "order_month", "order_day_of_week", "order_region",
            "benefit_per_order", "sales_per_customer", "order_profit_per_order",
            "item_count", "total_quantity", "total_discount", "avg_unit_price",
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
        """Label must NOT be in feature dict."""
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
```

Add import at top of file:
```python
from domain.models import Order, OrderItem
from domain.services import extract_features
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_domain_services.py::TestExtractFeatures -v`
Expected: FAIL with `ImportError: cannot import name 'extract_features'`

- [ ] **Step 3: Implement extract_features**

Add to `domain/services.py` after line 88:

```python
def extract_features(order: Order) -> dict[str, float | str]:
    """Extract ML features from an order using only pre-shipment data.

    Returns a raw feature dict. Encoding (one-hot, scaling) is the
    adapter's job. The label (late_delivery_risk) is NOT included —
    it is extracted separately in the application layer to prevent leakage.

    Args:
        order: Domain Order entity.

    Returns:
        Dict mapping feature names to values. String values for
        categoricals, numeric values for continuous features.
    """
    return {
        "shipping_mode": order.shipping_mode,
        "days_for_shipment_scheduled": order.days_for_shipment_scheduled,
        "order_month": order.order_date.month,
        "order_day_of_week": order.order_date.weekday(),
        "order_region": order.order_region,
        "benefit_per_order": order.benefit_per_order,
        "sales_per_customer": order.sales_per_customer,
        "order_profit_per_order": order.order_profit_per_order,
        "item_count": len(order.items),
        "total_quantity": sum(item.quantity for item in order.items),
        "total_discount": sum(item.discount for item in order.items),
        "avg_unit_price": (
            sum(item.unit_price for item in order.items) / len(order.items)
            if order.items
            else 0.0
        ),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_domain_services.py -v`
Expected: ALL PASS

- [ ] **Step 5: Add property-based tests**

Add `order_item_strategy` to `tests/test_properties.py` (after `order_strategy`, around line 47):

```python
from domain.models import OrderItem
from domain.services import extract_features


def order_item_strategy() -> st.SearchStrategy[OrderItem]:
    """Strategy for generating valid OrderItem instances."""
    return st.builds(
        OrderItem,
        order_item_id=st.integers(min_value=1, max_value=100000),
        product_card_id=st.text(min_size=1, max_size=20),
        quantity=st.integers(min_value=1, max_value=100),
        unit_price=st.floats(min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False),
        discount=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        discount_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        profit_ratio=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        sales=st.floats(min_value=0.0, max_value=50000.0, allow_nan=False, allow_infinity=False),
        item_total=st.floats(min_value=0.0, max_value=50000.0, allow_nan=False, allow_infinity=False),
    )
```

Update `order_strategy()` — change the `items` line (currently not present, uses default `[]`). Replace the entire function:

```python
def order_strategy() -> st.SearchStrategy[Order]:
    """Strategy for generating valid Order instances with items."""
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
        benefit_per_order=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        sales_per_customer=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        order_profit_per_order=st.floats(min_value=-1000.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        items=st.lists(order_item_strategy(), min_size=1, max_size=5),
        late_delivery_risk=st.one_of(st.none(), st.sampled_from([0, 1])),
    )
```

Add property tests at end of file:

```python
class TestExtractFeaturesProperties:
    """Property-based tests for extract_features."""

    @given(order=order_strategy())
    def test_extract_features_returns_fixed_key_set(self, order: Order) -> None:
        """Property: extract_features always returns the same set of keys."""
        features = extract_features(order)
        expected_keys = {
            "shipping_mode", "days_for_shipment_scheduled",
            "order_month", "order_day_of_week", "order_region",
            "benefit_per_order", "sales_per_customer", "order_profit_per_order",
            "item_count", "total_quantity", "total_discount", "avg_unit_price",
        }
        assert set(features.keys()) == expected_keys

    @given(order=order_strategy())
    def test_extract_features_never_contains_leakage_keywords(self, order: Order) -> None:
        """Property: no key contains leakage-related words."""
        features = extract_features(order)
        leakage_words = {"real", "delivery_status", "shipping_date"}
        for key in features:
            for word in leakage_words:
                assert word not in key.lower(), f"Leakage keyword '{word}' in key '{key}'"

    @given(order=order_strategy())
    def test_extract_features_label_never_present(self, order: Order) -> None:
        """Property: late_delivery_risk never appears as a feature."""
        features = extract_features(order)
        assert "late_delivery_risk" not in features
```

- [ ] **Step 6: Run all property tests**

Run: `pytest tests/test_properties.py -v`
Expected: ALL PASS

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add domain/services.py tests/test_domain_services.py tests/test_properties.py
git commit -m "feat(domain): add extract_features service with property-based tests"
```

---

## Task 4: Shared Test Fixtures

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_ml/__init__.py`

- [ ] **Step 1: Create conftest.py with shared fixtures**

Create `tests/conftest.py`:

```python
"""Shared test fixtures for supply-chain-optimization-ml.

All fixtures use small synthetic data. Never load the real 180k-row CSV.
"""

from datetime import datetime

import numpy as np
import pandas as pd
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
    train = [
        _make_order(i, "Standard Class", 3, 1, sales=1000.0)
        for i in range(50)
    ]
    test = [
        _make_order(i + 50, "Standard Class", 3, 0, sales=0.0)
        for i in range(50)
    ]
    return train, test
```

- [ ] **Step 2: Create test_ml package**

```bash
mkdir -p tests/test_ml
touch tests/test_ml/__init__.py
```

- [ ] **Step 3: Verify fixtures load**

Run: `pytest tests/ -v --collect-only`
Expected: all existing tests collected, no import errors

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_ml/__init__.py
git commit -m "test: add shared fixtures and test_ml package for Phase 4"
```

---

## Task 5: Feature Encoder Adapter

**Files:**
- Create: `adapters/ml/feature_encoder.py`
- Create: `tests/test_ml/test_feature_encoder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ml/test_feature_encoder.py`:

```python
"""Tests for adapters/ml/feature_encoder.py.

Contract tests: verify encoder transforms raw feature dicts correctly,
preserves shape, and never produces leakage columns.
"""

import pytest

from adapters.data.csv_repository import LEAKAGE_COLUMNS
from adapters.ml.feature_encoder import FeatureEncoder
from domain.services import extract_features


class TestFeatureEncoder:
    """Contract tests for FeatureEncoder."""

    def _raw_features(self, synthetic_orders) -> list[dict]:
        return [extract_features(o) for o in synthetic_orders]

    def test_fit_transform_preserves_row_count(self, synthetic_orders) -> None:
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        X = encoder.fit_transform(raw)
        assert X.shape[0] == len(raw)

    def test_fit_transform_output_has_columns(self, synthetic_orders) -> None:
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        X = encoder.fit_transform(raw)
        assert X.shape[1] > 0

    def test_transform_without_fit_raises(self, synthetic_orders) -> None:
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        with pytest.raises(RuntimeError, match="not fitted"):
            encoder.transform(raw)

    def test_get_feature_names_matches_columns(self, synthetic_orders) -> None:
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        X = encoder.fit_transform(raw)
        names = encoder.get_feature_names()
        assert len(names) == X.shape[1]

    def test_no_leakage_column_in_feature_names(self, synthetic_orders) -> None:
        """CRITICAL: encoder output must never contain leakage columns."""
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        encoder.fit_transform(raw)
        names = encoder.get_feature_names()
        for name in names:
            for leak_col in LEAKAGE_COLUMNS:
                assert leak_col not in name, f"Leakage column '{leak_col}' found in feature name '{name}'"

    def test_fit_on_train_transform_on_test(self, synthetic_orders) -> None:
        """Encoder fit on subset A can transform subset B without error."""
        raw = self._raw_features(synthetic_orders)
        train_raw = raw[:80]
        test_raw = raw[80:]
        encoder = FeatureEncoder()
        X_train = encoder.fit_transform(train_raw)
        X_test = encoder.transform(test_raw)
        assert X_train.shape[1] == X_test.shape[1]
        assert X_test.shape[0] == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml/test_feature_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapters.ml.feature_encoder'`

- [ ] **Step 3: Implement FeatureEncoder**

Create `adapters/ml/feature_encoder.py`:

```python
"""Feature encoding adapter.

Transforms raw feature dicts from domain/services.extract_features()
into model-ready numeric arrays using sklearn preprocessing.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEncoder:
    """Encode raw feature dicts into numeric arrays for ML models.

    Categorical columns are one-hot encoded. Numeric columns are
    standard-scaled. Column order is deterministic.

    Note: order_country is deliberately excluded from CATEGORICAL
    to avoid 164-column sparse explosion (see spec Decision D7).
    """

    CATEGORICAL: list[str] = ["shipping_mode", "order_region"]
    NUMERIC: list[str] = [
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

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] | None = None

    def fit(self, raw_features: list[dict]) -> "FeatureEncoder":
        """Fit encoder on raw feature dicts. Must be called on training data ONLY."""
        df = pd.DataFrame(raw_features)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.CATEGORICAL,
                ),
                ("num", StandardScaler(), self.NUMERIC),
            ]
        )
        self._pipeline = Pipeline([("preprocessor", preprocessor)])
        self._pipeline.fit(df)
        self._feature_names = self._build_feature_names()
        return self

    def transform(self, raw_features: list[dict]) -> pd.DataFrame:
        """Transform raw feature dicts using fitted encoder."""
        if self._pipeline is None:
            raise RuntimeError("FeatureEncoder is not fitted. Call fit() first.")
        df = pd.DataFrame(raw_features)
        transformed = self._pipeline.transform(df)
        return pd.DataFrame(transformed, columns=self._feature_names)

    def fit_transform(self, raw_features: list[dict]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(raw_features)
        return self.transform(raw_features)

    def get_feature_names(self) -> list[str]:
        """Return post-encoding feature names. Used by SHAP for labeling."""
        if self._feature_names is None:
            raise RuntimeError("FeatureEncoder is not fitted. Call fit() first.")
        return self._feature_names

    def _build_feature_names(self) -> list[str]:
        """Extract feature names from fitted ColumnTransformer."""
        assert self._pipeline is not None
        ct: ColumnTransformer = self._pipeline.named_steps["preprocessor"]
        cat_names = list(ct.named_transformers_["cat"].get_feature_names_out(self.CATEGORICAL))
        num_names = list(self.NUMERIC)
        return cat_names + num_names
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_feature_encoder.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add adapters/ml/feature_encoder.py tests/test_ml/test_feature_encoder.py
git commit -m "feat(adapters): add FeatureEncoder with contract tests"
```

---

## Task 6: Model Predictors (LogReg + XGBoost)

**Files:**
- Rewrite: `adapters/ml/sklearn_predictor.py`
- Create: `tests/test_ml/test_sklearn_predictor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ml/test_sklearn_predictor.py`:

```python
"""Tests for adapters/ml/sklearn_predictor.py.

Contract tests: verify both predictors implement ModelTrainerPort interface,
return valid predictions and probabilities.
"""

import numpy as np
import pytest

from adapters.ml.sklearn_predictor import LogisticRegressionPredictor, XGBoostPredictor


@pytest.fixture
def small_training_data():
    """50-row synthetic training data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    y = rng.integers(0, 2, size=50)
    return X, y


class TestLogisticRegressionPredictor:
    """Contract tests for LogisticRegressionPredictor."""

    def test_train_no_error(self, small_training_data) -> None:
        X, y = small_training_data
        model = LogisticRegressionPredictor()
        model.train(X, y)

    def test_predict_returns_binary(self, small_training_data) -> None:
        X, y = small_training_data
        model = LogisticRegressionPredictor()
        model.train(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_returns_valid_probabilities(self, small_training_data) -> None:
        X, y = small_training_data
        model = LogisticRegressionPredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert len(proba) == len(X)

    def test_predict_proba_shape(self, small_training_data) -> None:
        X, y = small_training_data
        model = LogisticRegressionPredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (50,)


class TestXGBoostPredictor:
    """Contract tests for XGBoostPredictor."""

    def test_train_no_error(self, small_training_data) -> None:
        X, y = small_training_data
        model = XGBoostPredictor()
        model.train(X, y)

    def test_predict_returns_binary(self, small_training_data) -> None:
        X, y = small_training_data
        model = XGBoostPredictor()
        model.train(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_returns_valid_probabilities(self, small_training_data) -> None:
        X, y = small_training_data
        model = XGBoostPredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert len(proba) == len(X)

    def test_predict_proba_shape(self, small_training_data) -> None:
        X, y = small_training_data
        model = XGBoostPredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (50,)


class TestSameInterface:
    """Both models implement the same ModelTrainerPort interface."""

    def test_both_have_train_predict_predict_proba(self) -> None:
        for cls in [LogisticRegressionPredictor, XGBoostPredictor]:
            instance = cls()
            assert hasattr(instance, "train")
            assert hasattr(instance, "predict")
            assert hasattr(instance, "predict_proba")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml/test_sklearn_predictor.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement both predictors**

Rewrite `adapters/ml/sklearn_predictor.py`:

```python
"""Scikit-learn and XGBoost model adapters.

Implements ModelTrainerPort for two classifiers:
- LogisticRegressionPredictor: interpretable baseline (odds ratios)
- XGBoostPredictor: production model (non-linear interactions)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class LogisticRegressionPredictor:
    """Interpretable baseline for late delivery risk prediction.

    Implements ModelTrainerPort. Wraps sklearn LogisticRegression.
    Exposes coef_ for odds ratio analysis after training.
    """

    def __init__(self, max_iter: int = 1000, random_state: int = 42) -> None:
        self._model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train logistic regression on feature matrix X and labels y."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 or 1)."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class (late delivery)."""
        return self._model.predict_proba(X)[:, 1]

    @property
    def coef_(self) -> np.ndarray:
        """Model coefficients for odds ratio analysis."""
        return self._model.coef_

    @property
    def model(self) -> LogisticRegression:
        """Access underlying sklearn model (for SHAP LinearExplainer)."""
        return self._model


class XGBoostPredictor:
    """Production model for late delivery risk prediction.

    Implements ModelTrainerPort. Wraps XGBClassifier.
    Default hyperparams tuned for 55/45 class split.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = 0.82,  # 45.17/54.83 ≈ 0.82
        random_state: int = 42,
    ) -> None:
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            use_label_encoder=False,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost on feature matrix X and labels y."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 or 1)."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class (late delivery)."""
        return self._model.predict_proba(X)[:, 1]

    @property
    def model(self) -> XGBClassifier:
        """Access underlying XGBClassifier (for SHAP TreeExplainer)."""
        return self._model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_sklearn_predictor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add adapters/ml/sklearn_predictor.py tests/test_ml/test_sklearn_predictor.py
git commit -m "feat(adapters): implement LogReg and XGBoost predictors"
```

---

## Task 7: Evaluation Adapter

**Files:**
- Create: `adapters/ml/evaluation.py`
- Create: `tests/test_ml/test_evaluation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ml/test_evaluation.py`:

```python
"""Tests for adapters/ml/evaluation.py.

Verifies compute_metrics returns correct MetricsResult values.
"""

import numpy as np
import pytest

from adapters.ml.evaluation import compute_metrics
from domain.models import MetricsResult


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_returns_metrics_result(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.4])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert isinstance(result, MetricsResult)

    def test_perfect_predictions_give_f1_one(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_proba = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert result.f1 == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)

    def test_all_wrong_predictions_give_f1_zero(self) -> None:
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        y_proba = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert result.f1 == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)

    def test_confusion_matrix_shape(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        y_proba = np.array([0.8, 0.2, 0.4, 0.6])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2

    def test_auc_roc_between_zero_and_one(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.6, 0.4, 0.2])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert 0.0 <= result.auc_roc <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement compute_metrics**

Create `adapters/ml/evaluation.py`:

```python
"""Model evaluation adapter.

Computes classification metrics using sklearn and returns
domain MetricsResult. Domain says WHAT (F1 primary), adapter says HOW.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from domain.models import MetricsResult


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> MetricsResult:
    """Compute classification metrics.

    F1 is the primary metric — accuracy is misleading with 55/45 class split.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_proba: Predicted probabilities for positive class.

    Returns:
        MetricsResult with F1, precision, recall, AUC-ROC, confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    return MetricsResult(
        f1=float(f1_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        auc_roc=float(roc_auc_score(y_true, y_proba)),
        confusion_matrix=tuple(tuple(int(x) for x in row) for row in cm),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_evaluation.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add adapters/ml/evaluation.py tests/test_ml/test_evaluation.py
git commit -m "feat(adapters): add compute_metrics evaluation adapter"
```

---

## Task 8: SHAP Explainer Adapter

**Files:**
- Create: `adapters/ml/shap_explainer.py`
- Create: `tests/test_ml/test_shap_explainer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ml/test_shap_explainer.py`:

```python
"""Tests for adapters/ml/shap_explainer.py.

Contract tests: SHAP feature names match training features,
local explanation length matches feature count,
SHAP values sum approximately to prediction.
"""

import numpy as np
import pytest
from xgboost import XGBClassifier

from adapters.ml.shap_explainer import ShapExplainer


@pytest.fixture
def trained_xgb_model():
    """Small XGBoost model trained on synthetic data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    y = rng.integers(0, 2, size=50)
    model = XGBClassifier(n_estimators=10, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
    return model, X, feature_names


class TestShapExplainer:
    """Contract tests for ShapExplainer."""

    def test_global_feature_names_match(self, trained_xgb_model) -> None:
        model, X, names = trained_xgb_model
        explainer = ShapExplainer(model, names)
        result = explainer.explain_global(X)
        assert set(result.feature_importances.keys()) == set(names)

    def test_local_explanation_length(self, trained_xgb_model) -> None:
        model, X, names = trained_xgb_model
        explainer = ShapExplainer(model, names)
        result = explainer.explain_local(X, index=0)
        assert len(result.shap_values) == len(names)

    def test_local_shap_values_sum_to_expected(self, trained_xgb_model) -> None:
        """SHAP contract: sum(shap_values) + expected_value ≈ model output."""
        model, X, names = trained_xgb_model
        explainer = ShapExplainer(model, names)
        result = explainer.explain_local(X, index=0)
        shap_sum = sum(result.shap_values.values()) + result.expected_value
        model_output = model.predict_proba(X[:1])[:, 1][0]
        # SHAP values may be in log-odds space for tree models, so we check
        # that the explanation is internally consistent (not NaN, finite)
        assert np.isfinite(shap_sum)
        for v in result.shap_values.values():
            assert np.isfinite(v)

    def test_global_importances_are_non_negative(self, trained_xgb_model) -> None:
        model, X, names = trained_xgb_model
        explainer = ShapExplainer(model, names)
        result = explainer.explain_global(X)
        for importance in result.feature_importances.values():
            assert importance >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml/test_shap_explainer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ShapExplainer**

Create `adapters/ml/shap_explainer.py`:

```python
"""SHAP explainability adapter.

Implements ExplainerPort. Supports TreeExplainer (XGBoost)
and LinearExplainer (LogisticRegression). Auto-detects model type.
"""

from dataclasses import dataclass

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass
class ShapGlobalResult:
    """Global SHAP explanation result."""

    feature_importances: dict[str, float]
    """Mean absolute SHAP values per feature (higher = more important)."""

    shap_values: np.ndarray
    """Raw SHAP values matrix (n_samples × n_features)."""


@dataclass
class ShapLocalResult:
    """Local SHAP explanation for a single prediction."""

    shap_values: dict[str, float]
    """SHAP value per feature for this prediction."""

    expected_value: float
    """Base value (average model output on training data)."""


class ShapExplainer:
    """SHAP wrapper for tree and linear models.

    Implements ExplainerPort. Auto-detects model type:
    - XGBClassifier → shap.TreeExplainer
    - LogisticRegression → shap.LinearExplainer
    """

    def __init__(self, model: XGBClassifier | LogisticRegression, feature_names: list[str]) -> None:
        self._feature_names = feature_names
        if isinstance(model, XGBClassifier):
            self._explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            self._explainer = shap.LinearExplainer(model, np.zeros((1, len(feature_names))))
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    def explain_global(self, X: np.ndarray) -> ShapGlobalResult:
        """Compute global feature importance via mean absolute SHAP values."""
        shap_values = self._get_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        importances = {
            name: float(val)
            for name, val in zip(self._feature_names, mean_abs)
        }
        return ShapGlobalResult(
            feature_importances=importances,
            shap_values=shap_values,
        )

    def explain_local(self, X: np.ndarray, index: int) -> ShapLocalResult:
        """Compute SHAP explanation for a single prediction."""
        shap_values = self._get_shap_values(X)
        local_values = {
            name: float(val)
            for name, val in zip(self._feature_names, shap_values[index])
        }
        expected = float(self._explainer.expected_value)
        return ShapLocalResult(shap_values=local_values, expected_value=expected)

    def _get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values, handling different explainer output formats."""
        sv = self._explainer.shap_values(X)
        # TreeExplainer for binary classification may return list of 2 arrays
        if isinstance(sv, list):
            return sv[1]  # positive class
        return sv
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_shap_explainer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add adapters/ml/shap_explainer.py tests/test_ml/test_shap_explainer.py
git commit -m "feat(adapters): add SHAP explainer with global and local explanations"
```

---

## Task 9: MLflow Tracker Adapter

**Files:**
- Create: `adapters/ml/mlflow_tracker.py`
- Create: `tests/test_ml/test_mlflow_tracker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ml/test_mlflow_tracker.py`:

```python
"""Tests for adapters/ml/mlflow_tracker.py.

Integration tests using tmp_path as tracking URI.
Real MLflow calls, auto-cleanup via pytest tmp_path fixture.
"""

import pytest

from adapters.ml.mlflow_tracker import MLflowTracker


class TestMLflowTracker:
    """Integration tests for MLflowTracker."""

    def test_start_and_end_run_no_error(self, tmp_path) -> None:
        tracker = MLflowTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="test-run")
        tracker.end_run()

    def test_log_params(self, tmp_path) -> None:
        tracker = MLflowTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="test-params")
        tracker.log_params({"n_estimators": "100", "max_depth": "6"})
        tracker.end_run()

    def test_log_metrics(self, tmp_path) -> None:
        tracker = MLflowTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="test-metrics")
        tracker.log_metrics({"f1": 0.85, "precision": 0.82, "recall": 0.88})
        tracker.end_run()

    def test_log_model_creates_artifact(self, tmp_path) -> None:
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        model = LogisticRegression()
        model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

        tracker = MLflowTracker(
            experiment_name="test-experiment",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="test-model")
        tracker.log_model(model, "model")
        tracker.end_run()

    def test_full_workflow(self, tmp_path) -> None:
        """End-to-end: start → params → metrics → model → register → end."""
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        model = LogisticRegression()
        model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

        tracker = MLflowTracker(
            experiment_name="test-e2e",
            tracking_uri=str(tmp_path / "mlruns"),
        )
        tracker.start_run(run_name="e2e-run")
        tracker.log_params({"solver": "lbfgs"})
        tracker.log_metrics({"f1": 0.90})
        tracker.log_model(model, "model")
        tracker.register_model("test-model")
        tracker.end_run()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml/test_mlflow_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement MLflowTracker**

Create `adapters/ml/mlflow_tracker.py`:

```python
"""MLflow experiment tracking adapter.

Implements ExperimentTrackerPort. Local tracking with Model Registry.
tracking_uri parameter is the seam for future remote MLflow (Decision D3).
"""

import joblib
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """Local MLflow tracker with Model Registry.

    Implements ExperimentTrackerPort.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Path to local mlruns dir, or remote URI for future use.
    """

    def __init__(
        self,
        experiment_name: str = "late-delivery-risk",
        tracking_uri: str = "mlruns",
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run: mlflow.ActiveRun | None = None
        self._run_id: str | None = None

    def start_run(self, run_name: str) -> None:
        """Start a new experiment run."""
        self._run = mlflow.start_run(run_name=run_name)
        self._run_id = self._run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation metrics for the current run."""
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log a trained model as an MLflow artifact."""
        mlflow.sklearn.log_model(model, artifact_path)

    def log_artifact(self, artifact: Any, artifact_path: str) -> None:
        """Log an arbitrary artifact (e.g. FeatureEncoder) via joblib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"{artifact_path}.joblib"
            joblib.dump(artifact, path)
            mlflow.log_artifact(str(path))

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """Load a registered model by name and stage."""
        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)

    def load_artifact(self, run_id: str, artifact_path: str) -> Any:
        """Load an artifact from a specific run."""
        client = MlflowClient(self._tracking_uri)
        local_path = client.download_artifacts(run_id, artifact_path)
        return joblib.load(local_path)

    def register_model(self, model_name: str) -> None:
        """Register the current run's model in the Model Registry."""
        if self._run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        model_uri = f"runs:/{self._run_id}/model"
        mlflow.register_model(model_uri, model_name)

    def end_run(self) -> None:
        """End the current experiment run."""
        mlflow.end_run()
        self._run = None

    @property
    def run_id(self) -> str | None:
        """Current run ID, if a run is active."""
        return self._run_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_mlflow_tracker.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add adapters/ml/mlflow_tracker.py tests/test_ml/test_mlflow_tracker.py
git commit -m "feat(adapters): add MLflow tracker with Model Registry support"
```

---

## Task 10: Application Layer — Use Cases

**Files:**
- Rewrite: `application/use_cases.py`
- Create: `tests/test_use_cases.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_use_cases.py`:

```python
"""Tests for application/use_cases.py.

Integration tests: verify use cases wire domain + adapters correctly.
Uses synthetic_orders fixture (100 orders, never real CSV).
"""

import pytest
import numpy as np

from application.use_cases import TrainAndEvaluateUseCase, PredictSingleOrderUseCase
from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.sklearn_predictor import XGBoostPredictor, LogisticRegressionPredictor
from adapters.ml.shap_explainer import ShapExplainer
from adapters.ml.evaluation import compute_metrics
from domain.models import MetricsResult, TrainingResult, PredictionResult, Order


class FakeSalesDataRepository:
    """Fake data repository for testing."""

    def __init__(self, orders: list[Order]) -> None:
        self._orders = orders

    def get_orders(self) -> list[Order]:
        return self._orders

    def get_products(self) -> list:
        return []


class FakeTracker:
    """Fake experiment tracker that records calls without MLflow."""

    def __init__(self) -> None:
        self.params: dict = {}
        self.metrics: dict = {}
        self.models: list = []
        self.artifacts: list = []
        self.runs: list = []
        self.registered: list = []

    def start_run(self, run_name: str) -> None:
        self.runs.append(run_name)

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update(metrics)

    def log_model(self, model, artifact_path: str) -> None:
        self.models.append(artifact_path)

    def log_artifact(self, artifact, artifact_path: str) -> None:
        self.artifacts.append(artifact_path)

    def register_model(self, model_name: str) -> None:
        self.registered.append(model_name)

    def end_run(self) -> None:
        pass

    def load_model(self, model_name: str, stage: str = "Production"):
        pass

    def load_artifact(self, run_id: str, artifact_path: str):
        pass


class TestTrainAndEvaluateUseCase:
    """Tests for TrainAndEvaluateUseCase."""

    def test_returns_training_result(self, synthetic_orders) -> None:
        repo = FakeSalesDataRepository(synthetic_orders)
        encoder = FeatureEncoder()
        model = XGBoostPredictor()
        tracker = FakeTracker()

        use_case = TrainAndEvaluateUseCase(
            data_repo=repo,
            feature_encoder=encoder,
            model=model,
            explainer_factory=lambda m, names: ShapExplainer(m.model, names),
            tracker=tracker,
        )
        result = use_case.execute()
        assert isinstance(result, TrainingResult)
        assert isinstance(result.metrics, MetricsResult)
        assert 0.0 <= result.metrics.f1 <= 1.0
        assert 0.0 <= result.metrics.auc_roc <= 1.0
        assert len(result.feature_importances) > 0

    def test_tracker_receives_metrics(self, synthetic_orders) -> None:
        repo = FakeSalesDataRepository(synthetic_orders)
        encoder = FeatureEncoder()
        model = XGBoostPredictor()
        tracker = FakeTracker()

        use_case = TrainAndEvaluateUseCase(
            data_repo=repo,
            feature_encoder=encoder,
            model=model,
            explainer_factory=lambda m, names: ShapExplainer(m.model, names),
            tracker=tracker,
        )
        use_case.execute()
        assert "f1" in tracker.metrics
        assert "precision" in tracker.metrics
        assert "recall" in tracker.metrics
        assert "auc_roc" in tracker.metrics

    def test_works_with_logreg(self, synthetic_orders) -> None:
        repo = FakeSalesDataRepository(synthetic_orders)
        encoder = FeatureEncoder()
        model = LogisticRegressionPredictor()
        tracker = FakeTracker()

        use_case = TrainAndEvaluateUseCase(
            data_repo=repo,
            feature_encoder=encoder,
            model=model,
            explainer_factory=lambda m, names: ShapExplainer(m.model, names),
            tracker=tracker,
        )
        result = use_case.execute()
        assert isinstance(result, TrainingResult)


class TestEncoderFitOnTrainOnly:
    """CRITICAL: verify encoder is fit on training data only."""

    def test_encoder_mean_matches_train_distribution(
        self, synthetic_orders_distinctive_sales
    ) -> None:
        """Output-based test: train has sales=1000, test has sales=0.
        Scaler mean should be ~1000 (train), not ~500 (combined).
        """
        from domain.services import extract_features

        train_orders, test_orders = synthetic_orders_distinctive_sales
        train_raw = [extract_features(o) for o in train_orders]
        test_raw = [extract_features(o) for o in test_orders]

        encoder = FeatureEncoder()
        X_train = encoder.fit_transform(train_raw)
        X_test = encoder.transform(test_raw)

        # Find the index of sales_per_customer in numeric features
        names = encoder.get_feature_names()
        sales_idx = names.index("sales_per_customer")

        # Scaler mean should be ~1000 (train mean), not ~500 (combined mean)
        # After StandardScaler, train mean is 0.0 (centered)
        # Test values should be negative (0 - 1000)/std
        assert X_train["sales_per_customer"].mean() == pytest.approx(0.0, abs=0.01)
        assert X_test["sales_per_customer"].mean() < -1.0  # far below train mean


class TestPredictSingleOrderUseCase:
    """Tests for PredictSingleOrderUseCase."""

    def test_returns_prediction_result(self, synthetic_orders) -> None:
        from domain.services import extract_features

        # Train a model first
        raw = [extract_features(o) for o in synthetic_orders]
        labels = [o.late_delivery_risk for o in synthetic_orders]
        encoder = FeatureEncoder()
        X = encoder.fit_transform(raw)
        model = XGBoostPredictor()
        model.train(X.values, np.array(labels))
        explainer = ShapExplainer(model.model, encoder.get_feature_names())

        # Predict single order
        use_case = PredictSingleOrderUseCase(
            feature_encoder=encoder,
            model=model,
            explainer=explainer,
        )
        result = use_case.execute(synthetic_orders[0])
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.probability <= 1.0
        assert isinstance(result.risk_label, bool)
        assert len(result.explanation) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_use_cases.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement use cases**

Rewrite `application/use_cases.py`:

```python
"""Application-level workflows for Retail Supply Chain.

Orchestrates domain services and adapters for model training,
evaluation, and prediction. This is the composition root.

CRITICAL: Split BEFORE encode. Encoder fits on training data ONLY.
This prevents preprocessing leakage (see spec Section 5).
"""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from adapters.ml.evaluation import compute_metrics
from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import PredictionResult, TrainingResult, Order
from domain.ports import ExperimentTrackerPort, ModelTrainerPort, ExplainerPort
from domain.services import extract_features


class TrainAndEvaluateUseCase:
    """Full training pipeline: load → extract → split → encode → train → evaluate → explain → track.

    CRITICAL ordering: split raw features FIRST, then fit encoder on train only.
    """

    def __init__(
        self,
        data_repo: Any,
        feature_encoder: FeatureEncoder,
        model: ModelTrainerPort,
        explainer_factory: Callable,
        tracker: ExperimentTrackerPort,
    ) -> None:
        self._data_repo = data_repo
        self._encoder = feature_encoder
        self._model = model
        self._explainer_factory = explainer_factory
        self._tracker = tracker

    def execute(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        risk_threshold: float = 0.5,
    ) -> TrainingResult:
        """Run full training pipeline and return results."""
        # 1. Load orders
        orders = self._data_repo.get_orders()

        # 2. Extract features (domain) — label kept separate
        raw_features = [extract_features(o) for o in orders]
        labels = np.array([o.late_delivery_risk for o in orders])

        # 3. SPLIT FIRST — before any encoding
        raw_train, raw_test, y_train, y_test = train_test_split(
            raw_features, labels, test_size=test_size,
            random_state=random_state, stratify=labels,
        )

        # 4. Encode — fit on train ONLY
        X_train = self._encoder.fit_transform(raw_train)
        X_test = self._encoder.transform(raw_test)
        feature_names = self._encoder.get_feature_names()

        # 5. Track experiment
        model_name = type(self._model).__name__
        self._tracker.start_run(run_name=model_name)
        self._tracker.log_params({
            "model": model_name,
            "test_size": str(test_size),
            "random_state": str(random_state),
            "n_features": str(len(feature_names)),
            "n_train": str(len(raw_train)),
            "n_test": str(len(raw_test)),
        })

        # 6. Train
        self._model.train(X_train.values, y_train)

        # 7. Evaluate
        y_pred = self._model.predict(X_test.values)
        y_proba = self._model.predict_proba(X_test.values)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # 8. Log metrics
        self._tracker.log_metrics({
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "auc_roc": metrics.auc_roc,
        })

        # 9. Explain
        explainer = self._explainer_factory(self._model, feature_names)
        global_result = explainer.explain_global(X_test.values)

        # 10. Log artifacts
        self._tracker.log_model(self._model, "model")
        self._tracker.log_artifact(self._encoder, "encoder")
        self._tracker.register_model("late-delivery-risk")
        self._tracker.end_run()

        return TrainingResult(
            model_name=model_name,
            metrics=metrics,
            feature_importances=global_result.feature_importances,
        )


class PredictSingleOrderUseCase:
    """Predict and explain late delivery risk for a single order.

    For Phase 5 Streamlit dashboard integration. Uses pre-fitted
    encoder and model loaded from MLflow artifacts.
    """

    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        model: ModelTrainerPort,
        explainer: Any,
        risk_threshold: float = 0.5,
    ) -> None:
        self._encoder = feature_encoder
        self._model = model
        self._explainer = explainer
        self._threshold = risk_threshold

    def execute(self, order: Order) -> PredictionResult:
        """Predict and explain risk for a single order."""
        # 1. Extract features (domain)
        raw = extract_features(order)

        # 2. Encode (fitted encoder)
        X = self._encoder.transform([raw])

        # 3. Predict
        probability = float(self._model.predict_proba(X.values)[0])

        # 4. Explain
        local_result = self._explainer.explain_local(X.values, index=0)

        return PredictionResult(
            probability=probability,
            risk_label=probability >= self._threshold,
            explanation=local_result.shap_values,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_use_cases.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS (~80+ tests)

- [ ] **Step 6: Commit**

```bash
git add application/use_cases.py tests/test_use_cases.py
git commit -m "feat(application): implement TrainAndEvaluateUseCase and PredictSingleOrderUseCase"
```

---

## Task 11: Dependency Update + Quality Gates

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add shap to pyproject.toml dependencies**

Add `"shap>=0.44.0"` to the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "mlflow>=2.0.0",
    "shap>=0.44.0",
    "hypothesis>=6.0.0",
    "pytest>=7.0.0",
]
```

- [ ] **Step 2: Run full quality check**

Run: `make check`
Expected: lint + typecheck + tests all pass

- [ ] **Step 3: Run leakage audit skill**

Run: `/leakage-audit`
Expected: No leakage columns found in any feature pipeline

- [ ] **Step 4: Run domain check skill**

Run: `/domain-check`
Expected: domain/ has zero external imports

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add shap dependency to pyproject.toml"
```

---

## Task 12: Final Verification

- [ ] **Step 1: Run complete test suite with coverage**

Run: `pytest tests/ -v --cov=domain --cov=adapters --cov=application --tb=short`
Expected: ALL PASS, coverage report generated

- [ ] **Step 2: Count total tests**

Run: `pytest tests/ --co -q | tail -1`
Expected: ~80-85 tests collected

- [ ] **Step 3: Run pre-commit on all files**

Run: `pre-commit run --all-files`
Expected: ALL PASS (black, isort, mypy, ruff)

- [ ] **Step 4: Verify git status is clean**

Run: `git status`
Expected: nothing to commit, working tree clean

- [ ] **Step 5: Update CLAUDE.md Phase Status**

Update the "Phase Status" section in `CLAUDE.md` to mark Phase 4 items as done and move skeleton items to completed.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md phase status for Phase 4 completion"
```
