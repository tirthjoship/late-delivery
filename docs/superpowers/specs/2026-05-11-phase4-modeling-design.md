# Phase 4 Design Spec: Predictive Modeling & Explainability

**Date:** 2026-05-11
**Phase:** 4 of 5
**Status:** Approved (brainstorming complete)
**Prior art:** Phases 0-3 complete (scaffold, EDA, domain + 56 tests)

---

## 1. Scope

Two supervised models compared rigorously:

- **Logistic Regression** — interpretable baseline (odds ratios)
- **XGBoost** — production model (non-linear interactions)

Plus:

- **SHAP** — global feature importance + local (single-order) explanations
- **MLflow** — local experiment tracking + Model Registry
- **Feature engineering** — domain extracts WHAT, adapter encodes HOW

### Out of scope (deferred)

| Item | Deferred to |
|------|-------------|
| K-Means clustering | Phase 4.5 |
| Chi-squared validation | Already done in EDA |
| SHAP dependence/cohort plots | Phase 5 (Streamlit dashboard) |
| Remote MLflow server | Future (architecture is C-ready) |
| Performance regression tests | CI gate after stable baseline |

---

## 2. Architecture: Port Per Concern

Approach 2 from brainstorming. Each ML responsibility gets its own port and adapter.

### Dependency flow

```
adapters/ml/*  →  domain/ports.py  ←  application/use_cases.py
                  domain/services.py
                  domain/models.py
```

No adapter imports another adapter. Application imports domain + injects adapters. Domain imports nothing external.

### New ports (`domain/ports.py`)

```python
class ModelTrainerPort(Protocol):
    """Train and predict late delivery risk."""
    def train(self, X: Any, y: Any) -> None: ...
    def predict(self, X: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...

class ExplainerPort(Protocol):
    """Explain model predictions."""
    def explain_global(self, X: Any) -> Any: ...
    def explain_local(self, X: Any, index: int) -> Any: ...

class ExperimentTrackerPort(Protocol):
    """Track experiments, log metrics, register models."""
    def start_run(self, run_name: str) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_model(self, model: Any, artifact_path: str) -> None: ...
    def log_artifact(self, artifact: Any, artifact_path: str) -> None: ...
    def load_model(self, model_name: str, stage: str = "Production") -> Any: ...
    def load_artifact(self, run_id: str, artifact_path: str) -> Any: ...
    def register_model(self, model_name: str) -> None: ...
    def end_run(self) -> None: ...
```

Ports use `Any` for X/y because domain cannot import numpy/pandas. Adapters type-narrow internally.

---

## 3. Domain Layer Changes

### `domain/services.py` — new function

```python
def extract_features(order: Order) -> dict[str, float | str]:
    """Extract ML features from order using only pre-shipment data.
    
    Returns raw feature dict. Encoding (one-hot, scaling) is adapter's job.
    Label (late_delivery_risk) is NOT included — kept separate to prevent leakage.
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
            if order.items else 0.0
        ),
    }
```

Design decisions:

- **`order_country` excluded from encoding** — 164 unique values creates sparse explosion. `order_region` captures geography. Country kept in output dict but excluded from `FeatureEncoder.CATEGORICAL` list. Revisit if SHAP shows region signal.
- **Label NOT in output** — `late_delivery_risk` extracted separately in application layer.
- **Derived features** (item_count, total_quantity, avg_unit_price) computed from OrderItems — business logic, belongs in domain.
- **Returns `str` for categoricals** — one-hot encoding is adapter's responsibility.

### `domain/models.py` — new result dataclasses

```python
@dataclass(frozen=True)
class MetricsResult:
    f1: float
    precision: float
    recall: float
    auc_roc: float
    confusion_matrix: tuple[tuple[int, ...], ...]

@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    metrics: MetricsResult
    feature_importances: dict[str, float]

@dataclass(frozen=True)
class PredictionResult:
    probability: float
    risk_label: bool          # probability > threshold
    explanation: dict[str, float]  # feature_name -> shap_value
```

No sklearn/shap imports. Plain data containers.

### No changes to `domain/exceptions.py`

Existing `DomainError`, `InvalidOrderDataError`, `InvalidPolicyError` suffice.

---

## 4. Adapter Layer

### `adapters/ml/feature_encoder.py` — NEW

Framework bridge. Takes raw feature dicts from domain, produces model-ready arrays.

```python
class FeatureEncoder:
    CATEGORICAL = ["shipping_mode", "order_region"]
    NUMERIC = [
        "days_for_shipment_scheduled", "order_month", "order_day_of_week",
        "benefit_per_order", "sales_per_customer", "order_profit_per_order",
        "item_count", "total_quantity", "total_discount", "avg_unit_price",
    ]

    def fit(self, raw_features: list[dict]) -> "FeatureEncoder": ...
    def transform(self, raw_features: list[dict]) -> pd.DataFrame: ...
    def fit_transform(self, raw_features: list[dict]) -> pd.DataFrame: ...
    def get_feature_names(self) -> list[str]: ...
```

Internals: `ColumnTransformer` with `OneHotEncoder` (handle_unknown="ignore") for categoricals + `StandardScaler` for numerics. Serializable via joblib — saved as MLflow artifact alongside model.

`get_feature_names()` returns post-encoding column names for SHAP labeling.

Note: `order_country` is in `extract_features()` output but NOT in `CATEGORICAL` — deliberately excluded to avoid 164-column sparse explosion. Easy to toggle back.

### `adapters/ml/sklearn_predictor.py` — REWRITE

Two classes, same interface:

```python
class LogisticRegressionPredictor:
    """Interpretable baseline. Implements ModelTrainerPort.
    Exposes .coef_ for odds ratio analysis."""

class XGBoostPredictor:
    """Production model. Implements ModelTrainerPort.
    Default: scale_pos_weight for 55/45 split, eval_metric='logloss'."""
```

Both implement `train(X, y)`, `predict(X)`, `predict_proba(X)`.

### `adapters/ml/shap_explainer.py` — NEW

```python
class ShapExplainer:
    """Implements ExplainerPort. Auto-detects model type."""

    def __init__(self, model, feature_names: list[str]): ...
    def explain_global(self, X) -> ShapGlobalResult: ...
    def explain_local(self, X, index: int) -> ShapLocalResult: ...
```

- `shap.TreeExplainer` for XGBoost, `shap.LinearExplainer` for LogReg — auto-detects
- Returns `matplotlib.Figure` objects — application decides where to save
- `ShapGlobalResult` / `ShapLocalResult` are adapter-internal dataclasses (not domain — they contain figure objects)

### `adapters/ml/mlflow_tracker.py` — NEW

```python
class MLflowTracker:
    """Implements ExperimentTrackerPort. Local + Model Registry."""

    def __init__(
        self,
        experiment_name: str = "late-delivery-risk",
        tracking_uri: str = "mlruns",
    ): ...
```

- `tracking_uri` parameter = seam for future remote MLflow (Decision: B now, C later)
- `register_model()` promotes best run as `late-delivery-risk/Production`
- `log_artifact()` saves FeatureEncoder alongside model
- `load_model()` + `load_artifact()` for Phase 5 Streamlit integration

### `adapters/ml/evaluation.py` — NEW

```python
def compute_metrics(y_true, y_pred, y_proba) -> MetricsResult:
    """Compute F1 (primary), precision, recall, AUC-ROC, confusion matrix.
    Returns domain MetricsResult. Uses sklearn.metrics internally."""
```

Domain says WHAT (MetricsResult structure, F1 is primary). Adapter says HOW (sklearn computation).

### File tree

```
adapters/ml/
├── __init__.py
├── feature_encoder.py      NEW
├── sklearn_predictor.py    REWRITE
├── shap_explainer.py       NEW
├── mlflow_tracker.py       NEW
├── evaluation.py           NEW
└── pytorch_predictor.py    UNCHANGED (stub)
```

---

## 5. Application Layer

### CRITICAL: Preprocessing leakage prevention

**Split BEFORE encoding.** Encoder fits on training data only. Test data is transformed with pre-fitted encoder. This was caught during design grilling (Question 1).

### `TrainAndEvaluateUseCase`

```python
class TrainAndEvaluateUseCase:
    def __init__(
        self,
        data_repo: SalesDataRepository,
        feature_encoder: FeatureEncoder,
        model: ModelTrainerPort,
        explainer_factory: Callable,
        tracker: ExperimentTrackerPort,
    ): ...

    def execute(self, test_size: float = 0.2, random_state: int = 42) -> TrainingResult: ...
```

Execute flow (corrected after grill):

```
 1. orders = data_repo.get_orders()
 2. raw_features = [extract_features(o) for o in orders]     <- domain
 3. labels = [o.late_delivery_risk for o in orders]           <- separate
 4. raw_train, raw_test, y_train, y_test = train_test_split(  <- SPLIT FIRST
        raw_features, labels, stratify=labels)
 5. X_train = feature_encoder.fit_transform(raw_train)        <- fit on TRAIN ONLY
 6. X_test = feature_encoder.transform(raw_test)              <- transform (no fit!)
 7. tracker.start_run(run_name=...)
 8. tracker.log_params(...)
 9. model.train(X_train, y_train)
10. y_pred = model.predict(X_test)
11. y_proba = model.predict_proba(X_test)
12. metrics = compute_metrics(y_test, y_pred, y_proba)
13. tracker.log_metrics(metrics)
14. explainer = explainer_factory(model, feature_names)
15. global_explanation = explainer.explain_global(X_test)
16. tracker.log_model(model, "model")
17. tracker.log_artifact(feature_encoder, "encoder")
18. tracker.register_model("late-delivery-risk")
19. tracker.end_run()
20. return TrainingResult(metrics, global_explanation)
```

Steps 4-6 are the preprocessing leakage fix: split raw dicts first, then fit on train only.

### `PredictSingleOrderUseCase`

```python
class PredictSingleOrderUseCase:
    """For Phase 5 Streamlit dashboard. Predict + explain single order."""

    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        model: ModelTrainerPort,
        explainer: ExplainerPort,
    ): ...

    def execute(self, order: Order) -> PredictionResult: ...
```

Execute flow:

```
1. raw = extract_features(order)               <- domain
2. X = feature_encoder.transform([raw])        <- fitted encoder
3. probability = model.predict_proba(X)[0]     <- risk score
4. explanation = explainer.explain_local(X, 0)  <- why THIS order
5. return PredictionResult(probability, explanation)
```

Model, encoder, and explainer loaded from MLflow artifacts (via `ExperimentTrackerPort.load_model()` / `load_artifact()`).

---

## 6. Testing Strategy

Contract + invariant tests. ~25-30 new tests, targeting ~80-85 total.

### Principles

- All tests use small fixtures — never load 180k-row CSV
- Property-based tests with Hypothesis for domain invariants
- MLflow tests use `tmp_path` as `tracking_uri` — real calls, auto-cleanup
- Boundary tests verify contracts, not implementations

### Domain tests (extend existing)

**`test_domain_services.py`:**

- `test_extract_features_returns_only_presshipment_keys` — no key matches LEAKAGE_COLUMNS
- `test_extract_features_empty_items_returns_zero_avg_price`
- `test_extract_features_returns_correct_types` — str for categoricals, float/int for numerics

**`test_properties.py`:**

- Add `order_item_strategy()` — generates valid OrderItem objects
- Update `order_strategy()` — `items=st.lists(order_item_strategy(), min_size=1, max_size=5)`
- Property: for ANY valid Order, `extract_features()` returns dict with fixed key set
- Property: for ANY valid Order, no key name contains "real" or "delivery_status"

### Adapter tests (`tests/test_ml/`)

**`test_feature_encoder.py`:**

- `test_fit_transform_output_shape` — n_rows preserved, n_cols = expected
- `test_transform_without_fit_raises`
- `test_get_feature_names_matches_columns`
- `test_no_leakage_column_in_output` — CRITICAL: LEAKAGE_COLUMNS intersect feature_names == empty

**`test_sklearn_predictor.py`:**

- `test_predict_returns_binary_labels` — all values in {0, 1}
- `test_predict_proba_returns_valid_probabilities` — all values in [0.0, 1.0]
- `test_train_on_small_fixture_no_error` — 100-row synthetic fixture
- `test_logreg_and_xgboost_same_interface` — both implement train/predict/predict_proba

**`test_shap_explainer.py`:**

- `test_global_feature_names_match_training_features`
- `test_local_explanation_has_correct_length` — one SHAP value per feature
- `test_shap_values_sum_to_expected_value` — SHAP contract: sum(shap_values) + expected ~ prediction

**`test_evaluation.py`:**

- `test_compute_metrics_returns_metrics_result`
- `test_perfect_predictions_give_f1_one`
- `test_all_wrong_predictions_give_f1_zero`

**`test_mlflow_tracker.py`:**

- `test_start_and_end_run_no_error` — uses `tmp_path` tracking URI
- `test_log_metrics_records_values`
- `test_log_model_creates_artifact`

### Application tests

**`test_use_cases.py`:**

- `test_train_and_evaluate_returns_training_result` — mock data_repo, 100 synthetic orders
- `test_predict_single_order_returns_prediction_result` — probability in [0, 1]
- `test_encoder_fitted_on_train_only` — output-based: train set has distinctive distribution (e.g., sales_per_customer=1000), verify scaler mean matches train only, not combined mean

### Fixtures (`tests/conftest.py`)

```python
@pytest.fixture
def synthetic_orders() -> list[Order]:
    """100 synthetic Orders with realistic distributions.
    ~55% late_delivery_risk=1. Mix of shipping modes, regions."""

@pytest.fixture
def small_feature_matrix() -> pd.DataFrame:
    """Pre-encoded 100x15 feature matrix for ML tests."""
```

---

## 7. Phase 4 Step 0: Quick Interaction EDA

Before modeling, add one notebook cell to `notebooks/eda_initial.ipynb`:

**Interaction analysis: shipping mode x scheduled days x order value**

```python
df.groupby(
    ['Shipping Mode', pd.cut(df['Days for shipment (scheduled)'], bins=5)]
)['Late_delivery_risk'].mean()
```

Purpose: confirm whether scheduled days has different effect per shipping mode. Informs whether interaction features worth engineering. 30 minutes max. SHAP on trained XGBoost extends this analysis deeper.

---

## 8. Decisions Log

| # | Decision | Choice | Rationale | Deferred option |
|---|----------|--------|-----------|-----------------|
| D1 | Model scope | LogReg + XGBoost | Interpretable vs powerful comparison | K-Means -> Phase 4.5 |
| D2 | Feature engineering split | Domain WHAT / Adapter HOW | Domain stays pure, testable | — |
| D3 | MLflow depth | Local + Model Registry | Shows lifecycle management | Remote server -> future |
| D4 | SHAP depth | Global + local | Individual explanations matter | Dependence plots -> Phase 5 |
| D5 | Testing depth | Contract + invariant | Catches leakage, preserves contracts | Perf regression -> CI later |
| D6 | Architecture | Port Per Concern | Each adapter swappable independently | — |
| D7 | Country feature | Exclude from encoder | 164 categories = sparse noise | Revisit if region shows signal |
| D8 | Preprocessing order | Split before encode | Prevents preprocessing leakage | — |
| D9 | compute_metrics location | adapters/ml/evaluation.py | Imports sklearn, returns domain type | — |
| D10 | Model deserialization | MLflow handles both | Encoder as separate artifact | — |
| D11 | MLflow test strategy | tmp_path integration | Real calls, auto-cleanup | — |
| D12 | Encoder fit verification | Output-based test | Tests contract not implementation | — |

---

## 9. Grill Findings

Issues caught during design review that changed the spec:

1. **Preprocessing leakage** — fit_transform before split leaked test distribution into encoder. Fixed: split first, fit on train only.
2. **compute_metrics location** — originally unspecified. Placed in `adapters/ml/evaluation.py` (imports sklearn).
3. **Model deserialization** — needed for Phase 5. MLflow handles via `load_model()` + `load_artifact()`.
4. **Country cardinality** — 164 unique countries would create sparse explosion. Excluded from encoder.
5. **EDA gaps** — interaction effects not analyzed. Added Phase 4 Step 0.
6. **OrderItem strategy missing** — Hypothesis tests only fuzzed empty-items path. Added `order_item_strategy()`.
7. **MLflow test side effects** — solved with `tmp_path` as tracking URI.
8. **Encoder fit verification** — output-based test (check scaler mean) instead of spy pattern.

---

## 10. Skills Usage Plan

| Phase | Skill | Purpose |
|-------|-------|---------|
| Planning | `superpowers:writing-plans` | Implementation plan from this spec |
| Implementation | `superpowers:test-driven-development` | Tests before code |
| Implementation | `superpowers:executing-plans` | Execute with review checkpoints |
| Quality | `/leakage-audit` | Verify LEAKAGE_COLUMNS excluded |
| Quality | `/domain-check` | Verify domain/ stays pure |
| Quality | `superpowers:verification-before-completion` | Verify before claiming done |
| Quality | `superpowers:systematic-debugging` | If tests fail during TDD |
| Review | `superpowers:requesting-code-review` | Review against this spec |
| Ship | `/pr` | PR creation |
| Ship | `caveman:caveman-commit` | Commit messages |
