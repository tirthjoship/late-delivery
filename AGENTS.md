# Coding Standards

## Python

- Python 3.12+
- Formatting: `black` (line-length 88)
- Type checking: `mypy` with strict mode enabled
- Linting: `ruff`
- Import sorting: `isort` (profile: black)
- No bare `except` — use specific exception types
- Type hints on all public function signatures
- Prefer `X | None` over `Optional[X]` (Python 3.12 syntax)

## Naming Conventions

- **Variables and functions**: `snake_case` (e.g., `get_orders`, `baseline_late_delivery_risk_flag`)
- **Classes**: `PascalCase` (e.g., `DataCoCSVRepository`, `Order`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `LEAKAGE_COLUMNS`, `HIGH_RISK_SHIPPING_MODES`)
- **Modules**: `snake_case` (e.g., `csv_repository.py`, `use_cases.py`)
- **Test functions**: `test_<description>` (e.g., `test_first_class_always_high_risk`)
- **Private methods**: prefix with `_` (e.g., `_parse_order_date`, `_safe_float`)

## Architecture Rules (NON-NEGOTIABLE)

- Hexagonal architecture: `domain/` → `adapters/` → `application/`
- `domain/` has ZERO imports from `adapters/`, `application/`, or external frameworks
- `domain/` imports ONLY: `typing`, `dataclasses`, `datetime`, `enum`, `collections.abc`
- Adapters implement domain port `Protocol` interfaces from `domain/ports.py`
- `application/` orchestrates domain + adapters — it is the composition root
- New external tool = new adapter. Never put framework code in `domain/`
- Domain models use `@dataclass(frozen=True)` for immutable entities (e.g., `Product`)

## Data Integrity Rules (NON-NEGOTIABLE)

- NEVER use as features: `Days for shipping (real)`, `Delivery Status`, `shipping date (DateOrders)`
- These are post-shipment data — using them is data leakage
- `LEAKAGE_COLUMNS` in `adapters/data/csv_repository.py` is the single source of truth
- Evaluate models with F1, precision, recall — NEVER accuracy alone (55/45 class split)
- All feature engineering must use only pre-shipment information
- When adding new data sources, audit every column for temporal leakage before use

## Testing Rules (NON-NEGOTIABLE)

- Tests use small fixtures — NEVER load the full 180k-row CSV
- Property-based tests with Hypothesis for domain invariants
- pytest with `-v --tb=short` default
- Test categories to cover:
  - **Happy path**: valid input, expected output
  - **Error path**: invalid input, correct exception raised
  - **Boundary**: exact edge of a condition
  - **Edge case**: extreme but valid input
- One logical assertion per test function
- Use `pytest.raises` for expected exceptions
- Use `pytest.approx` for float comparisons
- Use `tmp_path` fixture for file I/O tests

## Project Layout

```
domain/                 Pure business logic
├── models.py           Frozen dataclasses (Product, Order, OrderItem, Forecast)
├── ports.py            Protocol interfaces (SalesDataRepository)
├── services.py         Business rules (risk classification, baseline prediction)
└── exceptions.py       Domain-specific errors

adapters/               External connections
├── data/               Data source connectors
│   ├── csv_repository.py   DataCo CSV with leakage shield
│   ├── database_repository.py
│   └── api_client.py
├── ml/                 Model adapters
│   ├── sklearn_predictor.py
│   └── pytorch_predictor.py
└── visualization/      Charting adapters
    └── plotly_charts.py

application/            Orchestration
└── use_cases.py        Wires domain + adapters for business workflows

tests/                  Mirrors source layout
├── test_domain_models.py
├── test_domain_services.py
├── test_csv_repository.py
└── test_properties.py      Property-based (Hypothesis)

notebooks/              Exploration and EDA only — no production logic
data/raw/               Untouched source data (gitignored)
data/interim/           Intermediate artifacts (gitignored)
data/processed/         Model-ready data (gitignored)
```

## Git (NON-NEGOTIABLE)

- Commit format: `feat:` / `fix:` / `docs:` / `chore:` / `test:` followed by lowercase description, no period
- Keep commits small and focused
- Never commit directly to `main` or `dev` — use feature branches
- Branch naming: `feat/<slug>` or `fix/<slug>`
- PR target: `dev` (confirm with user before targeting `main`)
- Never commit secrets, raw data, model artifacts, or `.env` files
- Prefer new commits over `--amend` on pushed branches

## Commands

```bash
# Test
pytest -v --tb=short
pytest -v --cov=domain --cov=adapters --cov=application --tb=short

# Lint and format
pre-commit run --all-files

# Type check
mypy domain/ adapters/ application/ --strict

# Environment
conda activate supply-chain-ml
```

## Strong Preferences

These are not hard stops but should be followed unless there is a clear reason not to:

- Use structured logging over `print()` (loguru when added)
- Avoid heavyweight dependencies without justification (e.g., no TensorFlow for tabular data)
- Prefer `X | None` over `Optional[X]` for modern Python 3.12 annotations
- Type hints on private functions too when practical
