# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository. Read `AGENTS.md` for all coding standards, architecture rules, and testing requirements before touching any code.

## Project Context

Retail supply chain late delivery risk prediction using the DataCo Supply Chain dataset (~180k orders, $36.7M sales). Core finding from EDA: **54.83% of orders are delivered late** — this is a binary classification problem (late vs on-time) with a slight class imbalance.

The project uses hexagonal architecture (ports & adapters) so the domain logic stays pure and any ML framework, data source, or UI can be swapped without touching business rules.

## Commands

```bash
# Full quality check (lint + typecheck + test with coverage)
make check

# Individual targets
make test          # pytest -v --tb=short
make test-cov      # pytest with --cov-fail-under=90
make lint          # pre-commit run --all-files
make typecheck     # mypy strict on domain/ adapters/ application/
make setup         # conda env + pip install + pre-commit install

# Single test
pytest tests/test_domain_services.py::TestBaselineLateDeliveryRiskFlag -v
```

## Architecture

**Hexagonal (Ports & Adapters) with inward-pointing dependencies.**

```
adapters/     →  domain/  ←  application/
(external)       (pure)      (orchestration)
```

- `domain/` — Business rules, models, port interfaces, exceptions. ZERO external framework imports.
- `adapters/data/` — Data source connectors (CSV today, DB or API later).
- `adapters/ml/` — Model training and prediction adapters.
- `adapters/visualization/` — Charting and dashboard adapters.
- `application/` — Use case orchestration wiring domain + adapters.

Port interfaces in `domain/ports.py` define contracts. Adapters implement them. New tool = new adapter, never new domain code.

## Critical Domain Knowledge

**Data leakage — the biggest risk in this project.**

Three columns MUST NEVER be used as features:
- `Days for shipping (real)` — only known after delivery (leaks the outcome)
- `Delivery Status` — directly encodes late/on-time (IS the outcome)
- `shipping date (DateOrders)` — future information unavailable at prediction time

These are enforced via `LEAKAGE_COLUMNS` constant in `adapters/data/csv_repository.py`. The CSV adapter drops them by default.

**Shipping mode risk classification (from EDA):**
- First Class: **95.3% late** → always high-risk
- Second Class: **76.6% late** → always high-risk
- Standard Class: 38% late → risk depends on other factors
- Same Day: lowest late rate → not high-risk by mode alone

This evidence is codified in `domain/services.py` as `HIGH_RISK_SHIPPING_MODES`.

**Class imbalance:** 54.83% late vs 45.17% on-time. Not extreme, but enough that accuracy is misleading. Always evaluate with F1, precision, recall, and confusion matrix.

## Non-Negotiable Rules

Five hard stops — see `AGENTS.md` for full details:

1. **No framework imports in domain/** — domain/ imports only typing, dataclasses, datetime, enum
2. **No leakage columns as features** — Days for shipping (real), Delivery Status, shipping date
3. **Evaluate with F1, not accuracy** — 55/45 class split makes accuracy misleading
4. **No direct commits to main or dev** — feature branches only, PR to dev
5. **Tests use small fixtures** — never load the full 180k-row CSV in tests

## Phase Status

**Done:**
- Domain layer (models, ports, services, exceptions) — 280 lines
- CSV adapter with leakage shield — 307 lines
- Test suite (56 tests) — domain, adapter, property-based
- EDA notebook — full analysis of 180k orders
- CI workflows (test + lint + security) — 3 GitHub Actions
- Pre-commit hooks — black, isort, mypy strict, ruff, gitleaks, file hygiene
- Makefile — test, lint, typecheck, setup, check targets
- Agent Development Kit — 4 agents, 2 skills, guardrail hooks
- CLAUDE.md + AGENTS.md — project orientation and coding standards
- PR and issue templates

**Skeleton (1-line stubs):**
- `adapters/ml/sklearn_predictor.py` — ML model adapter
- `adapters/ml/pytorch_predictor.py` — neural net adapter
- `adapters/visualization/plotly_charts.py` — charting adapter
- `adapters/data/database_repository.py` — DB adapter
- `adapters/data/api_client.py` — API adapter
- `application/use_cases.py` — orchestration layer

**Planned:**
- ML model training adapter (framework TBD — architecture supports any)
- Experiment tracking (MLflow or similar)
- Model explainability (SHAP or similar)
- Application layer orchestration
- Streamlit dashboard
- Expanded CI (lint, security, release workflows)
- Observability (structured logging, tracing)
