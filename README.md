# Supply Chain Optimization ML

An end-to-end Machine Learning project analyzing 180,000+ e-commerce orders from the [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) dataset.

## Business Questions

This project was built around three core operational questions that routinely come up in retail and logistics data science roles:

1. **When will an order be late — and why?**  
   Initial EDA revealed a 54.83% baseline late delivery rate across all shipping modes. First Class shipping fails 95.3% of the time — a structural logistics problem, not a demand issue. The goal is to predict late delivery risk *at the moment an order is placed*, using only information available at that point in time.

2. **Which products should we prioritize for inventory?**  
   Using ABC analysis across departments, I identified that Fan Shop drives the highest order volume (~66k orders) while Apparel yields the highest average profit margins (12.2%). This informs smarter restocking decisions.

3. **How do we forecast demand without overfitting to seasonality noise?**  
   Monthly sales trend analysis establishes a baseline for time-series forecasting, distinguishing genuine seasonal patterns from one-off spikes.

---

## Key Technical Decisions

- **Hexagonal Architecture:** The core ML business logic (`domain/`) is fully decoupled from data sources and UI. This means the domain model can be unit tested in isolation and the CSV adapter can be swapped for a database with zero changes to business logic.
- **Strict leakage prevention:** Features like `Days for shipping (real)` and `Delivery Status` are excluded from all modeling. They encode the outcome, not the predictors.
- **Property-based testing via Hypothesis:** Rather than writing individual test cases, I used generative testing to verify invariants (e.g. "delivery risk score must always be between 0 and 1 for any valid order").

---

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML Modeling | XGBoost, Scikit-Learn |
| Experiment Tracking | MLflow |
| Testing | Pytest, Hypothesis |
| Data | Pandas, NumPy |
| Code Quality | Black, Ruff, isort, Mypy |
| Architecture | Hexagonal (Ports & Adapters) |

---

## Developer Tooling

I treated this project as a software engineering exercise, not just a data science one. That meant using tools the same way a professional team would — for code quality, planning, and test coverage:

| Tool | How I Used It |
|---|---|
| **[Kiro](https://kiro.dev)** | Spec-driven planning — wrote acceptance criteria and EARS-notation requirements before any code. This kept the architecture from drifting mid-project. |
| **[Cursor](https://cursor.com)** | IDE with context-aware code completion. Practical for wiring ports, adapters, and services across multiple files simultaneously. |
| **Automated EDA pipeline** | Scripted data interrogation for leakage auditing, class imbalance checks, and visualization generation across the 180k+ record dataset. |
| **AI pair reviewer** | Used during the TDD phase to run `pytest` iteratively and trace root causes of failing property-based tests in `domain/services.py`. |

---

## How to Reproduce

This project uses a strict `conda` environment for full reproducibility.

### 1. Clone
```bash
git clone git@github.com:tirthjoship/supply-chain-optimization-ml.git
cd supply-chain-optimization-ml
```

### 2. Create the Environment
```bash
conda env create -f environment.yml
conda activate supply-chain-ml
```

### 3. Install Pre-commit Hooks
```bash
pre-commit install
```

### 4. Fetch the Dataset
The dataset is excluded from Git (91MB CSV). Download it via the Kaggle API:
```bash
kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis -p "data/raw" --unzip
```

### 5. Validate the Domain (Run Tests)
```bash
pytest -v
```
All 44 tests should pass before any modeling begins.

---

## Project Structure

```text
supply-chain-optimization-ml/
├── domain/            # Business rules — no external dependencies allowed here
├── adapters/          # CSV data reader, ML model adapter, visualization
├── application/       # Wires domain to adapters
├── tests/             # 44 unit + property-based tests
├── notebooks/         # EDA notebooks
├── data/              # Gitignored — raw, interim, processed data folders
├── environment.yml    # Conda environment spec
└── pyproject.toml     # Black, Ruff, Mypy config
```

---

## Project Phases

- ✅ Phase 0 — Governance & toolchain setup
- ✅ Phase 1 — Hexagonal architecture scaffold
- ✅ Phase 2 — EDA: late delivery risk baseline, class imbalance audit, ABC analysis
- ✅ Phase 3 — Domain implementation + 44 TDD tests
- 🔜 Phase 4 — XGBoost pipeline, SHAP explainability, Streamlit dashboard

---

## Author

**Tirth Joshi**  
Master of Data Science, University of British Columbia  
[github.com/tirthjoship](https://github.com/tirthjoship)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
