# Supply Chain Optimization ML

An end-to-end Machine Learning project focusing on Demand Forecasting, Inventory Optimization (ABC Analysis), and Late Delivery Risk Prediction. 

This repository was designed to mirror production-grade ML engineering environments, specifically focusing on the operational challenges of retail and supply chain networks like Amazon, Walmart, or Costco. 

## Key Highlights

- **Quality & Standards:** Enforced strict typing (`mypy`), code formatting (`black`, `isort`), and linting (`ruff`) using pre-commit hooks.
- **Hexagonal Architecture:** The core ML business logic (`domain/`) is completely decoupled from external integrations like CSV files or visualization UIs. This prevents target leakage and keeps the models highly testable.
- **Test-Driven:** Implemented a suite of 44 tests using `pytest` alongside `Hypothesis` for edge-case property-based testing.
- **Deep EDA:** Analyzed a 180,000+ record supply chain dataset (`DataCo`). I discovered a 54.83% baseline late delivery rate, which drove a pivot from simple forecasting directly to proactive risk management.

## Technology Stack

- **Core:** Python 3.12
- **Testing:** Pytest, Hypothesis 
- **Data & Modeling:** Pandas, NumPy, XGBoost, Scikit-Learn
- **Experiment Tracking:** MLflow
- **Architecture:** Hexagonal (Ports & Adapters)

## Developer Tooling
I built this project utilizing modern developer tools to treat ML as a mature software discipline:
- **Cursor IDE:** Used as an intelligent autocomplete to speed up multi-file boilerplate and hexagonal scaffolding.
- **Claude:** Used as a pair-programming assistant specifically during the TDD phase to generate edge-case property tests (`Hypothesis`) and audit the domain logic prior to ML modeling.

## How to Reproduce the Environment

To ensure perfect reproducibility, this project uses a strict `conda` environment. 

### 1. Clone the Repository
```bash
git clone git@github.com:tirthjoship/supply-chain-optimization-ml.git
cd supply-chain-optimization-ml
```

### 2. Create and Activate the Conda Environment
The `environment.yml` handles all dependencies:
```bash
conda env create -f environment.yml
conda activate supply-chain-ml
```

### 3. Setup Pre-commit Hooks (Optional but Recommended)
To maintain the strict linting and Python standards before you commit:
```bash
pre-commit install
```

### 4. Fetch the Dataset
The dataset (DataCo Supply Chain) is massive and excluded from Git using `.gitignore`. Use the [Kaggle API](https://www.kaggle.com/docs/api) to retrieve it:

```bash
# Ensure your Kaggle API token is configured (~/.kaggle/kaggle.json)
kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis -p "data/raw" --unzip
```

### 5. Run the Test Suite
Ensure the 44 domain and property-based tests are passing on your machine to validate the Hexagonal Architecture:
```bash
pytest -v
```

## Project Structure

```text
supply-chain-optimization-ml/
├── domain/               # Core business rules (Product, Order entities)
├── adapters/             # External integration (CSV repos, Models, UI)
├── application/          # Service interconnects
├── tests/                # 44+ TDD Pytest & Hypothesis tests
├── notebooks/            # Jupyter Exploratory Data Analysis
├── data/                 # Ignored by git (raw, interim, processed data)
├── pyproject.toml        # Ruff, Black, Mypy configurations
└── environment.yml       # Reproducible Conda dependencies
```

## Project Phases

- ✅ **Phase 0:** Governance & Architecture (Hooks, Configs, Hexagonal Scaffold)
- ✅ **Phase 1:** Infrastructure Setup
- ✅ **Phase 2:** Proactive Risk Mitigation & Stakeholder Intelligence (EDA)
- ✅ **Phase 3:** Domain Implementation & TDD Audits (44 passing tests)
- 🔜 **Phase 4:** ML Pipeline & Streamlit UI

## Author & Contributors

- **Author:** Tirth Joshi
- **Role:** Lead Data Scientist & ML Engineer
- **Institution:** University of British Columbia (UBC MDS)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. This allows for open collaboration while maintaining creator attribution.
