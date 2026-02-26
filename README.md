# Supply Chain Optimization ML

An end-to-end Machine Learning project focusing on **Demand Forecasting, Inventory Optimization (ABC Analysis), and Late Delivery Risk Prediction**. Built using robust **Hexagonal Architecture** and strictly driven by **Test-Driven Development (TDD)** principles.

This repository is designed to demonstrate production-grade ML engineering, specifically aligned with the operational complexities of major retailers and supply chain networks (e.g., Amazon, Walmart, Costco).

## 🚀 Key Highlights

- **Aesthetic & Structural Governance:** Built using Python 3.12 with absolute strict typing (`mypy --strict`), auto-formatting (`black`, `isort`), and linting (`ruff`), all enforced by pre-commit hooks.
- **Hexagonal Architecture:** The core ML business logic (`domain/`) is entirely decoupled from external infrastructure (csv files, databases, UI). Features zero data-leakage and extremely high maintainability.
- **TDD First:** Includes a comprehensive suite of 44 tests using `pytest`, integrated with `Hypothesis` for robust property-based testing and edge-case validation to ensure audit-readiness.
- **Deep EDA:** Analyzed a 180,000+ record supply chain dataset (`DataCo`) identifying a massive 54.83% baseline late delivery rate, driving a pivot from basic forecasting directly to proactive risk management.

## 🛠️ Technology Stack

- **Core:** Python 3.12
- **Testing:** Pytest, Hypothesis (Property-based testing)
- **Data & Modeling:** Pandas, NumPy, XGBoost, Scikit-Learn
- **Experiment Tracking:** MLflow
- **Architecture:** Hexagonal (Ports & Adapters)

## 📦 How to Reproduce the Environment

To ensure perfect reproducibility, this project uses a strict `conda` environment. 

### 1. Clone the Repository
```bash
git clone git@github.com:tirthjoship/supply-chain-optimization-ml.git
cd supply-chain-optimization-ml
```

### 2. Create and Activate the Conda Environment
The `environment.yml` handles all core requirements:
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

## 📂 Project Structure

```text
supply-chain-optimization-ml/
├── domain/               # Core business rules (Product, Order entities)
├── adapters/             # External integration (CSV repos, Models, UI)
├── application/          # Service interconnects
├── tests/                # 44+ TDD Pytest & Hypothesis tests
├── notebooks/            # Jupyter Exploratory Data Analysis
├── data/                 # Ignored by git (raw, interim, processed data)
├── pyproject.toml        # Ruff, Black, Mypy configurations
├── environment.yml       # Reproducible Conda dependencies
```

## 🔄 Project Phases

- ✅ **Phase 0:** Governance & Architecture (Hooks, Configs, Hexagonal Scaffold)
- ✅ **Phase 1:** Infrastructure Setup
- ✅ **Phase 2:** Proactive Risk Mitigation & Stakeholder Intelligence (EDA)
- ✅ **Phase 3:** Domain Implementation & TDD Audits (44 passing tests)
- 🔜 **Phase 4:** XGBoost Pipeline, SHAP Explainability & Streamlit UI (In Progress)
