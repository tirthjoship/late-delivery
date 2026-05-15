# Project Deep Dive — Supply Chain Late Delivery Risk Prediction

> A plain-English walkthrough of what this project does, why every decision was made, and how data flows from raw CSV to interactive dashboard. Written for interview prep and future-you when your brain isn't working.

---

## The One-Sentence Summary

**We predict which e-commerce orders will arrive late _before_ they ship, so a logistics team can reroute, prioritize, or notify customers proactively.**

---

## The Business Problem

The DataCo supply chain dataset has ~180,000 orders. **54.83% are delivered late.** That's not a small edge case — more than half of all orders miss their delivery window.

The late rates by shipping mode tell the real story:

| Shipping Mode | Late Rate | What This Means |
|---------------|-----------|-----------------|
| First Class | 95.3% | Almost always late — the "premium" option is broken |
| Second Class | 76.6% | Three out of four are late |
| Standard Class | 38.0% | Best performer, but still significant |
| Same Day | Lowest | Works as promised |

**The counterintuitive finding:** "First Class" sounds premium but has the worst late rate. This is the kind of insight that makes a recruiter pause and ask questions — which is exactly what you want.

---

## End-to-End Data Flow

This is how data moves through the entire system, from CSV file to risk prediction:

```mermaid
flowchart TD
    A[DataCo CSV\n180k orders, 53 columns] -->|csv_repository.py| B[Leakage Shield\nDrop 3 dangerous columns]
    B --> C[Domain Entities\nOrder, OrderItem, Product]
    C -->|extract_features| D[Raw Feature Dicts\n12 features per order]
    D -->|train_test_split| E{Split FIRST\n80% train / 20% test}
    E -->|Train set only| F[FeatureEncoder.fit\nLearn scaling + categories]
    E -->|Test set| G[FeatureEncoder.transform\nApply learned encoding]
    F --> H[Encoded Train Matrix\nOneHot + StandardScaler]
    G --> I[Encoded Test Matrix\nSame encoding, no leakage]
    H -->|XGBoost / LogReg| J[Train Model]
    J --> K[Predict on Test Set]
    K --> L[Evaluate\nF1, Precision, Recall, AUC-ROC]
    L --> M[SHAP Explain\nGlobal + Local feature importance]
    M --> N[MLflow Log\nParams, metrics, model artifacts]
    N --> O[Model Registry\nVersion + stage management]

    style B fill:#ff6b6b,color:#fff
    style E fill:#ffd93d,color:#333
    style F fill:#6bcb77,color:#fff
```

**The three red-flag steps** (where data leakage can sneak in):
1. **Leakage Shield** — Must drop columns before anything else
2. **Split FIRST** — If you encode before splitting, test data statistics leak into training
3. **Fit on Train ONLY** — Encoder learns means/categories from training data only

---

## Why Hexagonal Architecture?

Most ML projects look like this:

```mermaid
flowchart LR
    A[Jupyter Notebook] --> B[pandas everywhere]
    B --> C[sklearn everywhere]
    C --> D[Everything tangled together]
```

This project looks like this:

```mermaid
flowchart LR
    subgraph Adapters["Adapters (framework code)"]
        CSV[CSV Reader]
        XGB[XGBoost]
        SHAP[SHAP]
        MLF[MLflow]
        PLT[Plotly]
    end

    subgraph Domain["Domain (pure Python)"]
        MOD[Models\nOrder, Product]
        SVC[Services\nextract_features]
        PRT[Ports\ninterfaces]
    end

    subgraph App["Application (orchestration)"]
        UC[Use Cases\nTrain, Predict, Cluster]
    end

    CSV -->|implements| PRT
    XGB -->|implements| PRT
    SHAP -->|implements| PRT
    MLF -->|implements| PRT
    UC --> MOD
    UC --> SVC
    UC --> PRT
```

**Why this matters in an interview:**

| Question They Ask | Your Answer |
|-------------------|-------------|
| "What if you need to swap XGBoost for LightGBM?" | Write a new adapter. Domain untouched. |
| "What if data comes from a database instead of CSV?" | Write a new adapter. Domain untouched. |
| "How do you prevent data leakage?" | Leakage columns are blocked at the adapter boundary. Domain never sees them. |
| "How do you test without real data?" | Domain is pure Python. Tests use tiny fixtures. No CSV, no sklearn, no network. |

**The rule:** Dependencies point inward. Domain imports nothing external. Adapters implement domain Protocols. Application wires them together.

---

## The 12 Features We Use

These are extracted by `domain/services.py:extract_features()`:

| Feature | Type | Source | Why It's Safe |
|---------|------|--------|---------------|
| `shipping_mode` | Categorical | Order | Known at order time |
| `days_for_shipment_scheduled` | Numeric | Order | Scheduled, not actual |
| `order_month` | Numeric | Order date | When order was placed |
| `order_day_of_week` | Numeric | Order date | Day patterns matter |
| `order_region` | Categorical | Order | Geographic risk factor |
| `benefit_per_order` | Numeric | Order | Profit metrics |
| `sales_per_customer` | Numeric | Customer | Customer value tier |
| `order_profit_per_order` | Numeric | Order | Margin indicator |
| `item_count` | Numeric | Order items | Complexity proxy |
| `total_quantity` | Numeric | Order items | Volume proxy |
| `total_discount` | Numeric | Order items | Discount aggressiveness |
| `avg_unit_price` | Numeric | Order items | Product tier |

### The 3 Features We NEVER Use (Leakage)

| Banned Column | Why It's Dangerous |
|---------------|-------------------|
| `Days for shipping (real)` | Only known AFTER delivery. Using it = telling the model the answer. |
| `Delivery Status` | Literally IS the target variable in different words. |
| `shipping date (DateOrders)` | Future info — we can't know the ship date at order time. |

---

## Model Comparison — What We Tried and Why

```mermaid
flowchart LR
    subgraph Baseline["Baseline"]
        LR[Logistic Regression\nF1: 0.6531]
    end

    subgraph Production["Production Candidates"]
        XG1[XGBoost depth=6\nF1: 0.6543]
        XG2[XGBoost depth=3\nF1: 0.6532]
        XG3[XGBoost depth=8\nF1: 0.6543]
    end

    LR -.->|"Marginal improvement"| XG1
    XG2 -.->|"Underfitting"| XG1
    XG3 -.->|"Same as default"| XG1
```

**The honest truth about these numbers:**

All models cluster around F1 ~0.65. XGBoost barely beats LogReg. This is not a model problem — **it's a signal problem.** Shipping mode dominates so heavily (95.3% late for First Class) that the model is essentially learning a lookup table.

**Why we use F1 instead of accuracy:**
- Class split is 55/45 (late vs on-time)
- A model that predicts "always late" gets 55% accuracy — sounds decent, is useless
- F1 balances precision (don't cry wolf) and recall (don't miss real late orders)

---

## SHAP Explainability — Why Each Order Is Flagged

```mermaid
flowchart TD
    subgraph Global["Global Importance (across all orders)"]
        G1["shipping_mode_Standard Class → 0.616"]
        G2["shipping_mode_First Class → 0.432"]
        G3["shipping_mode_Second Class → 0.134"]
        G4["shipping_mode_Same Day → 0.075"]
        G5["sales_per_customer → 0.054"]
    end

    subgraph Local["Local Explanation (one order)"]
        L1["This order uses First Class → +0.43 risk"]
        L2["High sales per customer → -0.02 risk"]
        L3["Scheduled 4 days → +0.01 risk"]
    end

    Global --> |"TreeExplainer"| Local
```

**Why SHAP over feature importance:**
- Random Forest `feature_importances_` shows what features are USED but not HOW
- SHAP shows DIRECTION: does Standard Class push risk UP or DOWN?
- SHAP is additive: local explanations sum to the prediction
- Recruiters ask "how do you explain your model?" — SHAP is the gold standard answer

---

## Experiment Tracking with MLflow

```mermaid
flowchart LR
    subgraph Training["Training Pipeline"]
        T1[Start Run] --> T2[Log Params\nhyperparameters]
        T2 --> T3[Log Metrics\nF1, AUC, etc.]
        T3 --> T4[Log Artifacts\nmodel.pkl, encoder.joblib]
        T4 --> T5[Register Model\nversion + stage]
        T5 --> T6[End Run]
    end

    subgraph Registry["Model Registry"]
        R1[v1: LogReg baseline]
        R2[v2: XGBoost depth=6]
        R3[v3: XGBoost depth=3]
        R4[v4: XGBoost depth=8]
    end

    T5 --> Registry
```

**What's logged per run:**
- `hp_*` prefixed hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- Metrics: F1, precision, recall, AUC-ROC
- Artifacts: trained model + fitted encoder (for serving)
- Model Registry entry with version number

**Storage:** Local SQLite (`mlflow.db`) — not a remote server. Good for portfolio, easy to demo.

---

## K-Means Risk Cohorts — Unsupervised Segmentation

Beyond prediction, we cluster orders into risk cohorts to find natural groupings:

```mermaid
flowchart TD
    A[All Orders] -->|Extract 10 numeric features\n+ shipping_mode one-hot| B[Feature Matrix]
    B -->|Exclude order_region| C[Clustering Input]
    C -->|K=2..10| D[Elbow + Silhouette Analysis]
    D -->|Pick optimal K| E[K-Means Fit]
    E --> F[Assign Labels]
    F --> G[Profile Each Cluster]

    subgraph Profiling["Post-hoc Profiling"]
        G --> H[Late Rate per cluster]
        G --> I[Dominant shipping mode]
        G --> J[Region distribution]
        G --> K[Auto-label:\nHigh/Medium/Low Risk]
    end
```

**Key design choice:** `order_region` is excluded from clustering INPUT but included in cluster PROFILING. Why? Region adds noise to clustering (too many categories) but is useful for understanding what each cluster looks like.

---

## The Streamlit Dashboard — 4 Tabs

```mermaid
flowchart LR
    subgraph Tab1["Predict & Explain"]
        P1[Enter order details] --> P2[Risk score gauge]
        P2 --> P3[SHAP waterfall chart]
    end

    subgraph Tab2["Model Performance"]
        M1[LogReg vs XGBoost metrics] --> M2[SHAP importance bars]
    end

    subgraph Tab3["Risk Cohorts"]
        C1[PCA scatter plot] --> C2[Cohort profile cards]
    end

    subgraph Tab4["Dataset"]
        D1[Key statistics] --> D2[Shipping mode + region charts]
    end
```

Models train on the 1000-row sample at startup (~3 seconds, cached with `@st.cache_resource`).

---

## Architecture Decision Record

| Decision | Choice | Why | Alternative Considered |
|----------|--------|-----|----------------------|
| Architecture | Hexagonal (ports & adapters) | Swappable adapters, testable domain, interview talking point | Flat scripts, layered MVC |
| Primary metric | F1 score | 55/45 split makes accuracy misleading | AUC-ROC (good but doesn't capture threshold trade-off) |
| Explainability | SHAP | Additive, local + global, theoretically grounded | Feature importance, LIME |
| Experiment tracking | MLflow local (SQLite) | Industry standard, Model Registry, free | Weights & Biases (better UI but adds dependency) |
| Feature encoding | Split-before-encode | Prevents preprocessing leakage | Encode-then-split (common but wrong) |
| Clustering | K-Means with silhouette | Simple, interpretable, good for portfolio | DBSCAN (no K needed but less explainable) |
| Dashboard | Streamlit | Python-native, fast to build, free hosting | Dash (more control but more code) |
| Testing | pytest + Hypothesis | Property-based tests catch edge cases | unittest (verbose, less powerful) |
| Type checking | mypy strict | Catches bugs before runtime, shows rigor | pyright (faster but less standard) |
| Logging | loguru | One-line setup, structured output | stdlib logging (verbose config) |
| Data format | DataCo CSV (Kaggle) | Public, realistic, 180k rows, multi-domain | Synthetic data (less credible) |
| Leakage protection | Constant + adapter shield | Impossible to accidentally use banned columns | Documentation only (humans forget) |

---

## What the Code Structure Maps To

```
supply-chain-optimization-ml/
│
├── domain/                     ← "The WHAT" — business rules
│   ├── models.py               ← Data shapes (Order, Product, MetricsResult)
│   ├── ports.py                ← Interfaces (what adapters must implement)
│   ├── services.py             ← Logic (extract_features, risk classification)
│   └── exceptions.py           ← Domain errors (CSVValidationError)
│
├── adapters/                   ← "The HOW" — framework glue
│   ├── data/csv_repository.py  ← Reads CSV, builds domain entities, blocks leakage
│   ├── ml/feature_encoder.py   ← OneHot + StandardScaler encoding
│   ├── ml/sklearn_predictor.py ← LogReg + XGBoost wrappers
│   ├── ml/evaluation.py        ← F1/precision/recall/AUC computation
│   ├── ml/shap_explainer.py    ← SHAP TreeExplainer + LinearExplainer
│   ├── ml/mlflow_tracker.py    ← MLflow logging + Model Registry
│   └── ml/kmeans_clusterer.py  ← K-Means with silhouette scoring
│
├── application/                ← "The WHEN" — orchestration
│   └── use_cases.py            ← TrainAndEvaluate, PredictSingleOrder,
│                                  FitClusters, ProfileClusters
│
├── app/                        ← "The FACE" — user interface
│   ├── streamlit_app.py        ← 4-tab dashboard entry point
│   └── components/             ← Tab implementations
│
├── tests/                      ← 154 tests, 92% coverage
├── scripts/train.py            ← CLI training entry point
├── notebooks/                  ← EDA + training narrative
└── data/                       ← Raw (gitignored) + sample (committed)
```

---

## The Quality Stack

```mermaid
flowchart TD
    subgraph Dev["Developer Writes Code"]
        A[Edit file]
    end

    subgraph PreCommit["Pre-Commit Hooks (local)"]
        B[black — formatting]
        C[isort — import order]
        D[mypy strict — type safety]
        E[ruff — linting]
        F[gitleaks — secret scanning]
    end

    subgraph CI["GitHub Actions (remote)"]
        G[Lint workflow]
        H[Test Suite — 154 tests, 90% gate]
        I[mypy strict]
        J[Secret scanning]
    end

    A --> PreCommit --> |"commit blocked\nif any fail"| CI
    CI --> |"PR blocked\nif any fail"| K[Merge to dev]
```

---

## Common Interview Questions and Answers

### "Why not just use accuracy?"

Class split is 55% late / 45% on-time. A model that always predicts "late" gets 55% accuracy. Sounds decent, is completely useless — it can't distinguish anything. F1 forces the model to be both precise (when it says late, it's right) and sensitive (it catches most actual late orders).

### "How do you prevent data leakage?"

Three layers:
1. `LEAKAGE_COLUMNS` constant in the CSV adapter — physically drops the columns
2. Split-before-encode — encoder never sees test data during fit
3. Property-based tests (Hypothesis) — verify leakage column names never appear in feature output

### "Why hexagonal architecture for an ML project?"

Two reasons. First, it makes the domain logic testable without loading data or importing sklearn — tests run in milliseconds. Second, it makes the project extensible: swapping XGBoost for LightGBM means writing one new 30-line adapter file. The domain, use cases, and tests don't change.

### "What's the business impact?"

If a logistics team acts on the model's predictions:
- Flag high-risk orders for rerouting or priority handling
- Proactively notify customers about potential delays
- Focus intervention on First Class and Second Class shipments (95% and 77% late rates)
- Even a 10% reduction in late deliveries at scale means fewer refunds, better NPS, lower churn

### "Isn't XGBoost barely better than LogReg here?"

Yes — and that's an honest finding, not a failure. The signal is dominated by shipping mode. More complex models can't extract much more because the underlying problem is operational (First Class shipping is broken), not a modeling challenge. This is a strength — knowing when model complexity won't help is a senior-level insight.

### "What would you do differently in production?"

1. **Feature store** — serve pre-computed features with point-in-time correctness
2. **A/B testing** — measure if interventions on flagged orders actually reduce late rate
3. **Monitoring** — track prediction drift as shipping patterns change seasonally
4. **Real-time scoring** — API endpoint (FastAPI adapter) instead of batch CSV
5. **Feedback loop** — retrain periodically as new delivery data arrives

---

## Quick Reference Card

| Aspect | Answer |
|--------|--------|
| **Dataset** | DataCo Supply Chain, 180k orders, Kaggle |
| **Problem** | Binary classification: will this order be late? |
| **Target** | `Late_delivery_risk` (0/1) |
| **Class split** | 54.83% late / 45.17% on-time |
| **Primary metric** | F1 score |
| **Best model** | XGBoost (depth=6), F1=0.6543 |
| **Top feature** | `shipping_mode` (SHAP importance 0.616) |
| **Leakage protection** | 3 columns banned, split-before-encode |
| **Architecture** | Hexagonal (ports & adapters) |
| **Tests** | 154, 92% coverage, property-based |
| **Tracking** | MLflow local (SQLite) |
| **Dashboard** | Streamlit, 4 tabs |
| **Tech stack** | Python 3.12, XGBoost, sklearn, SHAP, MLflow, Streamlit, pytest, Hypothesis |
