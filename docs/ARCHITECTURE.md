# Architecture & Design Decisions

> How this project works and why every choice was made.

## Data Flow

```mermaid
flowchart LR
    subgraph Input["📦 Data In"]
        CSV[DataCo CSV\n180K orders]
    end

    subgraph Safety["🛡️ Leakage Shield"]
        DROP[Drop 3 post-shipment\ncolumns automatically]
    end

    subgraph Pipeline["🔄 ML Pipeline"]
        FEAT[Extract 12\npre-shipment features]
        SPLIT[Split FIRST\nthen encode]
        TRAIN[Train XGBoost\n+ LogReg]
    end

    subgraph Evaluate["📊 Evaluate & Explain"]
        EVAL[F1 / Precision\nRecall / AUC-ROC]
        SHAP[SHAP\nexplainability]
        MLF[MLflow\ntrack & version]
    end

    subgraph Output["🎯 Outputs"]
        DASH[Streamlit\nDashboard]
        REG[Model\nRegistry]
    end

    CSV --> DROP --> FEAT --> SPLIT --> TRAIN
    TRAIN --> EVAL --> MLF --> REG
    TRAIN --> SHAP --> DASH
    EVAL --> DASH
```

## Architecture Pattern

**Hexagonal (Ports & Adapters)** — dependencies point inward only.

```
adapters/     →  domain/  ←  application/
(frameworks)     (pure)      (orchestration)
```

- `domain/` imports ONLY Python stdlib. Zero sklearn, pandas, or numpy.
- Adapters implement domain Protocol interfaces.
- Swap XGBoost for LightGBM? One new adapter. Domain unchanged.

## Key Design Decisions

| Decision | Choice | Why | Alternative Considered |
|----------|--------|-----|----------------------|
| Architecture | Hexagonal | Swappable adapters, testable domain | Flat scripts |
| Primary metric | F1 score | 55/45 split makes accuracy misleading | AUC-ROC |
| Explainability | SHAP | Additive, local + global, theoretically grounded | LIME |
| Tracking | MLflow (SQLite) | Industry standard, Model Registry | W&B |
| Encoding | Split-before-encode | Prevents preprocessing leakage | Encode-then-split |
| Clustering | K-Means + silhouette | Interpretable, good for segmentation | DBSCAN |
| Dashboard | Streamlit | Python-native, free hosting | Dash |
| Testing | pytest + Hypothesis | Property-based catches edge cases | unittest |
| Data source toggle | Strategy C hybrid | Full metrics + live sample predictions | All-sample or all-full |

## Leakage Protection (3 Layers)

| Layer | What It Does |
|-------|-------------|
| `LEAKAGE_COLUMNS` constant | Physically drops `Days for shipping (real)`, `Delivery Status`, `shipping date` at CSV adapter |
| Split-before-encode | Encoder never sees test data during fit |
| Property-based tests | Hypothesis verifies leakage column names never appear in feature output |

## Dashboard Architecture (Strategy C)

```mermaid
flowchart TD
    subgraph Static["Pre-computed (instant load)"]
        FM[full_run_metrics.json\nF1, AUC, confusion matrices]
        FS[full_dataset_stats.json\nLate rates by mode/region]
        FP[full_pca_clusters.json\n2000-point PCA subsample]
    end

    subgraph Live["Sample-trained (cached ~5s)"]
        MODEL[XGBoost + LogReg\non 1K sample]
        SHAP2[SHAP explainers]
    end

    subgraph Toggle["Sidebar Toggle"]
        FULL[Full Dataset 180K]
        SAMPLE[Sample 1K]
    end

    FULL --> FM & FS & FP
    SAMPLE --> MODEL & SHAP2
    MODEL --> PREDICT[Risk Predictor\nalways live]
```

**Risk Predictor** always uses live sample model for interactivity. Stats tabs switch between pre-computed full metrics and live sample metrics via sidebar toggle.
