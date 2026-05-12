---
name: leakage-audit
description: Audit the codebase for data leakage — verify LEAKAGE_COLUMNS are excluded from all feature pipelines, no post-shipment data reaches models.
---

Audit the supply-chain-optimization-ml project for data leakage violations.

## What is data leakage in this project?

Three columns contain information unavailable at prediction time:
- `Days for shipping (real)` — only known after delivery completes
- `Delivery Status` — directly encodes the outcome (late/on-time)
- `shipping date (DateOrders)` — future information at order time

Using any of these as model features produces artificially high accuracy that won't replicate in production.

## Audit steps

### 1. Verify LEAKAGE_COLUMNS constant

```bash
grep -n "LEAKAGE_COLUMNS" adapters/data/csv_repository.py
```

Confirm it contains all three columns. If missing any — **critical violation**.

### 2. Scan all adapters for leakage column access

```bash
grep -rn "Days for shipping (real)\|Delivery Status\|shipping date" adapters/ application/ domain/
```

Any hit outside `LEAKAGE_COLUMNS` definition = potential leakage. Investigate each match.

### 3. Scan notebooks for leakage

```bash
grep -rn "Days for shipping (real)\|Delivery Status\|shipping date" notebooks/
```

EDA notebooks may reference these columns for analysis (OK). But if used in feature engineering cells — flag it.

### 4. Check ML adapters

For every file in `adapters/ml/`:
- If it builds a feature matrix, verify leakage columns are not present
- If it reads from CSV adapter, verify `drop_leakage=True` (default)
- If it reads from any OTHER data source, verify equivalent exclusion exists

### 5. Check evaluation metrics

```bash
grep -rn "accuracy_score\|accuracy" adapters/ application/ tests/
```

Flag any use of accuracy as primary metric without F1/precision/recall alongside. Class split is 55/45 — accuracy misleads.

### 6. Verify domain boundary

```bash
grep -rn "^import\|^from" domain/*.py | grep -v "typing\|dataclasses\|datetime\|enum\|collections\|__future__"
```

Any external import in domain/ = architecture violation that could introduce indirect leakage paths.

## Output format

```
## Leakage Audit — <date>

### LEAKAGE_COLUMNS constant
✅ All 3 columns present / ❌ Missing: <column>

### Adapter scan
✅ No leakage column access outside constant / ❌ <file>:<line> — <violation>

### Notebook scan
✅ No leakage in feature engineering / ⚠️ <notebook>: cell <N> references leakage column in feature context

### ML adapter scan
✅ All adapters exclude leakage / ❌ <file>:<line> — <violation>

### Evaluation metrics
✅ F1/precision/recall used / ❌ <file>:<line> — accuracy-only evaluation

### Domain boundary
✅ Zero external imports / ❌ <file>:<line> — <import>

### Verdict
✅ No leakage detected / ❌ <N> violations found — fix before proceeding
```
