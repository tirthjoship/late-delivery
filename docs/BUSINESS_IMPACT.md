# Business Impact (Illustrative)

> **Read this first.** The DataCo dataset ships **no cost data** and is tutorial-grade (54.82% late rate).
> Every dollar figure below is **illustrative** — built from a transparent formula with **labeled assumptions** you
> can change. The *model performance* numbers are real and traceable
> ([`reports/model_comparison.json`](../reports/model_comparison.json)); the *cost* numbers are scenario inputs, not measured outcomes.
> **Do not put any dollar figure on a résumé unless it is computed here with stated assumptions.**

---

## 1. Assumptions (edit these)

| Symbol | Meaning | Value used | Source / status |
|--------|---------|-----------|-----------------|
| `N` | Orders evaluated per period | 10,000 | Per-10k basis for readability (dataset has 65,752 orders) |
| `late_rate` | Base late-delivery rate | **54.82%** | Measured — `data/metrics/full_dataset_stats.json` |
| `R` | Model recall (default operating point) | **0.5825** | Measured — XGBoost temporal, `model_comparison.json` |
| `P` | Model precision (default operating point) | **0.7743** | Measured — XGBoost temporal, `model_comparison.json` |
| `C_late` | Cost of an unmitigated late delivery (service + refund/reship + churn) | **$15** | **Illustrative** — industry range $5–$25; midpoint chosen |
| `r` | Fraction of `C_late` recovered when a late order is caught early | **40%** | **Illustrative assumption** — proactive reroute/notify rarely recovers 100% |
| `C_int` | Cost of one proactive intervention (ops time, expedite) | **$3** | **Illustrative assumption** |

> Change any value and the conclusion may flip. The point of this doc is the **formula and the break-even**, not a headline number.

---

## 2. Order economics at the model's default operating point

Per **N = 10,000** orders, using the measured rates:

| Quantity | Formula | Value |
|----------|---------|-------|
| Actual late orders | `N × late_rate` | 5,482 |
| Late orders **caught** (true positives) | `late × R` | 3,193 |
| Late orders **missed** (false negatives) | `late − caught` | 2,289 |
| Orders **flagged** for intervention | `caught / P` | 4,124 |
| **False alarms** (false positives) | `flagged − caught` | 931 |

So the model asks ops to act on ~4,124 of every 10,000 orders, correctly catches ~58% of late orders, and ~23% of its flags are false alarms.

---

## 3. Net-value formula

```
Net value = (caught × C_late × r)  −  (flagged × C_int)
          = value recovered on caught late orders  −  cost of acting on all flags
```

**Worked example (assumptions above):**

```
Value recovered = 3,193 × $15 × 0.40   = $19,158
Intervention cost = 4,124 × $3          = $12,372
Net value (per 10,000 orders)           ≈ $6,786
```

Roughly **$0.68 of net value per order screened** — *under these illustrative assumptions*. The figure is positive but modest, which is consistent with the project's core finding: the signal is real but shallow.

---

## 4. Sensitivity — the answer depends entirely on `C_late` and `r`

Net value per 10,000 orders (`C_int` = $3 fixed):

| `C_late` ↓ / `r` → | r = 20% | r = 40% | r = 60% |
|--------------------|--------:|--------:|--------:|
| **$5**  | −$9,179 | −$5,986 | −$2,793 |
| **$15** | −$2,793 | **+$6,786** | +$16,365 |
| **$25** | +$3,593 | +$19,558 | +$35,523 |

**Break-even** (net value = 0) is `caught × C_late × r = flagged × C_int`, i.e. `C_late × r = 12,372 / 3,193 ≈ $3.87`. Below that the program loses money; above it, it pays. (E.g. at `C_late` = $15 you need only `r ≳ 26%`; at `C_late` = $5 you need `r ≳ 77%`.) This break-even is the number to defend in an interview, not a fixed ROI.

---

## 5. Rule-based alternative (the recommended design)

The model is **not** the cheapest way to capture most of the value. Late rate is near-deterministic by shipping mode (`full_dataset_stats.json`):

| Mode | Late rate | Rule | Effective precision of "flag all" |
|------|-----------|------|-----------------------------------|
| First Class | 95.3% | Always flag | 0.953 |
| Second Class | 76.7% | Always flag | 0.767 |
| Standard Class | 38.1% | **Model scores this tier** | model adds value here |
| Same Day | 46.2% | Flag, monitor | 0.462 |

A **flag-all rule on First + Second Class** captures their late orders at higher precision (0.767–0.953) than the model's blended 0.7743 — with zero model maintenance, full auditability, and no calibration risk. The model then earns its keep only on the **Standard Class** tier, where lateness is genuinely uncertain.

> Exact rule-vs-model *recall* depends on the volume share of each mode (visible in the dashboard's full-dataset view); the late-rate-by-mode figures above are measured.

---

## 6. Bottom line

- The model produces **modest, assumption-dependent** net value; it is not a slam-dunk ROI story, and the project does not pretend otherwise.
- The **defensible recommendation** is a shipping-mode rule for the high-risk tiers plus a model on the ambiguous Standard Class tier — cheaper, transparent, and nearly as effective.
- The portfolio value is the **decision discipline**: measuring lift honestly, pricing the cost asymmetry (threshold 0.35, FN ≈ 3× FP), and recommending the simpler tool when the data says so.

*See [`LIMITATIONS_AND_DECISIONS.md`](./LIMITATIONS_AND_DECISIONS.md) for the modelling rationale and [`reports/model_comparison.json`](../reports/model_comparison.json) for frozen metrics.*
