# Future Enhancements — Decision-Science Horizon

**Date:** 2026-06-11
**Status:** Roadmap — optional depth beyond the shipped model and docs.
**Scope guard:** This roadmap is deliberately scoped so this project and the companion
experimentation project never overlap. The two own different problems and methods:

| | This project (supply-chain-ml) | product-experimentation-analytics |
|---|---|---|
| Problem space | Operations / logistics / freight | Customer / payments / product levers |
| Core method | Predictive ML + cost-based operating decisions + MLOps | Controlled experiments + causal inference + PM judgment |
| Decision artifact | Operating policy (threshold, carrier rule) | Experiment readout / decision memo |

**Hard rule:** no causal-inference or uplift work in this repo (no "would upgrading shipping
class X→Y *cause* fewer lates?"). Causal methods are the other project's identity; duplicating
them dilutes both. Likewise, the experimentation project stays out of freight/logistics.

---

## E1 — Cost-sensitive thresholding (highest value, do first)

**Problem it fixes:** the honest "XGBoost F1 0.654 ≈ LogReg 0.653" conclusion currently reads as
"ML didn't help." Cost-sensitive operating-point selection turns the same frozen model into a
business answer: *which threshold minimizes expected cost?*

- Define `cost_fp` (needless expediting/review of an on-time order) and `cost_fn` (late order
  reaches customer) as **labeled assumptions** in `docs/BUSINESS_IMPACT.md` — never invented as
  facts; show the conclusion across a cost-ratio grid (e.g. 1:2 → 1:10) so no single guess is
  load-bearing.
- Sweep thresholds on the frozen validation predictions (`reports/model_comparison.json` run);
  pick `argmin(FP×cost_fp + FN×cost_fn)`; compare against the rule-based baseline
  (First/Second Class flag) under the same cost model.
- Artifact: `reports/cost_threshold_analysis.{md,json}` + expected-cost-vs-threshold curve.
- Interview line: "max-F1 is the wrong objective; I picked the operating point that minimizes
  expected cost, and showed when the simple rule is still cheaper."

## E2 — Probability calibration (prerequisite for E1's curve being trustworthy)

Reliability curve + Brier score for LogReg and XGBoost; isotonic/Platt recalibration if needed.
Expected-cost math in E1 assumes predicted probabilities mean something — measure that first.
Artifact: calibration plot + short note in `docs/` (extends the P1 calibration item).

## E3 — Drift monitoring & retraining trigger (MLOps credential)

Design, don't over-build: PSI/KS on the top SHAP features + label-rate tracking, alert
thresholds, and a written retraining policy (what drift level triggers what action). Simulate by
splitting DataCo by time and showing the monitor firing on a known shift. Artifact:
`docs/DRIFT_MONITORING.md` + a small `monitoring/` module with fixture-tested checks.
No counterpart exists in any other portfolio project.

## E4 — Carrier-assignment optimizer (finally justifies "optimization" in the repo name)

Prescriptive layer on top of the predictor: assign shipping mode per order to minimize total
shipping cost subject to a late-risk budget (predicted P(late) ≤ threshold per segment).
Linear programming (PuLP/OR-Tools) or a transparent greedy heuristic — compare both.
Freight/logistics optimization lives HERE, which is exactly why the experimentation project
dropped its free-shipping framing. Artifact: `reports/carrier_optimization.md` with cost-vs-risk
frontier.

## Explicitly rejected / deferred

| Item | Why |
|---|---|
| Uplift modeling / causal what-ifs | Portfolio-separation hard rule above |
| Demand forecasting | Deferred — DataCo too weak for time series |
| New dataset | Rejected — polish the narrative instead |
| Deep models / AutoML sweep | Shipping-mode dominance means ceiling is data-limited; more model
  won't move it, and the honest-conclusion narrative is the asset |

## Suggested order

E2 → E1 (calibration feeds cost curve) → E3 → E4. Each is its own spec → plan → PR cycle;
none requires touching the frozen model comparison.
