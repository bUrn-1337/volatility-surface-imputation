# Implied Volatility Surface Imputation

**Competition:** Finance Club IIT Roorkee — Open Projects 2026  
**Task:** Predict 5460 missing implied volatility values across 28 NIFTY 50 option contracts  
**Metric:** Mean Squared Error (lower is better)  
**Best score:** 0.0000423 (v6)

---

## Problem

A 975-row × 30-column time series of NIFTY 50 option IVs, sampled at 5-minute intervals from 7–27 Jan 2026.
About 20% of option IV values are missing. The task is to fill them without any lookahead bias.

**The hard part:** 27 Jan is expiry day. IVs spike from ~0.1 to 5+ over the session as options approach expiration. Methods that work well on normal days catastrophically fail on expiry day.

---

## Dataset

| Field | Value |
|---|---|
| Rows | 975 (5-min bars, 07–27 Jan 2026) |
| Option columns | 28 (14 CE strikes 25200–26500, 14 PE strikes 23800–25100) |
| Missing values | 5460 (~20%) |
| Expiry | 27 Jan 2026 15:30 |
| Underlying | NIFTY 50 spot price |

---

## Final Approach: TTE-Conditional QP Ensemble (v6)

Five cross-sectional smile-fitting methods are blended with weights optimised separately for **near-expiry** (TTE < 5 trading days = all Jan 27 rows) and **normal** timestamps.

### Fill Methods

Each method takes the observed IVs at a single timestamp and fits/interpolates/extrapolates to fill missing strikes.

| Method | Description |
|---|---|
| `poly2` | Quadratic polynomial: IV ~ a + bK + cK² |
| `poly2_var` | Quadratic in variance space: IV² ~ a + bK + cK² |
| `PCHIP` | Monotone cubic spline (interior); linear extrapolation (exterior) |
| `adaptive` | PCHIP for interior missing strikes; poly2 for wing extrapolation |
| `logwing` | PCHIP interior; log-linear (geometric) wing extrapolation |

### `fill_logwing` — Why It Matters

OTM wing contracts (24000PE: 32% missing, 26400CE: 28% missing on Jan 27) are 2–3× more missing than ATM contracts. Polynomial extrapolation diverges for deep OTM strikes. Log-linear uses the local smile slope continued in log-IV space:

```
log IV(K) = log IV(K_boundary) + slope × (K − K_boundary)
slope clipped to ±0.02 per strike unit
```

### QP Weight Optimisation

For each TTE group, find non-negative weights w₁…w₅ (summing to 1) minimising MSE on a 10% artificial mask:

```
min  Σ (trueIV − Σ wₖ · fillₖ(IV))²
s.t. Σ wₖ = 1,  wₖ ≥ 0
```

Solved with SLSQP (scipy), 20 random Dirichlet restarts.

### Final Weights

| Method | Normal (TTE ≥ 1950 min) | Expiry (TTE < 1950 min) |
|---|---|---|
| `poly2` | 0.0% | 3.5% |
| `poly2_var` | **33.3%** | **41.3%** |
| `PCHIP` | 20.1% | 14.2% |
| `adaptive` | **46.6%** | 1.1% |
| `logwing` | 0.0% | **39.9%** |

**Interpretation:**
- Normal days: adaptive (PCHIP+poly2) + poly2_var dominate — these are the most accurate cross-sectional fits when IVs are stable (~0.1–0.2)
- Expiry day: poly2_var (fits the variance parabola) + logwing (geometric wing extrapolation) dominate — polynomial extrapolation breaks down when IVs span 0.1–5+

---

## What Didn't Work

| Approach | Simulated MSE | vs v6 | Reason |
|---|---|---|---|
| Temporal interpolation (adjacent bars) | 0.000703 | 18× worse | IV changes 0.5–1.0 units per 5-min bar on Jan 27 |
| Gap-gated temporal (gap ≤ 1, expiry only) | 0.00357 (expiry) | 8× worse | Same root cause |
| ATM-scaled temporal (IV(t,K) × ATM(t)/ATM(t±1)) | 0.0298 (expiry) | 70× worse | Same root cause |
| SVD matrix completion (rank 3, 5) | Fails | — | Expiry spike dominates all singular vectors |
| Log-space poly2 / log-space PCHIP | Worse | — | Poorly conditioned for normal-day IV range (0.1–0.2) |
| Full SVI (5-parameter Nelder-Mead) | 0.000433 | 11× worse | Optimizer gets stuck; only 15 restarts |
| poly3 / poly3_var | Worse | — | Overfitting with ~10 observations per timestamp |
| TTE-conditional Dirichlet sweep (v5) | 0.0000375 | +5% | Dirichlet random search misses optimum; QP is better |
| 7-method QP (+var_logwing, +logpoly2) | 0.0000354 sim / 0.0000478 actual | Worse actual | New methods overfit to 164 CV expiry positions; actual/sim ratio worsened |

---

## Cross-Validation & The Actual/Simulated Gap

**CV setup:** 10% of observed positions are randomly masked; MSE is computed on these ~2184 positions.

**Gap (v3):** Simulated 0.0000380, actual 0.000079 — ratio 2.08×  
**Gap (v6):** Simulated 0.0000356, actual 0.0000423 — ratio 1.19×

**Root cause:** Random masking underrepresents OTM wing positions. In the actual submission, 24000PE and 26400CE are 2–3× more missing than ATM contracts on expiry day (Jan 27). The wing bias drove the 2× gap in v3. Adding `logwing` to the ensemble improved wing handling and reduced the ratio to 1.19×.

---

## Submission History

| Version | Method | Simulated MSE | Actual MSE |
|---|---|---|---|
| v1 | Cubic spline, broken weight sweep | — | 0.000275 |
| v2 | poly2 + PCHIP blend, fixed weights | — | 0.0000927 |
| v3 | 4-way blend (p2+var+ph+adp), grid weights | 0.0000380 | 0.000079 |
| v4 | 7-method blend incl. temporal + SVD | 0.0000366 | — |
| v5 | TTE-conditional blend + SVI, Dirichlet search | 0.0000375 | — |
| **v6** | **5-method QP blend, TTE-conditional** | **0.0000356** | **0.0000423** |
| v7 | v6 + ATM-scaled (6th method) | 0.0000356 | — (no improvement) |
| v8 | 7-method QP + ATM-scaled | 0.0000354 | 0.0000478 (worse) |

---

## How to Run

```bash
# Execute the full pipeline (takes ~5 min; SVI benchmark in 4.5 takes ~2 min)
jupyter nbconvert --to notebook --execute --inplace iv_imputation.ipynb

# This generates:
#   filled_dataset.csv  — full IV surface with all gaps filled
#   submission.csv      — Kaggle submission (5460 rows: id||ticker, value)
```

### Notebook Structure

| Section | Cell | What it does |
|---|---|---|
| 1.1–1.3 | Load data, parse tickers, compute TTE | Setup |
| 2.1–2.4 | Missingness heatmap, smile plots, time-series | EDA |
| 3.1–3.3 | Long-format features, cross-sectional, lag | Feature engineering |
| 4.1–4.1c | Fill function definitions | Methods |
| 4.2 | 10% artificial masking → CV baseline | CV setup |
| 4.4–4.5 | SVI calibration + TTE breakdown benchmark | Research |
| 4.6 | TTE-conditional Dirichlet weight sweep | Research |
| 4.8–4.8e | Temporal test, logwing bench, QP weights, ATM-scaled | Research |
| 4.7 | Build all 5 fills on full dataset | Production |
| 5.1 | TTE-conditional QP blend → filled_dataset.csv | Submission |
| 5.2 | generate_solution() → submission.csv | Submission |

---

## File Structure

```
iv_imputation.ipynb    main pipeline and research notebook
dataset.csv            input: raw option IV data (975 × 30, ~20% missing)
filled_dataset.csv     output: complete IV surface
submission.csv         output: Kaggle submission (id||value, 5460 rows)
README.md              this file
```
