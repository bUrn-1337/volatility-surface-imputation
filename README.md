# Implied Volatility Surface Imputation

**Competition:** Finance Club IIT Roorkee — Open Projects 2026  
**Task:** Predict 5460 missing implied volatility values across 28 NIFTY 50 option contracts  
**Metric:** Mean Squared Error (lower is better)  
**Best score:** 0.0000388978 (v11)

---

## Problem

A 975-row × 30-column time series of NIFTY 50 option IVs, sampled at 5-minute intervals from 7–27 Jan 2026.
About 20% of option IV values are missing. The task is to fill them without any lookahead bias.

**The hard part:** 27 Jan is expiry day. IVs spike from ~0.1 to 5+ over the session as options approach expiration. Methods that work well on normal days fail on expiry day, and methods that handle expiry day must treat the final trading hour differently from the morning session.

**Jan 26 is Republic Day (NSE holiday)**, so the dataset jumps directly from Jan 23 (TTE ~5765 min) to Jan 27 (TTE 5–375 min) — no intermediate TTE data exists.

---

## Dataset

| Field | Value |
|---|---|
| Rows | 975 (5-min bars, 07–27 Jan 2026, 13 trading days) |
| Option columns | 28 (14 CE strikes 25200–26500, 14 PE strikes 23800–25100) |
| Missing values | 5460 (~20%) |
| Expiry | 27 Jan 2026 15:30 |
| Underlying | NIFTY 50 spot price |

---

## Final Approach: 3-Bucket TTE QP Ensemble with Stratified CV (v11)

Six cross-sectional smile-fitting methods are blended with QP-optimised weights. Two key improvements over a naive blend:

1. **Stratified CV** — mask each contract at its actual missing rate instead of a flat 10% random mask, so the CV distribution matches the real test distribution.
2. **3 TTE buckets** — separate weights for normal days, expiry-day main session, and the final trading hour (where IV dynamics are completely different).

### Fill Methods

Each method takes the observed IVs at a single timestamp and fits/interpolates/extrapolates across strikes. No temporal (cross-timestamp) information is used — all predictions are cross-sectional, preventing lookahead bias.

| Method | Description |
|---|---|
| `poly2` | Quadratic polynomial: IV ~ a + bK + cK² |
| `poly2_var` | Quadratic in variance space: IV² ~ a + bK + cK² |
| `PCHIP` | Monotone cubic spline with extrapolation |
| `adaptive` | PCHIP for interior missing strikes; poly2 for wing extrapolation |
| `logwing` | PCHIP interior; log-linear (geometric) wing extrapolation |
| `totvar` | PCHIP on total variance w = IV²×TTE (see below) |

### `fill_totvar` — Key Innovation

In the final trading hour of expiry day, ATM IV can exceed 3–5. Polynomial and spline methods become unstable at these levels. The insight: **total implied variance w = IV²×TTE stays bounded** as TTE → 0 even as IV → ∞. Fitting PCHIP in w-space gives stable smile interpolation and extrapolation even when absolute IVs are extreme.

```
w(K) = IV(K)² × TTE_years
PCHIP fit on {K_obs → w_obs}
IV_pred(K) = sqrt(w_pred(K) / TTE_years)
```

This is the non-parametric version of the same idea behind the SVI model.

### Stratified CV

Instead of masking a flat 10% of positions at random, each contract is masked at its actual missing rate (clipped to 5%–50%). This means OTM wing contracts like 24000PE (~22% missing) get 2× more masked positions in CV, making the QP weights calibrated for the real test distribution rather than an easier random mask.

### 3 TTE Buckets

| Bucket | TTE range | What it covers |
|---|---|---|
| Normal | TTE > 1950 min | Jan 7–23 (all normal trading days) |
| Mid-expiry | 60 < TTE ≤ 1950 min | Jan 27 full session before final hour |
| Final-stretch | TTE ≤ 60 min | Jan 27 last trading hour (14:30–15:25) |

### Final Weights

| Method | Normal | Mid-expiry | Final-stretch |
|---|---|---|---|
| `poly2` | 0.0% | 0.0% | **65.0%** |
| `poly2_var` | **33.7%** | 4.8% | 5.7% |
| `PCHIP` | 7.9% | **28.9%** | 5.6% |
| `adaptive` | **31.7%** | **32.1%** | 0.0% |
| `logwing` | **23.4%** | **24.3%** | 0.0% |
| `totvar` | 3.3% | 9.9% | **23.8%** |

**Key observations:**
- Final-stretch: poly2 dominates (65%) — in the last hour, log-linear and variance-space methods diverge; simple polynomial is most stable. totvar gets 24% as a bounded complement.
- logwing gets 0% in the final-stretch because its log-linear extrapolation is catastrophically bad when IV is 3–5+ (MSE 20× worse than poly2 in that bucket).
- Normal days: poly2_var + adaptive + logwing form a stable blend for OTM wing interpolation.

### QP Weight Optimisation

For each TTE bucket, non-negative weights w₁…w₆ (summing to 1) are found by SLSQP with 25 random Dirichlet restarts:

```
min  Σ (trueIV − Σ wₖ · fillₖ(IV))²
s.t. Σ wₖ = 1,  wₖ ≥ 0
```

---

## What Didn't Work

| Approach | Result | Reason |
|---|---|---|
| Temporal interpolation (adjacent bars) | 18× worse | IV changes 0.5–1.0 units per 5-min bar on Jan 27 |
| ATM-scaled temporal | 70× worse for expiry | Same root cause |
| SVD matrix completion (rank 3, 5) | Fails | Expiry spike dominates all singular vectors |
| Full SVI (5-parameter, Nelder-Mead) | 11× worse | Optimizer gets stuck |
| Full SVI (5-parameter, L-BFGS-B) | 4.7× worse | Better optimizer helps but narrow strike range ≈ poly2 |
| LightGBM with EWMA features | 3× worse actual | EWMA features are stale for heavily-missing OTM contracts at test time |
| 7-method QP (v8) | Worse actual | New methods overfit CV; actual/simulated ratio worsened |
| 3-seed averaged weights (v13) | 0.0000397 | Slightly worse — seed 42 weights were better calibrated |

---

## Cross-Validation Strategy

**Why stratified CV?** A flat random 10% mask underrepresents OTM wing contracts (24000PE, 26400CE) that are ~22% missing in the actual test. The random CV gave simulated/actual ratios of 1.30–2.08× across versions. Stratified CV (mask each contract at its actual rate) produced a CV distribution much closer to the real test, making QP weights that generalise better.

| Version | CV setup | Actual score | Notes |
|---|---|---|---|
| v6 | Random 10% mask | 0.0000465 | Baseline |
| v10 | Stratified CV (actual rates) | 0.0000410 | +12% from stratification alone |
| v11 | Stratified + totvar + 3 buckets | **0.0000389** | +5% from new method and finer TTE split |

---

## Submission History

| Version | Method | Actual MSE |
|---|---|---|
| v1 | Cubic spline | 0.000275 |
| v2 | poly2 + PCHIP blend | 0.0000927 |
| v3 | 4-way blend, grid weights | 0.000079 |
| v6 | 5-method QP, random CV | 0.0000465 |
| v8 | 7-method QP | 0.0000478 (worse) |
| v10 | 5-method QP, stratified CV | 0.0000410 |
| **v11** | **6-method QP, stratified CV, 3 TTE buckets** | **0.0000389** |
| v13 | v11 + 3-seed weight averaging | 0.0000397 (worse) |

---

## How to Run

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 iv_imputation.ipynb
```

Produces `filled_dataset.csv` and `submission.csv`. Runtime ~8 minutes (dominated by 6 full-dataset builds).

### Notebook Structure

| Section | What it does |
|---|---|
| 0. Imports & Config | Libraries, paths, constants |
| 1. Load Data | Parse CSV, extract option metadata, compute TTE |
| 2. EDA | Missingness heatmap, smile shape, IV time-series |
| 3. Fill Methods | Define all 6 fill functions + apply_fill helpers |
| 4. Stratified CV | Mask at actual missing rates; run 6 CV fills; QP weights |
| 5. QP Optimisation | SLSQP per TTE bucket; print weights and CV MSE |
| 6. Build & Submit | Full-dataset builds; 3-bucket blend → submission.csv |
| 7. Sanity Checks | Row count, value range, zero-negative check, plot |

---

## File Structure

```
iv_imputation.ipynb    pipeline notebook (reproduces submission.csv exactly)
dataset.csv            input: raw option IV data (975 × 30, ~20% missing)
filled_dataset.csv     output: complete IV surface
submission.csv         output: Kaggle submission (id||value, 5460 rows)
README.md              this file
```
