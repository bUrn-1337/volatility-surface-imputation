"""
Standalone script to regenerate v6 submission.csv without running the SVI benchmark.
Same logic as the notebook sections 4.1–4.8c + 4.7 + 5.1 + 5.2.
"""
import re
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize as _minimize
from sklearn.metrics import mean_squared_error

SEED = 42
np.random.seed(SEED)
DATASET_PATH    = 'dataset.csv'
FILLED_PATH     = 'filled_dataset.csv'
SUBMISSION_PATH = 'submission.csv'
SEPARATOR       = '||'
EXPIRY          = pd.Timestamp('2026-01-27 15:30')
TTE_SPLIT       = 5 * 390  # 1950 minutes = 5 trading days

# ── Load data ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATASET_PATH)
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], format='%d-%m-%Y %H:%M')
df_raw = df_raw.sort_values('datetime').reset_index(drop=True)
df_raw['tte_minutes'] = (EXPIRY - df_raw['datetime']).dt.total_seconds() / 60

def parse_ticker(col):
    m = re.match(r'NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)', col)
    return {'strike': int(m.group(2)), 'option_type': m.group(3)} if m else {}

option_cols = [c for c in df_raw.columns if c not in ('datetime', 'underlying_price')]
ce_cols     = [c for c in option_cols if c.endswith('CE')]
pe_cols     = [c for c in option_cols if c.endswith('PE')]
ce_strikes  = sorted([parse_ticker(c)['strike'] for c in ce_cols])
pe_strikes  = sorted([parse_ticker(c)['strike'] for c in pe_cols])
print(f'Loaded {df_raw.shape}, {df_raw.isna().sum().sum()} missing')

# ── Fill functions ───────────────────────────────────────────────────────────
def fill_linear(row_iv, strikes):
    obs = [(s,v) for s,v in zip(strikes, row_iv.values) if not np.isnan(v)]
    if len(obs) < 2: return row_iv
    s_arr = np.array([o[0] for o in obs]); v_arr = np.array([o[1] for o in obs])
    filled = np.interp(np.array(strikes), s_arr, v_arr)
    r = row_iv.copy()
    for i,s in enumerate(strikes):
        if np.isnan(row_iv[s]): r[s] = max(filled[i], 1e-4)
    return r

def fill_poly2(row_iv, strikes):
    obs = [(s,v) for s,v in zip(strikes, row_iv.values) if not np.isnan(v)]
    if len(obs) < 3: return fill_linear(row_iv, strikes)
    s_arr = np.array([o[0] for o in obs]); v_arr = np.array([o[1] for o in obs])
    try: p = np.polyfit(s_arr, v_arr, 2); filled = np.polyval(p, np.array(strikes))
    except: filled = np.interp(np.array(strikes), s_arr, v_arr)
    r = row_iv.copy()
    for i,s in enumerate(strikes):
        if np.isnan(row_iv[s]): r[s] = max(filled[i], 1e-4)
    return r

def fill_poly2_var(row_iv, strikes):
    obs = [(s,v**2) for s,v in zip(strikes, row_iv.values) if not np.isnan(v) and v > 0]
    if len(obs) < 3: return fill_linear(row_iv, strikes)
    s_arr = np.array([o[0] for o in obs]); v_arr = np.array([o[1] for o in obs])
    try:
        p = np.polyfit(s_arr, v_arr, 2)
        filled = np.sqrt(np.clip(np.polyval(p, np.array(strikes)), 1e-8, None))
    except: filled = np.interp(np.array(strikes), s_arr, np.sqrt(v_arr))
    r = row_iv.copy()
    for i,s in enumerate(strikes):
        if np.isnan(row_iv[s]): r[s] = max(filled[i], 1e-4)
    return r

def fill_pchip(row_iv, strikes):
    obs = [(s,v) for s,v in zip(strikes, row_iv.values) if not np.isnan(v)]
    if len(obs) < 2: return row_iv
    s_arr = np.array([o[0] for o in obs]); v_arr = np.array([o[1] for o in obs])
    all_s = np.array(strikes); filled = np.interp(all_s, s_arr, v_arr)
    if len(obs) >= 3:
        interior = (all_s >= s_arr[0]) & (all_s <= s_arr[-1])
        if interior.any(): filled[interior] = PchipInterpolator(s_arr, v_arr)(all_s[interior])
    r = row_iv.copy()
    for i,s in enumerate(strikes):
        if np.isnan(row_iv[s]): r[s] = max(filled[i], 1e-4)
    return r

def fill_adaptive(row_iv, strikes):
    obs = [(s,v) for s,v in zip(strikes, row_iv.values) if not np.isnan(v)]
    if len(obs) < 2: return fill_linear(row_iv, strikes)
    obs_s = np.array([o[0] for o in obs])
    s_min, s_max = obs_s.min(), obs_s.max()
    r_poly  = fill_poly2(row_iv, strikes)
    r_pchip = fill_pchip(row_iv, strikes)
    r = row_iv.copy()
    for i,s in enumerate(strikes):
        if np.isnan(row_iv[s]):
            r[s] = r_pchip[s] if s_min <= s <= s_max else r_poly[s]
    return r

def fill_logwing(row_iv, strikes):
    obs = [(s,v) for s,v in zip(strikes, row_iv.values) if not np.isnan(v) and v > 0]
    if len(obs) < 2: return fill_linear(row_iv, strikes)
    s_obs = np.array([o[0] for o in obs]); v_obs = np.array([o[1] for o in obs])
    lv_obs = np.log(v_obs); s_min, s_max = s_obs[0], s_obs[-1]
    all_s = np.array(strikes)
    filled = np.full(len(strikes), np.nan)
    interior = (all_s >= s_min) & (all_s <= s_max)
    if interior.any():
        filled[interior] = (PchipInterpolator(s_obs, v_obs)(all_s[interior])
                            if len(obs) >= 3 else
                            np.interp(all_s[interior], s_obs, v_obs))
    lslope = np.clip((lv_obs[1]-lv_obs[0])/(s_obs[1]-s_obs[0]), -0.02, 0.02)
    rslope = np.clip((lv_obs[-1]-lv_obs[-2])/(s_obs[-1]-s_obs[-2]), -0.02, 0.02)
    for i,s in enumerate(strikes):
        if not interior[i]:
            filled[i] = (np.exp(lv_obs[0]+lslope*(s-s_min)) if s < s_min
                         else np.exp(lv_obs[-1]+rslope*(s-s_max)))
    r = row_iv.copy()
    for i,s in enumerate(strikes):
        if np.isnan(row_iv[s]): r[s] = max(float(filled[i]), 1e-4)
    return r

def apply_fill(df, fill_fn):
    out = df.copy()
    for cols, strikes in [(ce_cols, ce_strikes), (pe_cols, pe_strikes)]:
        piv = out[cols].copy(); piv.columns = strikes
        for idx in piv.index: piv.loc[idx] = fill_fn(piv.loc[idx], strikes)
        out[cols] = piv.values
    return out

# ── CV mask (same seed as notebook) ─────────────────────────────────────────
np.random.seed(SEED)
obs_positions = [(i, col) for i in df_raw.index
                 for col in option_cols if not pd.isna(df_raw.loc[i, col])]
mask_idx = np.random.choice(len(obs_positions), size=int(len(obs_positions)*0.10), replace=False)
masked   = [obs_positions[i] for i in mask_idx]
true_ivs = np.array([df_raw.loc[i, col] for i, col in masked])
df_masked = df_raw.copy()
for i, col in masked: df_masked.loc[i, col] = np.nan

def get_preds(fill_fn):
    df_f = apply_fill(df_masked, fill_fn)
    return np.array([df_f.loc[i, col] for i, col in masked])

print('Computing CV predictions …')
p_p2  = get_preds(fill_poly2)
p_var = get_preds(fill_poly2_var)
p_ph  = get_preds(fill_pchip)
p_adp = get_preds(fill_adaptive)
p_lgw = get_preds(fill_logwing)

# TTE group split
tte_vals   = np.array([df_raw.loc[i, 'tte_minutes'] for i, _ in masked])
normal_idx = np.where(tte_vals >= TTE_SPLIT)[0]
expiry_idx = np.where(tte_vals <  TTE_SPLIT)[0]
print(f'CV positions: {len(normal_idx)} normal, {len(expiry_idx)} expiry')

# ── QP weight optimisation ───────────────────────────────────────────────────
def qp_blend_weights(preds_dict, true_vals, n_restarts=20):
    ks = list(preds_dict.keys())
    X  = np.stack([preds_dict[k] for k in ks], axis=1)
    def obj(w): return mean_squared_error(true_vals, X @ w)
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * len(ks)
    best = None
    np.random.seed(SEED)
    for _ in range(n_restarts):
        w0  = np.random.dirichlet(np.ones(len(ks)))
        res = _minimize(obj, w0, method='SLSQP', bounds=bounds,
                        constraints=constraints,
                        options={'ftol': 1e-12, 'maxiter': 2000})
        if best is None or res.fun < best.fun: best = res
    return dict(zip(ks, best.x)), best.fun

preds = {'p2': p_p2, 'var': p_var, 'ph': p_ph, 'adp': p_adp, 'lgw': p_lgw}
KEYS_QP = list(preds.keys())

print('QP search — normal …')
W_QP_NORMAL, _ = qp_blend_weights(
    {k: preds[k][normal_idx] for k in KEYS_QP}, true_ivs[normal_idx])
print('QP search — expiry …')
W_QP_EXPIRY, _ = qp_blend_weights(
    {k: preds[k][expiry_idx] for k in KEYS_QP}, true_ivs[expiry_idx])

pmat = np.stack([preds[k] for k in KEYS_QP], axis=1)
p_qp = np.empty_like(true_ivs)
p_qp[normal_idx] = pmat[normal_idx] @ np.array([W_QP_NORMAL[k] for k in KEYS_QP])
p_qp[expiry_idx] = pmat[expiry_idx] @ np.array([W_QP_EXPIRY[k] for k in KEYS_QP])
print(f'Normal weights : {" ".join(f"{k}={v:.3f}" for k,v in W_QP_NORMAL.items())}')
print(f'Expiry weights : {" ".join(f"{k}={v:.3f}" for k,v in W_QP_EXPIRY.items())}')
print(f'Simulated MSE  : {mean_squared_error(true_ivs, p_qp):.8f}')

# ── Build full-dataset fills ─────────────────────────────────────────────────
print('Building 5 fills on full dataset …')
df_p2_full  = apply_fill(df_raw.copy(), fill_poly2)
df_var_full = apply_fill(df_raw.copy(), fill_poly2_var)
df_ph_full  = apply_fill(df_raw.copy(), fill_pchip)
df_adp_full = apply_fill(df_raw.copy(), fill_adaptive)
df_lgw_full = apply_fill(df_raw.copy(), fill_logwing)

fills = {'p2': df_p2_full, 'var': df_var_full, 'ph': df_ph_full,
         'adp': df_adp_full, 'lgw': df_lgw_full}

# ── v6 QP blend → filled_dataset.csv ────────────────────────────────────────
df_filled = df_raw.copy()
w_n = np.array([W_QP_NORMAL[k] for k in KEYS_QP])
w_e = np.array([W_QP_EXPIRY[k] for k in KEYS_QP])

for col in option_cols:
    missing_mask = df_raw[col].isna()
    if not missing_mask.any(): continue
    for idx in df_raw.index[missing_mask]:
        tte = df_raw.loc[idx, 'tte_minutes']
        w   = w_e if tte < TTE_SPLIT else w_n
        vals = np.array([fills[k].loc[idx, col] for k in KEYS_QP])
        df_filled.loc[idx, col] = max(float(np.dot(vals, w)), 1e-4)

remaining = df_filled[option_cols].isna().sum().sum()
assert remaining == 0
df_filled['datetime'] = df_filled['datetime'].dt.strftime('%d-%m-%Y %H:%M')
df_filled.to_csv(FILLED_PATH, index=False)
print(f'Saved {FILLED_PATH}')

# ── Generate submission.csv ──────────────────────────────────────────────────
original = pd.read_csv(DATASET_PATH)
filled   = pd.read_csv(FILLED_PATH)
rows = []
for col in [c for c in original.columns if c != 'datetime']:
    for idx in original.index[original[col].isna()]:
        dt = original.loc[idx, 'datetime']
        rows.append({'id': f'{dt}{SEPARATOR}{col}', 'value': filled.loc[idx, col]})
solution = (pd.DataFrame(rows, columns=['id','value'])
            .sort_values('id').reset_index(drop=True))
solution.to_csv(SUBMISSION_PATH, index=False)
print(f'Saved {SUBMISSION_PATH}  ({len(solution)} rows)')
print(f'Value range: [{solution.value.min():.5f}, {solution.value.max():.5f}]')
