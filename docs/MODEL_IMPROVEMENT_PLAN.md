# Maximum Accuracy CSP Timing Model — Improvement Plan

## Current Baseline
- **Architecture**: XGBoost per-ticker-group classifier (4 groups: high_vol, tech_growth, tech_stable, etf)
- **Features**: 54 features (technical indicators, VIX, mean reversion, Hurst, autocorrelation)
- **Walk-forward AUC**: 0.62–0.72 depending on group and regime
- **Key weaknesses**:
  - Training threshold (fixed full-history P60) mismatches validation (rolling 90D P60)
  - Single model trained on all VIX regimes but they're fundamentally different tasks
  - Only one base learner — no ensemble diversity

## Expected Outcomes by Phase

| Phase | Change | Expected Walk-Forward AUC |
|---|---|---|
| Baseline | Current production | 0.62–0.72 |
| Phase 0 | Fix bugs + threshold mismatch | 0.64–0.73 |
| Phase 1 | Regime-separated training | 0.68–0.78 |
| Phase 2 | XGB + LGBM ensemble + stacking | 0.70–0.80 |
| Phase 3 | Optuna tuning (overnight) | 0.71–0.81 |
| Phase 4 | New features (sector RS, VIX9D, PCR) | 0.72–0.82 |
| Phase 5 | Ticker expansion | 0.73–0.82 |

**Realistic ceiling**: AUC ~0.80–0.82 for low-VIX conditions. The 35-day drawdown target has ~30% irreducible noise from macro shocks no feature set can capture.

---

## Phase 0 — Fix Bugs (do first, everything else builds on this)

### 0.1 Fix Days_To_Earnings
**File**: `data_collector.py`
**Problem**: Returns -2991 in production — out-of-distribution garbage that degrades predictions and skews the StandardScaler.
**Fix**: Clamp output to `clip(raw_value, -30, 90)`. Anything older than 30 days past or beyond 90 days future is irrelevant.

### 0.2 Align Training Target to Rolling 90D P60
**Files**: `train_timing_model_per_ticker.py`, `walkforward_validation.py` → new `feature_utils.py`
**Problem**: Training uses `calculate_optimal_threshold()` (fixed full-history P60). Validation uses `apply_rolling_threshold()` (rolling 90D P60). These produce different label distributions — current AUC numbers cannot be trusted.
**Fix**:
1. Create `feature_utils.py` — extract `apply_rolling_threshold()` as a shared utility
2. Replace the static threshold block in `train_timing_model_per_ticker.py` lines 110–157 with a call to `apply_rolling_threshold(df, window=90, quantile=0.60)`
3. Update `walkforward_validation.py` to import from `feature_utils.py`

---

## Phase 1 — Regime-Separated Training (biggest single lever)

**Insight from walk-forward**: `--vix-regime low` showed AUC 0.72+ improving trend. The combined model is trying to learn two different tasks simultaneously and doing neither optimally.

### Architecture
Train **2 models per ticker group** (8 sub-models total):
- `{group}_low_vix` — trained on days where VIX < 18 (~60% of trading days)
- `{group}_high_vix` — trained on days where VIX ≥ 18 (~40% of trading days)

Fallback: if a regime slice has < 400 training samples for a group, use combined model.

### File to Create: `train_timing_model_v3.py`
Same structure as current `train_timing_model_per_ticker.py` but adds VIX regime split inside `train_group_model()`. Saves to `csp_timing_model_v3.pkl` with `'regime_models': True`.

Model dict keys: `'{group}_low_vix'`, `'{group}_high_vix'` plus scalers and feature_cols.

### API Change (`simple_api_server.py`)
```python
regime = 'low_vix' if current_vix < 18 else 'high_vix'
model_key = f'{group}_{regime}'  # fallback to group if key missing
```
Add `vix_regime` label to API response `technical_context`.

---

## Phase 2 — XGB + LightGBM Ensemble with Stacking

### Base Learners
- **XGBoost**: existing config (keep `scale_pos_weight` for class balance)
- **LightGBM**: `boosting_type='dart'`, `num_leaves=63`, `min_child_samples=20`, `feature_fraction=0.7`, `bagging_fraction=0.8`, `bagging_freq=5`, `objective='binary'`, `metric='auc'`

DART boosting in LightGBM drops trees randomly during training — produces complementary signal to XGBoost's subsampling. Low agreement (~80-85%) between the two means real diversity gains.

### Meta-Learner (Stacking, not simple average)
Simple averaging assumes both models are equally good in all conditions. A stacked meta-learner learns regime-conditional blend weights.

**Protocol**:
1. 5-fold time-series CV (no shuffle) on training set → generate OOF predictions from both base learners
2. Meta-features: `[xgb_prob, lgbm_prob, VIX, VIX_Rank, Regime_Trend]`
3. Meta-learner: `LogisticRegression(C=1.0)` — simple, not prone to overfitting
4. Retrain final base learners on full training set; meta-learner trained on OOF

### File to Create: `train_ensemble_v3.py`
Combines Phase 1 regime split + Phase 2 XGB + LGBM + stacking. Saves to `csp_timing_ensemble.pkl`.

```python
# Saved pkl structure
{
    'ensemble': True,
    'regime_models': True,
    'base_models': {'{group}_{regime}': {'xgb': model, 'lgbm': model}},
    'meta_learners': {'{group}_{regime}': logistic_reg},
    'scalers': {...},
    'feature_cols': [...],
    'group_mapping': {...},
    'version': 'ensemble_v3'
}
```

### API Changes (`simple_api_server.py`)
- New globals: `ENSEMBLE_MODE`, `ENSEMBLE_BASE_MODELS`, `ENSEMBLE_META_LEARNERS`
- Load `csp_timing_ensemble.pkl` preferentially; old pkl format stays backward compatible
- Prediction dispatch:
  ```python
  xgb_prob = base_models[key]['xgb'].predict_proba(X)[0][1]
  lgbm_prob = base_models[key]['lgbm'].predict_proba(X)[0][1]
  meta_X = [[xgb_prob, lgbm_prob, vix, vix_rank, regime_trend]]
  final_prob = meta_learners[key].predict_proba(meta_X)[0][1]
  ```

---

## Phase 3 — Optuna Tuning (runs overnight)

### File to Create: `tune_ensemble_v3.py`
- 16 independent Optuna studies: 4 groups × 2 regimes × 2 base learners
- **Objective: maximize ROC-AUC** (NOT Brier score — that was the strike model mistake)
- CV: 3-fold expanding window within training data only (no test leakage)
- Config: TPE sampler, MedianPruner, 200 trials per study, 1-hour timeout per study
- Output: `optuna_params_v3.json`; `train_ensemble_v3.py` loads this

### XGBoost Search Space
`n_estimators` 300–1500, `max_depth` 4–10, `learning_rate` 0.01–0.15, `subsample` 0.6–1.0, `colsample_bytree` 0.5–1.0, `min_child_weight` 1–10, `gamma` 0–0.5, `reg_alpha` 0–1.0, `reg_lambda` 0.5–2.0

### LightGBM Search Space
`num_leaves` 20–200, `min_child_samples` 10–50, `learning_rate` 0.01–0.15, `feature_fraction` 0.5–1.0, `bagging_fraction` 0.6–1.0, `lambda_l1` 0–1.0, `lambda_l2` 0–2.0

---

## Phase 4 — New Features

All added to `data_collector.py` → `calculate_technical_indicators()` and `prepare_features()` feature_cols list.

### Tier 1 — Highest Impact (8 new features → 62 total)

**A. SPY-Relative Return (2 features)**
- `Stock_vs_SPY_5D`, `Stock_vs_SPY_20D`
- Stock that's down 5% while SPY is up 2% is far riskier than one down 5% with SPY down 4%
- Fetch SPY once per DataCollector instance, cache as class-level variable

**B. Sector ETF Relative Strength (2 features)**
- `Sector_RS_5D`, `Sector_RS_20D` — stock return minus its sector ETF return
- Add `SECTOR_ETF_MAP` to `CSPDataCollector`:
  ```python
  SECTOR_ETF_MAP = {
      'NVDA': 'SMH', 'AMD': 'SMH', 'TSLA': 'XLY',
      'META': 'XLC', 'GOOGL': 'XLC', 'AMZN': 'XLY',
      'AAPL': 'XLK', 'MSFT': 'XLK',
      'SPY': 'SPY', 'QQQ': 'QQQ',
  }
  ```

**C. VIX Term Structure — VIX9D (2 features)**
- `VIX9D_Ratio = ^VIX9D / VIX`, `VIX9D_vs_SMA5`
- Near-term fear vs medium-term fear — one of the strongest short-term options sentiment signals
- Fetch `^VIX9D` from yfinance alongside `^VIX` in `fetch_data()`
- Replaces/supplements existing `Vol_Term_Structure` (which uses realized vol, not implied)

**D. Put/Call Ratio (2 features)**
- `PCR_5D_MA`, `PCR_ZScore_20D`
- Elevated PCR (>1.2) = contrarian bullish signal; Low PCR (<0.7) = complacency warning
- Fetch `^PC-CBOE` from yfinance in `fetch_data()`

### Tier 2 — Medium Impact (add after Tier 1 validated)
- `AD_Line_Proxy_5D`: advance/decline breadth proxy from SPY
- `Recent_Beat_Rate`: earnings surprise history from `yfinance quarterly_earnings`

### Do NOT Add
- LSTM features, Transformer, TabNet — poor complexity/gain ratio for this dataset size
- Historical options IV surface — training data doesn't exist for historical rows

---

## Phase 5 — Ticker Expansion

Validate each candidate with walk-forward before adding (mean AUC must not drop > 0.005).

| Group | Add | Rationale |
|---|---|---|
| high_vol | MSTR, COIN, PLTR | Bitcoin-correlated, strong mean reversion |
| tech_growth | NFLX, CRM | Quarterly earnings cycles, solid CSP candidates |
| tech_stable | V, MA | Anchors the low-vol end of the spectrum |
| etf | IWM | Small-cap ETF, different regime behavior from SPY/QQQ |

**Do NOT add**: SQQQ, TUVT or any inverse ETF.

---

## Implementation Order

```
Step 1.  feature_utils.py                — extract apply_rolling_threshold() (shared)
Step 2.  data_collector.py               — fix Days_To_Earnings, add 8 Tier-1 features
Step 3.  train_timing_model_per_ticker.py — fix threshold alignment (import feature_utils)
Step 4.  train_timing_model_v3.py        — regime-split single XGB model
Step 5.  VALIDATE: python walkforward_validation.py --vix-regime low
Step 6.  train_ensemble_v3.py            — add LGBM + stacking
Step 7.  tune_ensemble_v3.py             — Optuna overnight run
Step 8.  train_ensemble_v3.py            — retrain with Optuna params
Step 9.  VALIDATE: python walkforward_validation.py
Step 10. simple_api_server.py            — wire ensemble pkl into production
Step 11. Ticker expansion (validate each addition)
```

---

## Files to Create
| File | Purpose |
|---|---|
| `feature_utils.py` | Shared `apply_rolling_threshold()`, `compute_vix_regime()` |
| `train_timing_model_v3.py` | Regime-split XGB model (Phase 1) |
| `train_ensemble_v3.py` | XGB + LGBM + stacking (Phase 2 + 3) |
| `tune_ensemble_v3.py` | Optuna hyperparameter search |

## Files to Modify
| File | Changes |
|---|---|
| `data_collector.py` | Fix Days_To_Earnings; add sector RS, VIX9D, PCR features; SECTOR_ETF_MAP |
| `train_timing_model_per_ticker.py` | Import + use rolling threshold from feature_utils |
| `walkforward_validation.py` | Import rolling threshold from feature_utils |
| `simple_api_server.py` | Ensemble dispatch, regime routing, backward-compatible |

## Verification Commands
```bash
python walkforward_validation.py --group etf          # quick sanity check
python walkforward_validation.py                      # full run, all groups
python walkforward_validation.py --vix-regime low     # primary use-case
```

**Target**: per-year AUC ≥ 0.72 for low-VIX years, overall mean AUC ≥ 0.75.
