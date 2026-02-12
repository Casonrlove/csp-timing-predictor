# CSP Timing Model — Round 3: Fix Validation + Improve Model Quality

## Context

After Round 2 (PCR removal, recency weighting, ticker expansion), walk-forward shows:
- Per-year AUC: 2021=0.688, 2022=0.588, 2023=0.659, 2024=0.650, 2025=0.564
- **Degrading trend persists** — 2022 and 2025 (rate-hike/regime-change years) are consistently bad
- Recency weighting alone was insufficient

Root cause analysis identified 5 high-impact problems. This plan addresses them in order of priority.

---

## 1. Match Walk-Forward Validation to Production Architecture (CRITICAL)

**Problem**: Walk-forward uses single XGBoost (300 trees, depth 6) with isotonic calibration. Production uses XGB+LGBM stacking ensemble (500 trees, depth 8) with no calibration. Current AUC numbers measure the *wrong model* — we can't trust them.

**File**: `walkforward_validation.py`

**Changes** to `run_walk_forward()` (lines 190–265):
- Replace single XGBoost training with full ensemble pipeline per fold:
  1. Apply VIX regime split (low/high at 18.0) within each fold's training data
  2. For each regime slice, run 5-fold TimeSeriesSplit OOF with XGBoost + LightGBM DART (matching `train_ensemble_v3.py` params)
  3. Fit LogisticRegression meta-learner on OOF `[xgb_prob, lgbm_prob, VIX, VIX_Rank, Regime_Trend]`
  4. Apply recency weighting with 2.0yr half-life via `sample_weight`
  5. At test time, route each row to correct regime model and predict through the full stack
- Remove the isotonic calibration step (production doesn't use it — yet; see improvement #3)
- Load Optuna params from `optuna_params_v3.json` when available (matching production)

**Impact**: No direct AUC change, but gives trustworthy numbers for measuring everything else.

---

## 2. Rolling 5-Year Training Window (HIGH IMPACT)

**Problem**: Expanding window means by 2025, training includes 2014–2024 (11 years). Pre-COVID data (2014–2018) is 45% of training rows and represents a fundamentally different market regime. Even with 2.0yr recency half-life, data 4+ years old still contributes ~25% effective weight. This is the primary driver of degradation on recent folds.

**Files**: `walkforward_validation.py`, `train_ensemble_v3.py`

**Changes**:

In `walkforward_validation.py` line 196, change:
```python
train_mask = df.index.year <= train_end_year
```
to:
```python
rolling_start = max(years[0], train_end_year - rolling_window + 1)
train_mask = (df.index.year >= rolling_start) & (df.index.year <= train_end_year)
```
Add `--rolling-window` CLI arg (default 5, 0 = expanding).

In `train_ensemble_v3.py`, filter collected data to most recent 5 years before training in `_train_regime_slice()`. Add `--rolling-window` CLI arg.

**Impact**: Expected +0.02–0.05 AUC on 2023–2025 folds. 2021 may dip slightly (less training data) but the degradation trend should flatten or reverse.

---

## 3. Add Isotonic Calibration to Production Training (HIGH IMPACT)

**Problem**: Production ensemble has **zero calibration**. Calibration curve shows predicted 0.45 → actual 0.88. This means edge calculations (`premium - expected_assignment_loss`) are computed on wrong probabilities, leading to bad trade decisions. The walk-forward uses isotonic calibration but production doesn't — another mismatch.

**Files**: `train_ensemble_v3.py`, `simple_api_server.py`

**Changes** in `train_ensemble_v3.py` `_train_regime_slice()`, after meta-learner fitting (line 314):
```python
# Hold out last 20% of OOF for calibration
cal_start = int(meta_fit_mask.sum() * 0.8)
cal_indices = np.where(meta_fit_mask)[0][cal_start:]
if len(cal_indices) >= 50:
    cal = CalibratedClassifierCV(FrozenEstimator(meta_learner), method='isotonic')
    cal.fit(meta_X[cal_indices], y[cal_indices])
    self.calibrators[model_key] = cal
```

Save `calibrators` dict in the pkl. In `simple_api_server.py` inference path (~line 616), apply calibrator when available:
```python
if model_key in ENSEMBLE_CALIBRATORS:
    p_safe = ENSEMBLE_CALIBRATORS[model_key].predict_proba(meta_X)[0, 1]
```

**Impact**: +0.01–0.02 AUC, but the real win is reliable edge calculations → better PnL.

---

## 4. Scale Meta-Learner Inputs + Remove 8 Redundant Features (MEDIUM IMPACT)

### 4A: Meta-Learner Scale Fix

**Problem**: LogisticRegression(C=1.0) receives `[xgb_prob(0–1), lgbm_prob(0–1), VIX(10–80), VIX_Rank(0–100), Regime_Trend(0/1)]`. L2 regularization penalizes all coefficients equally, so VIX and VIX_Rank dominate the meta-learner. It may learn "VIX is high → predict dangerous" instead of properly arbitrating between base model predictions.

**File**: `train_ensemble_v3.py` (line 313), `simple_api_server.py` (~line 610)

**Change**: Add `StandardScaler` for meta-features. Fit on OOF data, store in pkl as `meta_scalers`, apply at inference.

### 4B: Remove Redundant Features (64 → 56)

**Problem**: 8 features are algebraically redundant or degenerate noise:

| Feature | Why Remove |
|---------|-----------|
| `VIX_Percentile_252D` | ~0.95 corr with `VIX_Rank` |
| `Pullback_From_20D_High` | Algebraically identical to `Distance_From_High_20D` |
| `RSI_Oversold` | Threshold of `RSI` — trees handle this natively |
| `RSI_Extreme` | Threshold of `RSI` — trees handle this natively |
| `Recent_Drop_1D` | `abs(Return_1D) if <0 else 0` — perfectly reconstructible |
| `Hurst_Exponent_60D` | Only 2 sub-lags, extremely noisy. `Variance_Ratio_5D` covers the same concept |
| `VIX9D_Ratio` | Falls back to 1.0 when VIX9D unavailable (most historical data) |
| `VIX9D_vs_SMA5` | Falls back to 0.0 when VIX9D unavailable |

**File**: `data_collector.py` `prepare_features()` (lines 698–758)

**Change**: Remove these 8 from the `feature_cols` list. No other code changes needed — the features are still computed but just not included in the model input.

**Combined Impact**: +0.01–0.03 AUC from reduced noise and better meta-learner arbitration.

---

## 5. Target-Encoded Ticker Identifier (MEDIUM IMPACT)

**Problem**: All tickers within a group share one model. TSLA and PLTR behave very differently but the model has no way to distinguish them — no ticker ID feature exists.

**Files**: `train_ensemble_v3.py`, `walkforward_validation.py`, `simple_api_server.py`

**Changes**:
- In `_train_regime_slice()`, compute leave-one-out target-encoded ticker feature with Bayesian smoothing:
```python
ticker_means = df_slice.groupby('ticker')['target'].transform('mean')
n = df_slice.groupby('ticker')['target'].transform('count')
global_mean = df_slice['target'].mean()
smoothing = 100
ticker_encoded = (ticker_means * n + global_mean * smoothing) / (n + smoothing)
```
- Add as extra feature column before training
- In walk-forward, compute encoding using only training fold data (prevent leakage)
- Save `ticker_target_rates` map in pkl for inference
- At inference, look up ticker's historical rate

**Impact**: +0.01–0.02 AUC, most beneficial for `high_vol` group (6 diverse tickers).

---

## Implementation Order

| Step | Change | Files | Complexity |
|------|--------|-------|-----------|
| 1 | Match walk-forward to production | `walkforward_validation.py` | Medium |
| 2 | Rolling 5yr window | `walkforward_validation.py`, `train_ensemble_v3.py` | Low |
| 3 | Isotonic calibration in production | `train_ensemble_v3.py`, `simple_api_server.py` | Low-Medium |
| 4 | Meta-scaler + prune 8 features | `train_ensemble_v3.py`, `simple_api_server.py`, `data_collector.py` | Low |
| 5 | Ticker encoding | `train_ensemble_v3.py`, `walkforward_validation.py`, `simple_api_server.py` | Medium |

Steps 1–4 should be implemented together, then validated. Step 5 can follow as a separate iteration.

---

## Execution Pipeline

After implementing steps 1–4:
```bash
rm -f optuna_params_v3.json
python tune_ensemble_v3.py 2>&1 | tee tune_v3_round3.log
python train_ensemble_v3.py --use-optuna-params --recency-halflife 2.0 --rolling-window 5 2>&1 | tee train_v3_round3.log
python walkforward_validation.py --rolling-window 5 2>&1 | tee wf_v3_round3.log
```

## Verification

Compare walk-forward per-year AUC to Round 2 baseline:
- **Round 2**: 2021=0.688, 2022=0.588, 2023=0.659, 2024=0.650, 2025=0.564
- **Target**: Mean AUC ≥ 0.68, degradation trend flat or improving
- **Key metric**: 2025 AUC should improve from 0.564 → 0.62+

---

## Critical Files

- `walkforward_validation.py` — rewrite to mirror production ensemble, add rolling window
- `train_ensemble_v3.py` — add rolling window, calibration, meta-scaler, ticker encoding
- `data_collector.py` — remove 8 redundant features from `prepare_features()`
- `simple_api_server.py` — load/apply calibrators and meta-scalers at inference
- `feature_utils.py` — may need shared helpers for ticker encoding
