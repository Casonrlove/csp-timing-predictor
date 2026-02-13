# Architectural Reframe: Timing-Level to Contract-Level Prediction

## Context

After 3 rounds of iteration, the timing model has plateaued at ~0.63 mean walk-forward AUC. Ablation testing confirmed expanding window + XGB/LGBM ensemble is the best configuration, but the fundamental problem is architectural:

1. **Three misaligned targets**: Training uses rolling P60 of Max_Drawdown_35D, inference chains timing + strike models independently, settlement uses flat -5% drawdown
2. **Two independent models** that don't share information: timing ("is it safe?") and strike ("what's P(assignment)?")
3. **Edge calculation** treats timing as a hard gate instead of a soft prior

The reframe unifies everything into a single contract-level model: `P(strike_breach | ticker, date, market_features, strike_otm_pct)`.

---

## Phase 1: Multi-Strike Data Generation

**Files**: `data_collector.py`, `feature_utils.py`

### `data_collector.py` — Add `create_contract_targets()`

New method alongside existing `create_target_variable()` (no breaking changes):

```python
def create_contract_targets(
    self,
    forward_days: int = 35,
    delta_buckets: list[float] | None = None,
) -> pd.DataFrame:
```

For each (date, delta_bucket):
- Convert delta to `strike_otm_pct` using IV estimate (reuse logic from `train_strike_model_v3.py:132-149`)
- Compute `strike_price = close * (1 - strike_otm_pct)`
- `strike_breached = 1 if min(close[t:t+35]) < strike_price else 0`
- Add 7 contract features: `target_delta`, `delta_x_vol`, `delta_x_vix`, `delta_x_iv_rank`, `delta_squared`, `delta_x_rsi`, `strike_otm_x_atr`

Vectorized inner loop (compute `min_price` once per day, then broadcast across deltas).

Row expansion: ~2,500 rows/ticker * 9 deltas = ~22,500 rows/ticker, ~405K total across 18 tickers.

### `feature_utils.py` — Add contract utilities

```python
def create_breach_target(df: pd.DataFrame) -> pd.DataFrame:
    """Set target = strike_breached. No rolling threshold needed."""

def contract_aware_time_split(df, n_splits=5, purge_days=35) -> list[tuple]:
    """TimeSeriesSplit on unique dates, not rows.
    All 9 strikes for the same (date, ticker) stay in the same fold.
    Purge the last 35 calendar days of train before each val fold."""
```

The `contract_aware_time_split` is critical: splits on unique dates via `TimeSeriesSplit`, purges the last 35 days of train before each val fold, then maps back to row indices. This prevents both:
- leakage where same-day different-delta rows end up in train vs val
- forward-window overlap leakage from 35-day labels near fold boundaries

---

## Phase 2: Contract Model Training

**File**: New `train_contract_model.py`

Reuses the existing ensemble architecture from `train_ensemble_v3.py` almost verbatim:
- Same 4 ticker groups, VIX regime split at 18.0
- XGB + LGBM DART base learners, 5-fold OOF stacking
- LogisticRegression meta-learner on `[xgb_prob, lgbm_prob, VIX, VIX_Rank, Regime_Trend]`
- StandardScaler for meta-features, isotonic calibration on tail 20% OOF
- Recency weighting (2yr half-life), expanding window

**Differences from `train_ensemble_v3.py`**:
1. Target: `strike_breached` (direct physical outcome) instead of rolling P60 threshold
2. Features: 56 market + 7 contract = 63 total
3. CV: `contract_aware_time_split()` instead of `TimeSeriesSplit`
4. Data: multi-strike expanded rows from `create_contract_targets()`

Saves to `csp_contract_model.pkl` (separate file, coexists with existing model).

**Training time estimate**: 9x more rows, ~2 hours with GPU. Can reduce trees (500 -> 300) if needed.

---

## Phase 3: Walk-Forward Validation

**File**: `walkforward_validation.py`

Add `--contract-mode` flag to `run_walk_forward()`:

```python
def run_walk_forward(
    ...,
    contract_mode: bool = False,
    delta_buckets: list[float] | None = None,
) -> dict:
```

When `contract_mode=True`:
- Expand data via `create_contract_targets()` before fold loop
- Use `create_breach_target()` instead of `apply_rolling_threshold()`
- Use `contract_aware_time_split()` for inner OOF CV
- New `_policy_metrics_contract()`: per-date, select best EV contract from 9 strike levels, track actual breach outcomes

**Key metric changes**:
- AUC on breach prediction (stratified by delta bucket for insight)
- Brier score (calibration quality — matters more now since edge depends on calibrated probabilities)
- Policy: EV/trade, PnL, win rate based on per-date contract selection

---

## Phase 4: Inference Integration

**File**: `simple_api_server.py`

New `CONTRACT_MODE` dispatch block (highest priority in model loader chain):

```
1. Collect 56 market features (same as now)
2. For each option contract from Schwab/Yahoo:
   a. Compute 7 contract features from delta + market state
   b. Concatenate → 63-feature vector
   c. Route to correct regime model (group + VIX regime)
   d. XGB + LGBM → meta-learner → calibrator → P(breach)
   e. edge_prob = market_delta - P(breach)  # probability units
      edge_pp = edge_prob * 100             # percentage points
   f. EV = premium - P(breach) * assignment_loss - slippage
3. Select best contract by EV (no timing_multiplier — timing is embedded)
4. Abstain if: near earnings, no positive-EV contracts, or |edge_pp| < 1.0
```

Backward compatibility: derive `p_safe = 1 - P(breach)` at delta~0.30 for the API response.

Fallback chain: contract model -> ensemble V3 -> per-ticker -> global.

---

## Phase 5: Settlement Alignment

**File**: `settle_prediction_outcomes.py`

Modify `settle_prediction()` to use contract-specific settlement:

```python
# Horizon alignment:
# - If prediction logged label_forward_days, use it directly (contract mode)
# - For contract predictions, do NOT derive horizon from expiration
# - Legacy predictions may still use expiration-derived horizon fallback

# If prediction logged a strike price, use it
strike = pred.get("suggested_strike")
if strike and strike > 0:
    outcome = 1 if min_price >= strike else 0  # 1 = profitable (not breached)
else:
    # Legacy: flat -5% threshold
    outcome = 1 if max_drawdown > -drop_threshold else 0
```

Extend `_log_prediction()` to include `model_p_breach`, `suggested_delta`, `suggested_otm_pct`, `model_version`, `label_forward_days`.

---

## Phase 6: Calibration Pipeline

**File**: `recalibrate_probabilities.py`

Update to calibrate `P(breach)` against actual breach outcomes from settled predictions. Same isotonic/Platt mechanics, just applied to the contract-level probability.

---

## Implementation Order

| Step | What | Files | Risk | Gate |
|------|------|-------|------|------|
| 1 | Multi-strike data generation | `data_collector.py`, `feature_utils.py` | None | Verify row counts + target distributions |
| 2 | Train contract model | New `train_contract_model.py` | None (parallel) | OOF AUC >= 0.65 |
| 3 | Walk-forward validation | `walkforward_validation.py` | None (parallel) | Mean AUC >= 0.63, policy PnL > 0 |
| 4 | Shadow inference | `simple_api_server.py` | Low (shadow mode) | Log comparison vs current model |
| 5 | Settlement alignment | `settle_prediction_outcomes.py` | Low | Contract-aware settlement matches training |
| 6 | Production cutover | Flip priority | Medium | 2 weeks shadow data looks good |

Steps 1-3 have zero production impact. Step 4 runs in shadow mode. Step 6 is the only breaking change.

---

## Polarity Convention

- **Model predicts**: `P(breach)` where breach=1 means CSP assigned (bad)
- **Edge**: `edge_prob = market_delta - P(breach)` (probability units), `edge_pp = edge_prob * 100` (percentage points). Positive = market overprices risk (good for selling)
- **Settlement**: `outcome = 1` means profitable (not breached), `outcome = 0` means assigned
- **Conversion**: `outcome = 1 - model_target`

---

## Verification

After implementing phases 1-3:

```bash
# Train contract model
python train_contract_model.py --use-optuna-params --recency-halflife 2.0 2>&1 | tee train_contract.log

# Walk-forward validation (contract mode)
python walkforward_validation.py --contract-mode 2>&1 | tee wf_contract.log

# Compare against current baseline
python walkforward_validation.py 2>&1 | tee wf_baseline.log
```

**Success criteria**:
- Contract model walk-forward mean AUC >= 0.63 (at least matching current ensemble)
- Policy PnL/trade positive across all groups
- Brier score < 0.20 (reasonable calibration)
- No degradation trend in recent folds (2024-2025)

---

## Critical Files

| File | Role |
|------|------|
| `data_collector.py` (lines 630-693) | Add `create_contract_targets()` alongside existing target generation |
| `feature_utils.py` | Add `create_breach_target()`, `contract_aware_time_split()` |
| `train_contract_model.py` (NEW) | Contract-level ensemble trainer, modeled after `train_ensemble_v3.py` |
| `walkforward_validation.py` (lines 150-498) | Add `--contract-mode` flag and contract-aware policy metrics |
| `simple_api_server.py` (lines 103-290, 605-850) | Contract model loading + inference dispatch |
| `settle_prediction_outcomes.py` (lines 91-121) | Contract-aware settlement |
| `train_ensemble_v3.py` | Pattern/reference for ensemble architecture (reused in contract model) |
| `train_strike_model_v3.py` (lines 124-169) | Delta-to-OTM conversion logic to reuse |
