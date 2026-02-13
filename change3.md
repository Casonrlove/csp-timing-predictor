# change3.md

## Scope
Implemented the first two priority fixes from the review:
1. Add purge/embargo to contract-mode CV splits (35-day label horizon aware).
2. Standardize edge-unit handling so thresholding is unit-safe and explicit.
3. Align contract training/settlement horizons to the same label window.
4. Remove mutable-list default from plan signature example.
5. Add Schwab-only data-path controls and request timeout hardening for long runs.

## Files Changed
- `feature_utils.py`
- `train_contract_model.py`
- `walkforward_validation.py`
- `simple_api_server.py`
- `settle_prediction_outcomes.py`
- `CONTRACT_MODEL_PLAN.md`
- `data_collector.py`
- `schwab_client.py`
- `schwab_auth.py`

## 1) Purged Contract CV Splits

### `feature_utils.py`
Updated `contract_aware_time_split(...)`:
- Signature changed to include `purge_days`:
  - `contract_aware_time_split(df, n_splits=5, purge_days=35)`
- Split logic now:
  - splits on unique dates,
  - removes training dates newer than `(validation_start - purge_days)`,
  - maps back to row indices,
  - skips empty folds,
  - raises a clear error if no valid folds are produced.
- Docstring expanded to explain forward-window leakage risk and purge behavior.

### `train_contract_model.py`
- Added `FORWARD_DAYS = 35`.
- Updated split call to:
  - `contract_aware_time_split(df_slice, n_splits=N_META_FOLDS, purge_days=FORWARD_DAYS)`

### `walkforward_validation.py`
- Added `FORWARD_DAYS = 35`.
- Updated contract-mode inner OOF split call to:
  - `contract_aware_time_split(rdf, n_splits=N_META_FOLDS, purge_days=FORWARD_DAYS)`

## 2) Edge Unit Standardization

### `simple_api_server.py`
Added explicit edge unit constants:
- `MIN_EDGE_PP = 1.0`
- `MIN_EDGE_PROB = MIN_EDGE_PP / 100.0`

Updated edge computation and output fields per option:
- `edge_prob = market_delta - model_prob` (probability units)
- `edge_pp = edge_prob * 100.0` (percentage points)
- Added output fields:
  - `edge_prob`
  - `edge_pp`
- Kept backward compatibility:
  - `edge` is still present and now explicitly set as a legacy alias to `edge_pp`.
- `edge_signal` now keys off `edge_prob > 0`.

Updated no-trade edge band filter:
- From implicit/ambiguous `abs(opt['edge']) >= 1.0`
- To explicit `abs(opt['edge_prob']) >= MIN_EDGE_PROB`

## 3) Plan Doc Alignment

### `CONTRACT_MODEL_PLAN.md`
Aligned plan text with implemented behavior:
- CV split signature now includes `purge_days=35`.
- Added explicit statement that split purges the last 35 days before validation to avoid forward-window overlap leakage.
- Edge formula now uses both units:
  - `edge_prob = market_delta - P(breach)`
  - `edge_pp = edge_prob * 100`
- Abstain threshold now explicitly uses percentage points:
  - `|edge_pp| < 1.0`
- `create_contract_targets(...)` example now uses `delta_buckets: list[float] | None = None` (no mutable default).

## 4) Training/Settlement Horizon Alignment

### `simple_api_server.py`
- Added `CONTRACT_FORWARD_DAYS` runtime constant (default 35).
- When loading `csp_contract_model.pkl`, now reads `forward_days` metadata (if present).
- Contract prediction logs now include:
  - `model_version='contract_v1'` (for all contract-mode predictions)
  - `label_forward_days=<contract horizon>`
- `_log_prediction(...)` signature extended to accept `label_forward_days`.

### `train_contract_model.py`
- Model persistence now includes:
  - `forward_days` metadata in saved artifact (`csp_contract_model.pkl`).

### `settle_prediction_outcomes.py`
Settlement horizon selection now follows strict precedence:
1. Use logged `label_forward_days` when available.
2. For contract predictions (`model_version`/`model_p_breach`), use fixed contract horizon.
3. For legacy predictions, keep expiration-derived fallback behavior.

This prevents variable expiration windows from drifting away from fixed-horizon training labels in contract mode.

## Verification Performed
Ran syntax checks for touched Python files:

```bash
python -m py_compile feature_utils.py train_contract_model.py walkforward_validation.py simple_api_server.py settle_prediction_outcomes.py
python -m py_compile data_collector.py settle_prediction_outcomes.py
```

Result: passed (no syntax errors).

## Notes
- Horizon alignment is now implemented for contract-mode predictions via `label_forward_days` and settlement precedence rules.
- All four original review findings are now addressed in code/doc.

## 5) Schwab-Only + Runtime Hardening

### `data_collector.py`
- Added strict Schwab-only mode switch (now default-on):
  - `self.schwab_only = (os.getenv("SCHWAB_ONLY", "1") == "1")`
- In `fetch_data()`:
  - if `SCHWAB_ONLY=1` and Schwab data is unavailable, now hard-fails instead of falling back.
  - Yahoo fallback for prices, earnings, and VIX is disabled in Schwab-only mode.
  - VIX9D/market-context Yahoo reference fetches are disabled in Schwab-only mode (features safely fall back to neutral defaults).
- Added explicit logging lines so runtime shows when Yahoo paths are intentionally skipped.

### `schwab_client.py`
- Added request timeout guard:
  - `REQUEST_TIMEOUT_SECONDS = 20`
- `_make_request(...)` now passes `timeout=REQUEST_TIMEOUT_SECONDS` to all `requests.get(...)` calls, including retry-after-refresh requests.

### `schwab_auth.py`
- Added request timeout guard:
  - `REQUEST_TIMEOUT_SECONDS = 20`
- Token exchange and token refresh `requests.post(...)` calls now include timeout, preventing indefinite OAuth hangs during long jobs.

### `walkforward_validation.py`
- Added per-ticker watchdog timeout (`SIGALRM`) to avoid a single ticker blocking full-group collection.
- Added `SKIP_TICKERS` env support to bypass known-problem symbols during triage runs.

### `settle_prediction_outcomes.py`
- Added `SCHWAB_ONLY` default-on behavior for settlement data fetches.
- When `SCHWAB_ONLY=1`:
  - Yahoo fallback is disabled by default.
  - `--force-yahoo` is rejected to prevent accidental mixed-source settlement.
- `fetch_close_series(...)` now accepts `allow_yahoo_fallback` and respects strict Schwab-only mode.

## Validation Status (Current Session)
- Contract training completed successfully under `SCHWAB_ONLY=1`.
- Completed Schwab-only contract walk-forward run (`wf_contract.log`) with triage skips:
  - command: `SKIP_TICKERS=MSTR,COIN,PLTR SCHWAB_ONLY=1 XDG_CACHE_HOME=/tmp python walkforward_validation.py --period 10y --contract-mode`
  - `high_vol`: mean AUC `0.6738` (partial group due to skips)
  - `tech_growth`: mean AUC `0.7172`
  - `tech_stable`: mean AUC `0.6298`
  - `etf`: mean AUC `0.6617`
  - per-year average AUCs: `2021=0.6444`, `2022=0.6668`, `2023=0.6692`, `2024=0.6606`, `2025=0.7121`
- Attempted full no-skip Schwab-only rerun (`wf_contract_full.log`) with:
  - command: `SCHWAB_ONLY=1 XDG_CACHE_HOME=/tmp python walkforward_validation.py --period 10y --contract-mode`
  - blocked by DNS outage resolving `api.schwabapi.com` (`socket.gaierror: [Errno -2] Name or service not known`)
  - outcome: no full-group metrics produced in that attempt.

## Next Run (After DNS Recovery)
Run this exact command to complete strict all-symbol Schwab-only validation:

```bash
SCHWAB_ONLY=1 XDG_CACHE_HOME=/tmp python walkforward_validation.py --period 10y --contract-mode 2>&1 | tee wf_contract_full.log
```
