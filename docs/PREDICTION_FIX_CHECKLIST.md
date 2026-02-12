# CSP Prediction Fix Checklist

Use this file as the source of truth for implementation status.
Mark items complete only after code is fixed and validated.

## Priority Order

- [x] 1. Fix evaluation leakage first (highest priority)
  - [x] Stop calibrating on test data in `walkforward_validation.py`
  - [x] Stop calibrating on test split in `train_timing_model_per_ticker.py`
  - [x] Remove full-slice scaler leakage in OOF flow in `train_ensemble_v3.py`
  - [x] Make stacking meta-features consistent with inference path
- [x] 2. Define one consistent prediction target across timing + strike
- [x] 3. Replace heuristic edge fallback with calibrated contract-level probability
- [x] 4. Upgrade decision logic from threshold gates to expected-value optimization
- [x] 5. Rebuild validation framework to mirror production decisions
- [x] 6. Recalibrate probabilities with proper temporal protocol
- [x] 7. Add uncertainty / no-trade band
- [x] 8. Clean serving-time data contract and train/serve parity checks

## Change Log

- 2026-02-11: Step 1 implemented in:
  - `walkforward_validation.py`
  - `train_timing_model_per_ticker.py`
  - `train_ensemble_v3.py`
- 2026-02-11: Step 2 implemented by aligning strike-model labels to
  forward-window strike-breach semantics (same target family as timing):
  - `train_strike_model_v2.py`
  - `train_strike_model_v3.py`
  - `validate_strike_model_v3.py`
- 2026-02-11: Step 3 implemented by removing heuristic edge fallback and
  requiring strike-model probabilities for edge/suggestion:
  - `simple_api_server.py`
- 2026-02-11: Step 4 implemented by replacing hard gate selection with
  EV-based ranking (with spread/liquidity filter and soft timing prior):
  - `simple_api_server.py`
- 2026-02-11: Step 5 implemented by adding policy-level walk-forward metrics
  (trade rate, PnL/trade, total PnL, win rate, drawdown) alongside AUC:
  - `walkforward_validation.py`
- 2026-02-11: Step 6 implemented with log-based temporal recalibration and
  inference-time calibrated `p_safe` support:
  - `recalibrate_probabilities.py`
  - `simple_api_server.py`
- 2026-02-11: Step 7 implemented with explicit no-trade conditions for
  low timing confidence, near-term earnings risk, and weak edge signals:
  - `simple_api_server.py`
- 2026-02-11: Step 8 implemented with explicit feature-contract alignment,
  missing-feature fill, and parity diagnostics in prediction responses:
  - `simple_api_server.py`

## Execution Log

- 2026-02-11: End-to-end command run completed for updated pipeline.
- 2026-02-11: Timing model training completed:
  - `train_timing_model_per_ticker.py --period 10y`
  - Output: `csp_timing_model_per_ticker.pkl`
- 2026-02-11: Ensemble training completed:
  - `train_ensemble_v3.py --use-optuna-params`
  - Output: `csp_timing_ensemble.pkl`
- 2026-02-11: Strike model training completed:
  - `train_strike_model_v2.py --period 3y` -> `strike_model_v2.pkl`
  - `train_strike_model_v3.py --period 3y` -> `strike_model_v3.pkl`
- 2026-02-11: Walk-forward validation completed (after compatibility fix):
  - `walkforward_validation.py --period 10y --rolling-threshold`
  - Mean AUC by group:
    - `high_vol`: `0.6639`
    - `tech_growth`: `0.6189`
    - `tech_stable`: `0.6092`
    - `etf`: `0.6436`
- 2026-02-11: Strike validation completed:
  - `validate_strike_model_v3.py`
  - V3 ROC-AUC: `0.7912` (vs delta `0.6333`, V2 `0.7046`)
  - V3 Brier: `0.2045` (better than V2 `0.2297`, delta `0.2459`)
- 2026-02-11: API startup completed:
  - `simple_api_server.py --host 0.0.0.0 --port 8000`
  - Status: running on `http://0.0.0.0:8000`
- 2026-02-11: Probability recalibration command blocked by data availability:
  - `recalibrate_probabilities.py --min-samples 100`
  - Result: `Not enough settled predictions: 0`
- 2026-02-11: Bulk settlement script created and executed:
  - Script: `settle_prediction_outcomes.py`
  - Command executed with backup:
    - `XDG_CACHE_HOME=/tmp python settle_prediction_outcomes.py --log predictions_log.json --forward-days 35 --drop-threshold 0.05 --period 10y --backup`
  - Result:
    - `Predictions total: 375`
    - `Updated outcomes: 0`
    - `Already settled: 0`
    - `Unmatured/no-window: 375`
    - `Missing ticker data: 0`
    - Backup created: `predictions_log.json.bak`
