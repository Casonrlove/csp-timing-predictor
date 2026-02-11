"""
Optuna Hyperparameter Tuning for CSP Timing Ensemble V3
=======================================================
Runs 16 independent Optuna studies:
  4 groups × 2 VIX regimes × 2 base learners = 16 studies

Objective: maximize ROC-AUC via 3-fold expanding-window CV on training data.
Sampler: TPESampler  |  Pruner: MedianPruner
Trials:  200 per study  |  Timeout: 3600 s (1 hour) per study

Results saved to optuna_params_v3.json.
train_ensemble_v3.py loads this file automatically with --use-optuna-params.

Usage:
  python tune_ensemble_v3.py                    # full overnight run
  python tune_ensemble_v3.py --group etf        # single group (fast test)
  python tune_ensemble_v3.py --trials 50        # quick smoke test
  python tune_ensemble_v3.py --no-lgbm          # skip LightGBM studies
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("ERROR: optuna not installed. Run 'pip install optuna'")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from data_collector import CSPDataCollector
from feature_utils import apply_rolling_threshold

# Detect CUDA once at import time so objective functions can use it
def _cuda_available() -> bool:
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi'], capture_output=True)
        return r.returncode == 0
    except Exception:
        return False

USE_GPU = _cuda_available()
print(f"[tune] GPU available: {USE_GPU}")

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DEFAULT_TRIALS = 200
DEFAULT_TIMEOUT = 3600          # 1 hour per study
N_CV_FOLDS = 3                  # expanding-window folds within training data
VIX_BOUNDARY = 18.0
MIN_REGIME_SAMPLES = 300        # min samples for regime slice (lower than trainer for tuning)
OUTPUT_FILE = 'optuna_params_v3.json'

TICKER_GROUPS = {
    'high_vol':     ['TSLA', 'NVDA', 'AMD', 'MSTR', 'COIN', 'PLTR'],
    'tech_growth':  ['META', 'GOOGL', 'AMZN', 'NFLX', 'CRM'],
    'tech_stable':  ['AAPL', 'MSFT', 'V', 'MA'],
    'etf':          ['SPY', 'QQQ', 'IWM'],
}


# -------------------------------------------------------------------
# Data loading helpers
# -------------------------------------------------------------------

def load_group_data(group: str, tickers: list, period: str = '10y'):
    """Return (X_low, y_low, X_high, y_high, feature_cols)."""
    frames = []
    feature_cols = None
    for ticker in tickers:
        try:
            print(f"  {ticker}...", end=' ', flush=True)
            col = CSPDataCollector(ticker, period=period)
            df, cols = col.get_training_data()
            if feature_cols is None:
                feature_cols = cols
            df['ticker'] = ticker
            frames.append(df)
            print(f"ok ({len(df)})")
        except Exception as e:
            print(f"FAIL: {e}")

    if not frames:
        return None, None, None, None, None

    combined = pd.concat(frames, ignore_index=False).sort_index()
    combined = apply_rolling_threshold(combined, window=90, quantile=0.60)

    low = combined[combined['VIX'] < VIX_BOUNDARY]
    high = combined[combined['VIX'] >= VIX_BOUNDARY]

    def to_xy(df_slice, fallback_df):
        if len(df_slice) < MIN_REGIME_SAMPLES:
            df_slice = fallback_df
        X = df_slice[feature_cols].values
        y = df_slice['target'].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        return X_s, y

    X_low, y_low = to_xy(low, combined)
    X_high, y_high = to_xy(high, combined)

    return X_low, y_low, X_high, y_high, feature_cols


# -------------------------------------------------------------------
# Objective functions
# -------------------------------------------------------------------

def _cv_auc(X: np.ndarray, y: np.ndarray, build_fn, n_folds: int = N_CV_FOLDS) -> float:
    """3-fold expanding-window CV AUC (no test leakage)."""
    tscv = TimeSeriesSplit(n_splits=n_folds)
    aucs = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if len(np.unique(y_val)) < 2:
            continue
        model = build_fn()
        # XGBoost accepts verbose in fit(); LightGBM controls verbosity via constructor
        try:
            model.fit(X_tr, y_tr, verbose=False)
        except TypeError:
            model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, prob))
    return float(np.mean(aucs)) if aucs else 0.0


def xgb_objective(trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for XGBoost hyperparameter search."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 300, 1500),
        'max_depth':         trial.suggest_int('max_depth', 4, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 2.0),
        'scale_pos_weight':  spw,
        'eval_metric':       'auc',
        'tree_method':       'hist',
        'random_state':      42,
        'verbosity':         0,
    }
    # CPU multi-thread is faster than GPU for small datasets (<10K rows)
    params['n_jobs'] = -1

    def build():
        return xgb.XGBClassifier(**params)

    return _cv_auc(X, y, build)


def lgbm_objective(trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for LightGBM hyperparameter search."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    params = {
        'boosting_type':     'dart',
        'num_leaves':        trial.suggest_int('num_leaves', 20, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq':      5,
        'lambda_l1':         trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2':         trial.suggest_float('lambda_l2', 0.0, 2.0),
        'scale_pos_weight':  spw,
        'objective':         'binary',
        'metric':            'auc',
        'n_estimators':      500,
        'random_state':      42,
        'verbose':           -1,
    }
    # LightGBM pip package uses OpenCL (not CUDA) for GPU — run on CPU
    params['n_jobs'] = -1

    def build():
        return lgb.LGBMClassifier(**params)

    return _cv_auc(X, y, build)


# -------------------------------------------------------------------
# Main tuning loop
# -------------------------------------------------------------------

def run_study(
    study_name: str,
    objective_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    timeout: int,
) -> dict:
    """Run one Optuna study and return the best params dict."""
    print(f"\n  Study: {study_name}  (n={len(X)}, trials={n_trials}, timeout={timeout}s)")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective_fn(trial, X, y),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        n_jobs=1,        # single job to avoid data-race on shared yfinance cache
    )

    best = study.best_params
    best_auc = study.best_value
    print(f"  Best AUC: {best_auc:.4f}  Params: {best}")
    return best, best_auc


def main():
    if not OPTUNA_AVAILABLE:
        print("optuna is required. pip install optuna")
        return

    parser = argparse.ArgumentParser(description='Optuna tuning for CSP Ensemble V3')
    parser.add_argument('--period', default='10y')
    parser.add_argument('--group', default=None, help='Tune single group only')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS)
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument('--no-lgbm', action='store_true')
    parser.add_argument('--output', default=OUTPUT_FILE)
    args = parser.parse_args()

    groups_to_tune = {args.group: TICKER_GROUPS[args.group]} if args.group else TICKER_GROUPS
    if args.group and args.group not in TICKER_GROUPS:
        print(f"Unknown group: {args.group}")
        return

    run_lgbm = LGBM_AVAILABLE and not args.no_lgbm

    print('='*70)
    print('OPTUNA HYPERPARAMETER TUNING — CSP ENSEMBLE V3')
    print('='*70)
    print(f"Groups:  {list(groups_to_tune.keys())}")
    print(f"LightGBM: {'yes' if run_lgbm else 'no'}")
    print(f"Trials per study: {args.trials}  |  Timeout: {args.timeout}s")

    # Load existing params (resume-friendly)
    all_params: dict = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_params = json.load(f)
        print(f"Resuming from {args.output} ({len(all_params)} existing entries)")

    for group, tickers in groups_to_tune.items():
        print(f"\n{'='*70}")
        print(f"GROUP: {group.upper()}")
        print('='*70)

        X_low, y_low, X_high, y_high, feature_cols = load_group_data(
            group, tickers, period=args.period
        )
        if X_low is None:
            print(f"  No data for {group}, skipping.")
            continue

        for regime, X, y in [('low_vix', X_low, y_low), ('high_vix', X_high, y_high)]:
            print(f"\n  Regime: {regime}  X.shape={X.shape}")

            # XGBoost study
            xgb_key = f'{group}_{regime}_xgb'
            if xgb_key not in all_params:
                best_xgb, auc_xgb = run_study(
                    xgb_key, xgb_objective, X, y, args.trials, args.timeout
                )
                all_params[xgb_key] = best_xgb
                all_params[f'{xgb_key}_auc'] = auc_xgb
                # Save after each study (checkpoint)
                with open(args.output, 'w') as f:
                    json.dump(all_params, f, indent=2)
                print(f"  Saved checkpoint to {args.output}")
            else:
                print(f"  {xgb_key} already tuned (AUC={all_params.get(xgb_key+'_auc', '?')}), skipping.")

            # LightGBM study
            if run_lgbm:
                lgbm_key = f'{group}_{regime}_lgbm'
                if lgbm_key not in all_params:
                    best_lgbm, auc_lgbm = run_study(
                        lgbm_key, lgbm_objective, X, y, args.trials, args.timeout
                    )
                    all_params[lgbm_key] = best_lgbm
                    all_params[f'{lgbm_key}_auc'] = auc_lgbm
                    with open(args.output, 'w') as f:
                        json.dump(all_params, f, indent=2)
                    print(f"  Saved checkpoint to {args.output}")
                else:
                    print(f"  {lgbm_key} already tuned, skipping.")

    # Final save
    with open(args.output, 'w') as f:
        json.dump(all_params, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ TUNING COMPLETE — {len(all_params)} entries saved to {args.output}")
    print('='*70)

    # Summary table
    print(f"\n{'Study':<45} {'AUC':>8}")
    print('-' * 55)
    for k, v in all_params.items():
        if k.endswith('_auc'):
            study_name = k[:-4]
            print(f"  {study_name:<43} {v:>8.4f}")


if __name__ == '__main__':
    main()
