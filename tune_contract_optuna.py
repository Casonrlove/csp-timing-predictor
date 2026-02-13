"""
Optuna Hyperparameter Tuning for Contract Breach Model
======================================================
Tunes XGB + LGBM params on contract-level breach data using
contract-aware time splits (no date leakage).

Focused on a single group (default: high_vol) for targeted optimization.

Usage:
  python tune_contract_optuna.py                      # high_vol, 60 trials
  python tune_contract_optuna.py --group tech_growth   # different group
  python tune_contract_optuna.py --n-trials 100        # more trials
"""

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("pip install optuna")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from data_collector import CSPDataCollector
from feature_utils import (
    CONTRACT_FEATURE_COLS,
    contract_aware_time_split,
    create_breach_target,
)

warnings.filterwarnings("ignore", category=UserWarning)

VIX_BOUNDARY = 18.0

TICKER_GROUPS = {
    "high_vol":     ["TSLA", "NVDA", "AMD", "MSTR", "COIN", "PLTR"],
    "tech_growth":  ["META", "GOOGL", "AMZN", "NFLX", "CRM"],
    "tech_stable":  ["AAPL", "MSFT", "V", "MA"],
    "etf":          ["SPY", "QQQ", "IWM"],
}

OUTPUT_FILE = "optuna_contract_params.json"


def collect_contract_data(group: str, period: str = "10y"):
    """Collect multi-strike expanded data for a single group."""
    tickers = TICKER_GROUPS[group]
    frames = []
    market_feature_cols = None

    for ticker in tickers:
        print(f"  Fetching {ticker}...", end=" ", flush=True)
        try:
            col = CSPDataCollector(ticker, period=period)
            col.fetch_data()
            if len(col.data) < 250:
                print(f"SKIP ({len(col.data)} days)")
                continue
            col.calculate_technical_indicators()
            col.create_target_variable()
            _, mcols = col.prepare_features()
            if market_feature_cols is None:
                market_feature_cols = mcols
            expanded = col.create_contract_targets()
            expanded["ticker"] = ticker
            frames.append(expanded)
            print(f"ok ({len(expanded)} rows)")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not frames:
        raise RuntimeError(f"No data for group {group}")

    df = pd.concat(frames, ignore_index=False).sort_index()
    feature_cols = list(market_feature_cols) + CONTRACT_FEATURE_COLS
    return df, feature_cols


def _compute_sample_weights(df_slice, halflife=2.0):
    dates = pd.to_datetime(df_slice.index)
    days_ago = (dates.max() - dates).days.values
    halflife_days = halflife * 365.25
    return np.exp(-np.log(2) * days_ago / halflife_days)


def evaluate_params(df_regime, feature_cols, xgb_params, lgbm_params, n_folds=4):
    """Run OOF stacking CV and return (auc, brier).

    Uses 4 folds for speed during tuning (vs 5 in production).
    """
    df_regime = create_breach_target(df_regime)
    for col in feature_cols:
        if col not in df_regime.columns:
            df_regime[col] = 0.0

    X = df_regime[feature_cols].values
    y = df_regime["target"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sw = _compute_sample_weights(df_regime)

    splits = contract_aware_time_split(df_regime, n_splits=n_folds)

    oof_xgb = np.zeros(len(X))
    oof_lgbm = np.zeros(len(X))
    valid_mask = np.zeros(len(X), dtype=bool)

    for tr_idx, val_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])
        y_tr, y_val = y[tr_idx], y[val_idx]

        if len(np.unique(y_val)) < 2:
            continue
        valid_mask[val_idx] = True

        xm = xgb.XGBClassifier(**xgb_params)
        xm.fit(X_tr, y_tr, sample_weight=sw[tr_idx], verbose=False)
        oof_xgb[val_idx] = xm.predict_proba(X_val)[:, 1]

        if LGBM_AVAILABLE:
            lm = lgb.LGBMClassifier(**lgbm_params)
            lm.fit(X_tr, y_tr, sample_weight=sw[tr_idx])
            oof_lgbm[val_idx] = lm.predict_proba(X_val)[:, 1]
        else:
            oof_lgbm[val_idx] = oof_xgb[val_idx]

    if valid_mask.sum() < 100:
        return 0.5, 0.5

    avg_prob = (oof_xgb[valid_mask] + oof_lgbm[valid_mask]) / 2.0
    auc = roc_auc_score(y[valid_mask], avg_prob)
    brier = brier_score_loss(y[valid_mask], avg_prob)
    return auc, brier


def create_objective(df_regime, feature_cols, regime_key):
    """Create an Optuna objective that tunes both XGB and LGBM jointly."""

    def objective(trial):
        # XGB hyperparameters
        xgb_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 800),
            "max_depth": trial.suggest_int("xgb_max_depth", 4, 10),
            "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("xgb_mcw", 1, 15),
            "gamma": trial.suggest_float("xgb_gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("xgb_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_lambda", 0.1, 3.0),
            "random_state": 42,
            "eval_metric": "auc",
            "tree_method": "hist",
            "n_jobs": -1,
        }

        neg = (df_regime["strike_breached"] == 0).sum()
        pos = (df_regime["strike_breached"] == 1).sum()
        spw = neg / pos if pos > 0 else 1.0
        xgb_params["scale_pos_weight"] = spw

        # LGBM hyperparameters
        lgbm_params = {
            "boosting_type": "dart",
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 15, 200),
            "min_child_samples": trial.suggest_int("lgbm_mcs", 10, 60),
            "learning_rate": trial.suggest_float("lgbm_lr", 0.005, 0.15, log=True),
            "feature_fraction": trial.suggest_float("lgbm_ff", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("lgbm_bf", 0.6, 1.0),
            "bagging_freq": 5,
            "lambda_l1": trial.suggest_float("lgbm_l1", 0.01, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lgbm_l2", 0.1, 2.0),
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 300, 800),
            "objective": "binary",
            "metric": "auc",
            "scale_pos_weight": spw,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        auc, brier = evaluate_params(df_regime.copy(), feature_cols, xgb_params, lgbm_params)

        # Combined objective: maximize AUC, minimize Brier
        # Weight AUC more heavily since it's the primary metric
        score = 0.7 * auc + 0.3 * (1.0 - brier)
        return score

    return objective


def extract_best_params(study):
    """Extract XGB and LGBM params from best trial."""
    bp = study.best_trial.params

    xgb_params = {
        "n_estimators": bp["xgb_n_estimators"],
        "max_depth": bp["xgb_max_depth"],
        "learning_rate": bp["xgb_lr"],
        "subsample": bp["xgb_subsample"],
        "colsample_bytree": bp["xgb_colsample"],
        "min_child_weight": bp["xgb_mcw"],
        "gamma": bp["xgb_gamma"],
        "reg_alpha": bp["xgb_alpha"],
        "reg_lambda": bp["xgb_lambda"],
    }

    lgbm_params = {
        "num_leaves": bp["lgbm_num_leaves"],
        "min_child_samples": bp["lgbm_mcs"],
        "learning_rate": bp["lgbm_lr"],
        "feature_fraction": bp["lgbm_ff"],
        "bagging_fraction": bp["lgbm_bf"],
        "lambda_l1": bp["lgbm_l1"],
        "lambda_l2": bp["lgbm_l2"],
        "n_estimators": bp["lgbm_n_estimators"],
    }

    return xgb_params, lgbm_params


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for contract model")
    parser.add_argument("--group", default="high_vol", choices=list(TICKER_GROUPS.keys()))
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--period", default="10y")
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    print("=" * 70)
    print(f"OPTUNA CONTRACT MODEL TUNING — {args.group.upper()}")
    print("=" * 70)

    # Collect data
    t0 = time.time()
    df, feature_cols = collect_contract_data(args.group, args.period)
    print(f"\nCollected {len(df)} rows in {time.time()-t0:.1f}s")

    # Split by VIX regime
    low_df = df[df["VIX"] < VIX_BOUNDARY].copy()
    high_df = df[df["VIX"] >= VIX_BOUNDARY].copy()

    results = {}

    for regime, regime_df in [("low_vix", low_df), ("high_vix", high_df)]:
        regime_key = f"{args.group}_{regime}"
        n = len(regime_df)
        breach_rate = regime_df["strike_breached"].mean()
        print(f"\n{'='*50}")
        print(f"TUNING: {regime_key}  ({n} rows, breach rate={breach_rate:.1%})")
        print(f"{'='*50}")

        objective = create_objective(regime_df, feature_cols, regime_key)

        study = optuna.create_study(
            direction="maximize",
            study_name=regime_key,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        t1 = time.time()
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        elapsed = time.time() - t1

        xgb_params, lgbm_params = extract_best_params(study)

        # Evaluate best params with full 5-fold to get reliable estimate
        auc, brier = evaluate_params(regime_df.copy(), feature_cols,
                                      {**xgb_params, "random_state": 42, "eval_metric": "auc",
                                       "tree_method": "hist", "n_jobs": -1,
                                       "scale_pos_weight": (regime_df["strike_breached"]==0).sum() / max((regime_df["strike_breached"]==1).sum(), 1)},
                                      {**lgbm_params, "boosting_type": "dart", "bagging_freq": 5,
                                       "objective": "binary", "metric": "auc",
                                       "scale_pos_weight": (regime_df["strike_breached"]==0).sum() / max((regime_df["strike_breached"]==1).sum(), 1),
                                       "random_state": 42, "verbose": -1, "n_jobs": -1},
                                      n_folds=5)

        print(f"\n  Best score: {study.best_value:.4f}  (in {elapsed:.0f}s)")
        print(f"  5-fold AUC: {auc:.4f}  Brier: {brier:.4f}")
        print(f"  XGB params: {json.dumps(xgb_params, indent=2)}")
        print(f"  LGBM params: {json.dumps(lgbm_params, indent=2)}")

        results[f"{regime_key}_xgb"] = xgb_params
        results[f"{regime_key}_xgb_auc"] = auc
        results[f"{regime_key}_lgbm"] = lgbm_params
        results[f"{regime_key}_lgbm_auc"] = auc
        results[f"{regime_key}_brier"] = brier

    # Save results
    # Merge with existing params if file exists
    existing = {}
    try:
        with open(args.output) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    existing.update(results)
    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Saved tuned params to {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
