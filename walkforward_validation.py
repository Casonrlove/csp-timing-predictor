"""
Walk-Forward Validation for CSP Timing Models

Tests how well the per-ticker grouped model generalizes across time by using
an expanding training window with 1-year test folds:

  Fold 1: Train 2014-2018, Test 2019
  Fold 2: Train 2014-2019, Test 2020
  Fold 3: Train 2014-2020, Test 2021
  Fold 4: Train 2014-2021, Test 2022
  Fold 5: Train 2014-2022, Test 2023

Reports per-fold and overall ROC-AUC to detect overfitting or regime drift.

Usage:
  python walkforward_validation.py
  python walkforward_validation.py --group high_vol
  python walkforward_validation.py --period 10y --min-train-years 4
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from data_collector import CSPDataCollector
from feature_utils import apply_rolling_threshold, calculate_group_threshold

warnings.filterwarnings("ignore", category=UserWarning)

# Ticker groups (must match train_timing_model_per_ticker.py)
TICKER_GROUPS = {
    "high_vol":    ["TSLA", "NVDA", "AMD", "MSTR", "COIN", "PLTR"],
    "tech_growth": ["META", "GOOGL", "AMZN", "NFLX", "CRM"],
    "tech_stable": ["AAPL", "MSFT", "V", "MA"],
    "etf":         ["SPY", "QQQ", "IWM"],
}

# Default params matching train_ensemble_v3.py
DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": -1,
}

DEFAULT_LGBM_PARAMS = {
    "boosting_type": "dart",
    "num_leaves": 63,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "n_estimators": 500,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

VIX_BOUNDARY = 18.0
MIN_REGIME_SAMPLES = 400
N_META_FOLDS = 5
RECENCY_HALFLIFE = 2.0

META_FEATURE_NAMES = ["xgb_prob", "lgbm_prob", "VIX", "VIX_Rank", "Regime_Trend"]

OPTUNA_PARAMS_FILE = "optuna_params_v3.json"


def _load_optuna_params() -> dict:
    """Load Optuna-tuned hyperparameters if available."""
    if os.path.exists(OPTUNA_PARAMS_FILE):
        try:
            with open(OPTUNA_PARAMS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _compute_sample_weights(df_slice: pd.DataFrame, halflife: float = RECENCY_HALFLIFE) -> np.ndarray:
    """Exponential recency weighting."""
    if halflife <= 0:
        return np.ones(len(df_slice))
    dates = pd.to_datetime(df_slice.index)
    days_ago = (dates.max() - dates).days.values
    halflife_days = halflife * 365.25
    return np.exp(-np.log(2) * days_ago / halflife_days)


def collect_group_data(group_name: str, tickers: list[str], period: str = "10y") -> tuple[pd.DataFrame | None, list[str] | None]:
    """Collect and combine data for all tickers in a group.

    Returns (combined_df, feature_cols) — feature_cols comes from the first
    successful ticker so we don't need a separate fetch later.
    """
    frames = []
    feature_cols = None
    for ticker in tickers:
        try:
            print(f"  Fetching {ticker}...", end=" ", flush=True)
            collector = CSPDataCollector(ticker, period=period)
            df, cols = collector.get_training_data()
            if feature_cols is None:
                feature_cols = cols
            df["ticker"] = ticker
            frames.append(df)
            print(f"ok ({len(df)} rows)")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not frames:
        return None, None
    return pd.concat(frames, ignore_index=False).sort_index(), feature_cols


# apply_rolling_threshold and calculate_group_threshold are imported from feature_utils


def run_walk_forward(
    group_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_years: int = 4,
    test_years: int = 1,
    use_rolling_threshold: bool = True,
    rolling_window: int = 5,
    simple_mode: bool = False,
) -> dict:
    """
    Run walk-forward validation on a pre-collected group DataFrame.

    Returns a dict with per-fold and aggregate metrics.
    """
    # Drop rows without a valid target
    df = df.dropna(subset=["Max_Drawdown_35D"] + feature_cols)
    df = df.sort_index()

    # Determine fold boundaries using calendar years
    years = sorted(df.index.year.unique())
    if len(years) < min_train_years + test_years:
        print(f"  Not enough years ({len(years)}) for walk-forward — need at least {min_train_years + test_years}")
        return {}

    if use_rolling_threshold:
        # Per-ticker rolling 90D threshold stabilises the positive rate across years.
        # Each row is labelled relative to the recent local regime, not a single
        # full-history cut-point that swings wildly between bull and bear markets.
        df = apply_rolling_threshold(df, window=90, quantile=0.60)
        threshold_desc = "rolling-90D P60"
    else:
        # Fixed full-history threshold (original behaviour)
        threshold = calculate_group_threshold(df)
        df["target"] = (df["Max_Drawdown_35D"] > threshold).astype(int)
        threshold_desc = f"fixed {threshold:+.1f}%"

    print(f"  Threshold: {threshold_desc}  |  overall positive rate: {df['target'].mean():.1%}")

    fold_results = []

    def _policy_metrics(y_true: np.ndarray, y_prob: np.ndarray, prices: np.ndarray, atr_pct: np.ndarray) -> dict:
        """
        Approximate production policy metrics using an EV-based trade rule.
        Mirrors the current server policy style (EV + soft timing prior) with
        a synthetic premium model for historical backtests.
        """
        if len(y_true) == 0:
            return {
                "n_trades": 0,
                "trade_rate": 0.0,
                "total_pnl": 0.0,
                "pnl_per_trade": 0.0,
                "win_rate": 0.0,
                "avg_risk_adj_ev": 0.0,
                "max_drawdown": 0.0,
            }

        strike = prices * 0.95
        # Synthetic premium proxy (bounded): higher ATR implies richer premium.
        atr_pct = np.nan_to_num(atr_pct, nan=0.0)
        premium_rate = np.clip(0.01 + (atr_pct / 100.0) * 0.15, 0.005, 0.04)
        premium = strike * premium_rate

        p_assign = 1.0 - y_prob
        expected_assignment_loss = p_assign * strike * 0.05
        expected_ev = premium - expected_assignment_loss
        timing_multiplier = 0.75 + 0.5 * y_prob
        risk_adj_ev = expected_ev * timing_multiplier

        trade_mask = risk_adj_ev > 0
        n_trades = int(trade_mask.sum())

        realized_trade_pnl = np.where(y_true == 1, premium, premium - (strike * 0.05))
        realized_pnl = np.where(trade_mask, realized_trade_pnl, 0.0)

        equity_curve = np.cumsum(realized_pnl)
        if len(equity_curve) > 0:
            running_max = np.maximum.accumulate(equity_curve)
            max_drawdown = float(np.max(running_max - equity_curve))
        else:
            max_drawdown = 0.0

        wins = int(np.sum(trade_mask & (realized_trade_pnl > 0)))
        win_rate = (wins / n_trades) if n_trades > 0 else 0.0

        return {
            "n_trades": n_trades,
            "trade_rate": float(n_trades / len(y_true)),
            "total_pnl": float(np.sum(realized_pnl)),
            "pnl_per_trade": float(np.sum(realized_pnl) / n_trades) if n_trades > 0 else 0.0,
            "win_rate": float(win_rate),
            "avg_risk_adj_ev": float(np.mean(risk_adj_ev[trade_mask])) if n_trades > 0 else 0.0,
            "max_drawdown": max_drawdown,
        }
    # Load Optuna params if available
    optuna_params = _load_optuna_params()

    # Walk-forward: train on rolling (or expanding) window, test on next year(s)
    for i in range(min_train_years - 1, len(years) - test_years):
        train_end_year = years[i]
        test_start_year = years[i + 1]
        test_end_year = years[min(i + test_years, len(years) - 1)]

        if rolling_window > 0:
            rolling_start = max(years[0], train_end_year - rolling_window + 1)
            train_mask = (df.index.year >= rolling_start) & (df.index.year <= train_end_year)
        else:
            train_mask = df.index.year <= train_end_year
        test_mask = (df.index.year >= test_start_year) & (df.index.year <= test_end_year)

        df_train = df.loc[train_mask]
        df_test = df.loc[test_mask]
        X_test = df_test[feature_cols].values
        y_test = df_test["target"].values

        if len(df_train) < 200 or len(X_test) < 50:
            continue
        if len(np.unique(y_test)) < 2:
            continue

        if simple_mode:
            # ---- SIMPLE MODE: Single XGBoost + isotonic calibration (Round 2 style) ----
            X_train = df_train[feature_cols].values
            y_train = df_train["target"].values
            sw = _compute_sample_weights(df_train)
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            spw = neg / pos if pos > 0 else 1.0

            simple_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": spw,
                "random_state": 42,
                "eval_metric": "auc",
                "tree_method": "hist",
                "n_jobs": -1,
            }

            model = xgb.XGBClassifier(**simple_params)
            model.fit(X_train, y_train, sample_weight=sw, verbose=False)

            # Isotonic calibration using held-out tail of training data
            cal_start = int(len(X_train) * 0.8)
            cal_X = X_train[cal_start:]
            cal_y = y_train[cal_start:]
            if len(cal_X) >= 50 and len(np.unique(cal_y)) > 1:
                cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
                cal.fit(cal_X, cal_y)
                y_prob = cal.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict_proba(X_test)[:, 1]

        else:
            # ---- ENSEMBLE MODE: Full XGB+LGBM stacking (Round 3 style) ----
            # Split training by VIX regime
            regime_models = {}  # {regime: {'xgb', 'lgbm', 'meta', 'meta_scaler', 'scaler'}}

            for regime, regime_df in [
                ("low_vix", df_train[df_train["VIX"] < VIX_BOUNDARY]),
                ("high_vix", df_train[df_train["VIX"] >= VIX_BOUNDARY]),
            ]:
                rdf = regime_df.copy()
                # Fallback to full training data if regime slice is too small
                if len(rdf) < MIN_REGIME_SAMPLES:
                    rdf = df_train.copy()

                X_r = rdf[feature_cols].values
                y_r = rdf["target"].values
                sw = _compute_sample_weights(rdf)
                neg = (y_r == 0).sum()
                pos = (y_r == 1).sum()
                spw = neg / pos if pos > 0 else 1.0

                # Build Optuna-aware params
                model_key = f"{group_name}_{regime}"
                xgb_p = dict(DEFAULT_XGB_PARAMS)
                xgb_p["scale_pos_weight"] = spw
                op_xgb = optuna_params.get(f"{model_key}_xgb", {})
                if op_xgb.get("best_params"):
                    xgb_p.update(op_xgb["best_params"])

                lgbm_p = dict(DEFAULT_LGBM_PARAMS)
                lgbm_p["scale_pos_weight"] = spw
                op_lgbm = optuna_params.get(f"{model_key}_lgbm", {})
                if op_lgbm.get("best_params"):
                    lgbm_p.update(op_lgbm["best_params"])

                # Meta-feature column indices
                meta_feat_idx = []
                for mf in ["VIX", "VIX_Rank", "Regime_Trend"]:
                    meta_feat_idx.append(feature_cols.index(mf) if mf in feature_cols else -1)

                # 5-fold OOF stacking
                tscv = TimeSeriesSplit(n_splits=N_META_FOLDS)
                oof_xgb = np.zeros(len(X_r))
                oof_lgbm = np.zeros(len(X_r))
                oof_meta_feats = [np.zeros(len(X_r)) for _ in meta_feat_idx]
                valid_mask = np.zeros(len(X_r), dtype=bool)

                for tr_idx, val_idx in tscv.split(X_r):
                    fold_scaler = StandardScaler()
                    X_tr = fold_scaler.fit_transform(X_r[tr_idx])
                    X_val = fold_scaler.transform(X_r[val_idx])
                    y_tr = y_r[tr_idx]
                    sw_tr = sw[tr_idx]

                    if len(np.unique(y_r[val_idx])) < 2:
                        continue
                    valid_mask[val_idx] = True

                    xm = xgb.XGBClassifier(**xgb_p)
                    xm.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
                    oof_xgb[val_idx] = xm.predict_proba(X_val)[:, 1]

                    if LGBM_AVAILABLE:
                        lm = lgb.LGBMClassifier(**lgbm_p)
                        lm.fit(X_tr, y_tr, sample_weight=sw_tr)
                        oof_lgbm[val_idx] = lm.predict_proba(X_val)[:, 1]
                    else:
                        oof_lgbm[val_idx] = oof_xgb[val_idx]

                    for j, idx in enumerate(meta_feat_idx):
                        if idx >= 0:
                            oof_meta_feats[j][val_idx] = X_r[val_idx, idx]

                if valid_mask.sum() < 100:
                    continue

                # Fit meta-learner on scaled OOF
                meta_cols = [oof_xgb.reshape(-1, 1), oof_lgbm.reshape(-1, 1)]
                for c in oof_meta_feats:
                    meta_cols.append(c.reshape(-1, 1))
                meta_X = np.hstack(meta_cols)

                meta_scaler = StandardScaler()
                meta_X_s = np.copy(meta_X)
                meta_X_s[valid_mask] = meta_scaler.fit_transform(meta_X[valid_mask])

                meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                meta_lr.fit(meta_X_s[valid_mask], y_r[valid_mask])

                # Train final base learners on full regime slice
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_r)

                final_xgb = xgb.XGBClassifier(**xgb_p)
                final_xgb.fit(X_scaled, y_r, sample_weight=sw, verbose=False)

                final_lgbm = None
                if LGBM_AVAILABLE:
                    final_lgbm = lgb.LGBMClassifier(**lgbm_p)
                    final_lgbm.fit(X_scaled, y_r, sample_weight=sw)

                regime_models[regime] = {
                    "xgb": final_xgb,
                    "lgbm": final_lgbm,
                    "meta": meta_lr,
                    "meta_scaler": meta_scaler,
                    "scaler": scaler,
                    "meta_feat_idx": meta_feat_idx,
                }

            if not regime_models:
                continue

            # ---- Predict on test fold through full ensemble stack ----
            y_prob = np.zeros(len(X_test))
            vix_test = df_test["VIX"].values if "VIX" in df_test.columns else np.full(len(X_test), 15.0)

            for row_i in range(len(X_test)):
                regime = "low_vix" if vix_test[row_i] < VIX_BOUNDARY else "high_vix"
                if regime not in regime_models:
                    regime = list(regime_models.keys())[0]  # fallback

                rm = regime_models[regime]
                x_row = X_test[row_i:row_i+1]
                x_scaled = rm["scaler"].transform(x_row)

                xgb_p_val = rm["xgb"].predict_proba(x_scaled)[0, 1]
                lgbm_p_val = rm["lgbm"].predict_proba(x_scaled)[0, 1] if rm["lgbm"] else xgb_p_val

                meta_row = [xgb_p_val, lgbm_p_val]
                for idx in rm["meta_feat_idx"]:
                    meta_row.append(float(x_row[0, idx]) if idx >= 0 else 0.0)
                meta_arr = np.array([meta_row])
                meta_arr = rm["meta_scaler"].transform(meta_arr)
                y_prob[row_i] = rm["meta"].predict_proba(meta_arr)[0, 1]

        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)

        prices_test = df_test["Close"].values if "Close" in df_test.columns else np.full(len(y_test), 100.0)
        atr_test = df_test["ATR_Pct"].values if "ATR_Pct" in df_test.columns else np.zeros(len(y_test))
        policy = _policy_metrics(y_test, y_prob, prices_test, atr_test)

        pos_rate = y_test.mean()
        n_train = len(df_train)

        fold_results.append({
            "fold": len(fold_results) + 1,
            "train_through": train_end_year,
            "test_year": f"{test_start_year}" if test_years == 1 else f"{test_start_year}-{test_end_year}",
            "train_samples": n_train,
            "test_samples": len(X_test),
            "positive_rate": pos_rate,
            "roc_auc": auc,
            "accuracy": acc,
            "n_trades": policy["n_trades"],
            "trade_rate": policy["trade_rate"],
            "pnl_per_trade": policy["pnl_per_trade"],
            "total_pnl": policy["total_pnl"],
            "policy_win_rate": policy["win_rate"],
            "policy_ev": policy["avg_risk_adj_ev"],
            "policy_max_dd": policy["max_drawdown"],
        })

        print(
            f"  Fold {fold_results[-1]['fold']:>2}  "
            f"train≤{train_end_year}  test={fold_results[-1]['test_year']}  "
            f"n_train={n_train:>5}  n_test={len(X_test):>4}  "
            f"AUC={auc:.4f}  Acc={acc:.4f}  pos={pos_rate:.1%}  "
            f"trades={policy['n_trades']:>3}  pnl/trade={policy['pnl_per_trade']:+.3f}"
        )

    return {
        "group": group_name,
        "threshold": threshold_desc,
        "folds": fold_results,
        "mean_auc": float(np.mean([f["roc_auc"] for f in fold_results])) if fold_results else 0.0,
        "std_auc": float(np.std([f["roc_auc"] for f in fold_results])) if fold_results else 0.0,
        "min_auc": float(np.min([f["roc_auc"] for f in fold_results])) if fold_results else 0.0,
        "max_auc": float(np.max([f["roc_auc"] for f in fold_results])) if fold_results else 0.0,
        "mean_policy_pnl_per_trade": float(np.mean([f["pnl_per_trade"] for f in fold_results])) if fold_results else 0.0,
        "mean_policy_trade_rate": float(np.mean([f["trade_rate"] for f in fold_results])) if fold_results else 0.0,
        "mean_policy_win_rate": float(np.mean([f["policy_win_rate"] for f in fold_results])) if fold_results else 0.0,
        "total_policy_pnl": float(np.sum([f["total_pnl"] for f in fold_results])) if fold_results else 0.0,
        "worst_policy_max_dd": float(np.max([f["policy_max_dd"] for f in fold_results])) if fold_results else 0.0,
        "trend": _compute_trend([f["roc_auc"] for f in fold_results]),
    }


def _compute_trend(aucs: list[float]) -> str:
    """Returns 'improving', 'stable', or 'degrading' based on linear slope."""
    if len(aucs) < 3:
        return "insufficient data"
    x = np.arange(len(aucs))
    slope = np.polyfit(x, aucs, 1)[0]
    if slope > 0.005:
        return f"improving (+{slope*len(aucs):.3f} over {len(aucs)} folds)"
    elif slope < -0.005:
        return f"degrading ({slope*len(aucs):.3f} over {len(aucs)} folds)"
    else:
        return f"stable (slope={slope:.4f})"


def print_summary(all_results: list[dict]) -> None:
    """Print a formatted summary across all groups."""
    print(f"\n{'='*80}")
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("="*80)
    print(
        f"\n{'Group':<15} {'Threshold':<18} {'Mean AUC':>10} {'Std AUC':>9} "
        f"{'PnL/Trade':>10} {'Trade%':>8} {'Win%':>8} {'TotalPnL':>10} {'Trend'}"
    )
    print("-" * 122)
    for r in all_results:
        if not r.get("folds"):
            continue
        print(
            f"{r['group']:<15} {r['threshold']:<17}  "
            f"{r['mean_auc']:>9.4f}  {r['std_auc']:>8.4f}  "
            f"{r['mean_policy_pnl_per_trade']:>+9.3f} {r['mean_policy_trade_rate']*100:>7.1f}% "
            f"{r['mean_policy_win_rate']*100:>7.1f}% {r['total_policy_pnl']:>+9.2f}  {r['trend']}"
        )

    print("\nInterpretation:")
    print("  AUC > 0.65 = good predictive power")
    print("  PnL/Trade  = realized backtest utility under EV policy proxy")
    print("  Trade%     = how selective the policy is")
    print("  Std AUC    = stability (< 0.05 = stable across market regimes)")
    print("  Trend      = is performance improving or degrading over time?")
    print("  If mean_auc >> min_auc, the model is overfit to certain regimes")

    # Per-year cross-group view
    year_aucs: dict[str, list[float]] = {}
    for r in all_results:
        for f in r.get("folds", []):
            year_aucs.setdefault(f["test_year"], []).append(f["roc_auc"])

    if year_aucs:
        print(f"\nPer-Year Average AUC (across groups):")
        print(f"  {'Year':<8} {'Avg AUC':>9} {'N Groups':>9}")
        print(f"  {'-'*30}")
        for year in sorted(year_aucs):
            vals = year_aucs[year]
            print(f"  {year:<8} {np.mean(vals):>9.4f} {len(vals):>9}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation for CSP Timing Models")
    parser.add_argument("--period", type=str, default="10y",
                        help="Data period: 5y, 10y (default: 10y)")
    parser.add_argument("--group", type=str, default=None,
                        help="Only validate a specific group (default: all)")
    parser.add_argument("--min-train-years", type=int, default=4,
                        help="Minimum years in training window (default: 4)")
    parser.add_argument("--rolling-threshold", dest="rolling_threshold",
                        action="store_true", default=True,
                        help="Use per-ticker rolling 90D threshold (default: on)")
    parser.add_argument("--no-rolling-threshold", dest="rolling_threshold",
                        action="store_false",
                        help="Use fixed full-history threshold (original behaviour)")
    parser.add_argument("--rolling-window", type=int, default=0,
                        help="Rolling training window in years (default: 0 = expanding)")
    parser.add_argument("--simple", action="store_true", default=False,
                        help="Use simple single-XGBoost + isotonic (Round 2 style) instead of full ensemble")
    parser.add_argument("--vix-regime", type=str, default="all",
                        choices=["all", "low", "mid", "high"],
                        help="Filter by VIX regime: low (<18), mid (18–28), high (>28), all (default)")
    args = parser.parse_args()

    groups_to_run = {args.group: TICKER_GROUPS[args.group]} if args.group else TICKER_GROUPS
    if args.group and args.group not in TICKER_GROUPS:
        print(f"Unknown group '{args.group}'. Choose from: {list(TICKER_GROUPS.keys())}")
        return

    print("="*80)
    print("WALK-FORWARD VALIDATION")
    print("="*80)
    window_desc = f"rolling {args.rolling_window}yr" if args.rolling_window > 0 else "expanding"
    print(f"Period:          {args.period}")
    print(f"Min train years: {args.min_train_years}")
    print(f"Training window: {window_desc}")
    print(f"Groups:          {list(groups_to_run.keys())}")
    print(f"Threshold:       {'rolling 90D P60' if args.rolling_threshold else 'fixed full-history P60'}")
    print(f"Model:           {'simple (XGB + isotonic)' if args.simple else 'ensemble (XGB + LGBM + stacking)'}")
    print(f"VIX regime:      {args.vix_regime}")

    all_results = []

    for group_name, tickers in groups_to_run.items():
        print(f"\n{'='*80}")
        print(f"GROUP: {group_name.upper()}  ({', '.join(tickers)})")
        print("="*80)

        t0 = time.time()
        df, feature_cols = collect_group_data(group_name, tickers, period=args.period)
        if df is None or feature_cols is None:
            print(f"  No data collected for {group_name}, skipping.")
            continue

        elapsed = time.time() - t0
        print(f"\n  Data collected in {elapsed:.1f}s.")

        # Apply VIX regime filter if requested
        if args.vix_regime != "all":
            if "VIX" not in df.columns:
                print(f"  WARNING: VIX column not found — cannot filter by regime. Running on full data.")
            else:
                vix_filters = {
                    "low":  df["VIX"] < 18,
                    "mid":  (df["VIX"] >= 18) & (df["VIX"] <= 28),
                    "high": df["VIX"] > 28,
                }
                before = len(df)
                df = df[vix_filters[args.vix_regime]]
                print(f"  VIX regime '{args.vix_regime}': {before} → {len(df)} rows "
                      f"({len(df)/before:.0%} of data, "
                      f"VIX range {df['VIX'].min():.1f}–{df['VIX'].max():.1f})")

        print(f"  Running walk-forward folds...")
        result = run_walk_forward(
            group_name, df, feature_cols,
            min_train_years=args.min_train_years,
            use_rolling_threshold=args.rolling_threshold,
            rolling_window=args.rolling_window,
            simple_mode=args.simple,
        )

        if result:
            all_results.append(result)
            print(f"\n  Group {group_name}: mean AUC={result['mean_auc']:.4f} ± {result['std_auc']:.4f}  ({result['trend']})")

    if all_results:
        print_summary(all_results)
    else:
        print("\nNo results to summarize.")


if __name__ == "__main__":
    main()
