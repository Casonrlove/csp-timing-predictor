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
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from data_collector import CSPDataCollector

warnings.filterwarnings("ignore", category=UserWarning)

# Ticker groups (must match train_timing_model_per_ticker.py)
TICKER_GROUPS = {
    "high_vol":    ["TSLA", "NVDA", "AMD"],
    "tech_growth": ["META", "GOOGL", "AMZN"],
    "tech_stable": ["AAPL", "MSFT"],
    "etf":         ["SPY", "QQQ"],
}

# XGBoost params (mirrors training script; no GPU needed for validation)
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
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


def calculate_group_threshold(df: pd.DataFrame) -> float:
    """60th-percentile drawdown threshold (matches training script logic)."""
    drawdowns = df["Max_Drawdown_35D"].dropna()
    return float(drawdowns.quantile(0.60)) if len(drawdowns) >= 10 else -5.0


def apply_rolling_threshold(df: pd.DataFrame, window: int = 90, quantile: float = 0.60) -> pd.DataFrame:
    """Assign target using a per-ticker rolling window threshold.

    For each row the threshold is the `quantile`-th percentile of the trailing
    `window` days of Max_Drawdown_35D for that ticker.  This keeps the positive
    rate near (1 - quantile) in every calendar year regardless of whether the
    market is in a bull or bear regime — the model always sees roughly the same
    class balance, rather than wildly swinging from 17 % in a bear year to 46 %
    in a bull year.

    Falls back to the full-history threshold for early rows that don't yet have
    `window // 2` observations.
    """
    df = df.copy()
    df["target"] = 0
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        drawdowns = df.loc[mask, "Max_Drawdown_35D"]
        rolling_thresh = drawdowns.rolling(window=window, min_periods=window // 2).quantile(quantile)
        # rows without enough history fall back to the full-ticker median
        fallback = float(drawdowns.quantile(quantile)) if len(drawdowns) >= 10 else -5.0
        rolling_thresh = rolling_thresh.fillna(fallback)
        df.loc[mask, "target"] = (drawdowns > rolling_thresh).astype(int)
    return df


def run_walk_forward(
    group_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_years: int = 4,
    test_years: int = 1,
    use_rolling_threshold: bool = True,
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
    # Rolling expanding window: train on years[0..i], test on years[i+1..i+test_years]
    for i in range(min_train_years - 1, len(years) - test_years):
        train_end_year = years[i]
        test_start_year = years[i + 1]
        test_end_year = years[min(i + test_years, len(years) - 1)]

        train_mask = df.index.year <= train_end_year
        test_mask = (df.index.year >= test_start_year) & (df.index.year <= test_end_year)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "target"].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "target"].values

        if len(X_train) < 200 or len(X_test) < 50:
            continue
        if len(np.unique(y_test)) < 2:
            continue  # Can't compute AUC if only one class in test fold

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Class weight
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        params = dict(XGB_PARAMS)
        params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0

        model = xgb.XGBClassifier(**params)
        model.fit(X_train_s, y_train, verbose=False)

        # Calibrate on test set (prefit — uses test as calibration data)
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        cal.fit(X_test_s, y_test)

        y_prob = cal.predict_proba(X_test_s)[:, 1]
        y_pred = cal.predict(X_test_s)

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)

        pos_rate = y_test.mean()

        fold_results.append({
            "fold": len(fold_results) + 1,
            "train_through": train_end_year,
            "test_year": f"{test_start_year}" if test_years == 1 else f"{test_start_year}-{test_end_year}",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_rate": pos_rate,
            "roc_auc": auc,
            "accuracy": acc,
        })

        print(
            f"  Fold {fold_results[-1]['fold']:>2}  "
            f"train≤{train_end_year}  test={fold_results[-1]['test_year']}  "
            f"n_train={len(X_train):>5}  n_test={len(X_test):>4}  "
            f"AUC={auc:.4f}  Acc={acc:.4f}  pos={pos_rate:.1%}"
        )

    return {
        "group": group_name,
        "threshold": threshold_desc,
        "folds": fold_results,
        "mean_auc": float(np.mean([f["roc_auc"] for f in fold_results])) if fold_results else 0.0,
        "std_auc": float(np.std([f["roc_auc"] for f in fold_results])) if fold_results else 0.0,
        "min_auc": float(np.min([f["roc_auc"] for f in fold_results])) if fold_results else 0.0,
        "max_auc": float(np.max([f["roc_auc"] for f in fold_results])) if fold_results else 0.0,
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
    print(f"\n{'Group':<15} {'Threshold':<18} {'Mean AUC':>10} {'Std AUC':>9} {'Min AUC':>9} {'Max AUC':>9}  {'Trend'}")
    print("-" * 90)
    for r in all_results:
        if not r.get("folds"):
            continue
        print(
            f"{r['group']:<15} {r['threshold']:<17}  "
            f"{r['mean_auc']:>9.4f}  {r['std_auc']:>8.4f}  "
            f"{r['min_auc']:>8.4f}  {r['max_auc']:>8.4f}   {r['trend']}"
        )

    print("\nInterpretation:")
    print("  AUC > 0.65 = good predictive power")
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
    print(f"Period:          {args.period}")
    print(f"Min train years: {args.min_train_years}")
    print(f"Groups:          {list(groups_to_run.keys())}")
    print(f"Threshold:       {'rolling 90D P60' if args.rolling_threshold else 'fixed full-history P60'}")
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
