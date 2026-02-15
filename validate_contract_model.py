"""
Contract Model Validator
========================
Quick out-of-sample validation of csp_contract_model.pkl.

Approach:
  - Re-fetches recent data for a representative set of tickers
  - Holds out the LAST 252 trading days (≈1 year) as the test set
  - Trains on everything before that, evaluates on the holdout
  - Reports AUC, Brier score, calibration curve, and prediction distribution

Usage:
    python validate_contract_model.py
    python validate_contract_model.py --tickers NVDA AMD TSLA AAPL SPY
    python validate_contract_model.py --holdout-days 504   # 2-year holdout
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

# Representative sample — one from each group
DEFAULT_TICKERS = {
    'high_vol_semi': ['NVDA', 'AMD', 'TSLA', 'SMCI', 'MU'],
    'large_cap_tech': ['AAPL', 'MSFT', 'META', 'GOOGL', 'NFLX'],
    'dividend_value': ['TXN', 'JPM', 'V', 'JNJ', 'CVX'],
    'etf':            ['SPY', 'QQQ', 'IWM'],
}
HOLDOUT_DAYS = 252   # last ~1 year as test set


def run_validation(tickers_by_group: dict, holdout_days: int, model_path: str):
    from data_collector import CSPDataCollector
    from feature_utils import CONTRACT_FEATURE_COLS

    print(f"Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    base_models   = model_data['base_models']
    meta_learners  = model_data['meta_learners']
    meta_scalers   = model_data.get('meta_scalers', {})
    calibrators    = model_data.get('calibrators', {})
    scalers        = model_data['scalers']
    group_mapping  = model_data['group_mapping']
    feature_cols   = model_data['feature_cols']
    market_cols    = model_data.get('market_feature_cols', feature_cols[:56])
    contract_cols  = model_data.get('contract_feature_cols', CONTRACT_FEATURE_COLS)
    vix_boundary   = model_data.get('vix_boundary', 18.0)

    # Print any stored OOF metrics from training
    oof = model_data.get('oof_metrics', {})
    if oof:
        print("\n--- Stored OOF AUC (from training) ---")
        for k, v in sorted(oof.items()):
            print(f"  {k:35s}: AUC={v['oof_auc']:.4f}  n={v['n_samples']:,}  breach_rate={v['breach_rate']:.3f}")
        avg = np.mean([v['oof_auc'] for v in oof.values()])
        print(f"  {'AVERAGE':35s}: AUC={avg:.4f}")

    print(f"\n--- Holdout Validation (last {holdout_days} trading days) ---")

    all_preds, all_labels, all_groups = [], [], []

    for group, tickers in tickers_by_group.items():
        for ticker in tickers:
            try:
                print(f"  [{group}] {ticker}...", end=' ', flush=True)
                col = CSPDataCollector(ticker, period='10y')
                col.fetch_data()
                if len(col.data) < holdout_days + 100:
                    print(f"skip (only {len(col.data)} days)")
                    continue
                col.calculate_technical_indicators()
                df = col.create_contract_targets()
                if df is None or len(df) < holdout_days + 50:
                    print("skip (insufficient targets)")
                    continue

                # Identify market feature cols present in this df
                avail_market = [c for c in market_cols if c in df.columns]
                avail_contract = [c for c in contract_cols if c in df.columns]
                all_feat = avail_market + avail_contract

                df = df.dropna(subset=all_feat + ['strike_breached'])

                # Time split — train on everything before holdout window
                # Use index position since df is sorted by date
                unique_dates = df.index.unique() if df.index.name else pd.to_datetime(df.index).unique()
                n_dates = len(unique_dates)
                if n_dates < holdout_days + 50:
                    print(f"skip (only {n_dates} dates)")
                    continue

                holdout_date = sorted(unique_dates)[-holdout_days]
                test_df = df[df.index >= holdout_date] if df.index.name else \
                          df[pd.to_datetime(df.index) >= holdout_date]

                if len(test_df) < 10:
                    print(f"skip (test set too small: {len(test_df)})")
                    continue

                # Predict using the loaded model
                regime = 'high_vix' if test_df['VIX'].mean() >= vix_boundary else 'low_vix'
                grp = group_mapping.get(ticker, 'large_cap_tech')
                model_key = f'{grp}_{regime}'

                if model_key not in base_models:
                    # Try alternate regime
                    alt = f'{grp}_low_vix' if regime == 'high_vix' else f'{grp}_high_vix'
                    model_key = alt if alt in base_models else grp
                if model_key not in base_models:
                    print(f"no model for key {model_key}")
                    continue

                scaler   = scalers[model_key]
                bm       = base_models[model_key]
                meta     = meta_learners[model_key]
                ms       = meta_scalers.get(model_key)
                cal      = calibrators.get(model_key)

                X = test_df[all_feat].fillna(0).values
                y = test_df['strike_breached'].values.astype(float)

                X_sc = scaler.transform(X)
                xgb_p = bm['xgb'].predict_proba(X_sc)[:, 1]
                lgbm_p = bm['lgbm'].predict_proba(X_sc)[:, 1] if bm.get('lgbm') else xgb_p

                vix_vals   = test_df['VIX'].values if 'VIX' in test_df else np.full(len(X), 20.0)
                vixr_vals  = test_df['VIX_Rank'].values if 'VIX_Rank' in test_df else np.full(len(X), 50.0)
                trend_vals = test_df['Regime_Trend'].values if 'Regime_Trend' in test_df else np.ones(len(X))

                meta_X = np.column_stack([xgb_p, lgbm_p, vix_vals, vixr_vals, trend_vals])
                if ms:
                    meta_X = ms.transform(meta_X)
                if cal:
                    p = cal.predict_proba(meta_X)[:, 1]
                else:
                    p = meta.predict_proba(meta_X)[:, 1]

                all_preds.extend(p)
                all_labels.extend(y)
                all_groups.extend([group] * len(y))

                auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
                print(f"n={len(y):,} breach={y.mean():.2f} AUC={auc:.3f} "
                      f"pred_range=[{p.min():.3f},{p.max():.3f}]")

            except Exception as e:
                print(f"ERROR: {e}")

    if not all_preds:
        print("\nNo validation data collected.")
        return

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    overall_auc   = roc_auc_score(labels, preds)
    overall_brier = brier_score_loss(labels, preds)

    print(f"\n{'='*60}")
    print(f"OVERALL HOLDOUT RESULTS ({len(labels):,} predictions)")
    print(f"{'='*60}")
    print(f"  AUC:         {overall_auc:.4f}  (0.5=no skill, 0.7=good, 0.8=great)")
    print(f"  Brier score: {overall_brier:.4f}  (lower=better, 0.25=no skill)")
    print(f"  Breach rate: {labels.mean():.3f}")
    print(f"  Pred range:  [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"  Pred median: {np.median(preds):.3f}")
    print(f"  Pred >0.50:  {(preds > 0.50).mean()*100:.1f}%  ← should be ~50% if calibrated")
    print(f"  Pred >0.60:  {(preds > 0.60).mean()*100:.1f}%")
    print(f"  Pred <0.40:  {(preds < 0.40).mean()*100:.1f}%")

    print(f"\n  Per-group AUC:")
    for grp in sorted(set(all_groups)):
        idx = [i for i, g in enumerate(all_groups) if g == grp]
        g_preds  = preds[idx]
        g_labels = labels[idx]
        if len(np.unique(g_labels)) > 1:
            g_auc = roc_auc_score(g_labels, g_preds)
            print(f"    {grp:20s}: AUC={g_auc:.4f}  n={len(idx):,}  breach={g_labels.mean():.2f}  "
                  f"pred=[{g_preds.min():.3f},{g_preds.max():.3f}]")

    # Calibration check
    print(f"\n  Calibration (ideal: predicted prob ≈ actual breach rate):")
    try:
        frac_pos, mean_pred = calibration_curve(labels, preds, n_bins=5, strategy='quantile')
        for mp, fp in zip(mean_pred, frac_pos):
            bar = '█' * int(fp * 20)
            print(f"    pred={mp:.2f}  actual={fp:.2f}  {'✓' if abs(mp-fp)<0.10 else '✗'} {bar}")
    except Exception as e:
        print(f"    (calibration plot failed: {e})")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', help='Override ticker list (flat, no groups)')
    parser.add_argument('--holdout-days', type=int, default=HOLDOUT_DAYS)
    parser.add_argument('--model', default='csp_contract_model.pkl')
    args = parser.parse_args()

    if args.tickers:
        tickers_by_group = {'custom': args.tickers}
    else:
        tickers_by_group = DEFAULT_TICKERS

    run_validation(tickers_by_group, args.holdout_days, args.model)


if __name__ == '__main__':
    main()
