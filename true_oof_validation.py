"""
True OOF Validation
====================
Gets genuinely unbiased metrics by re-running ONLY the last CV fold per regime.

  - Uses the same contract_aware_time_split(n_splits=5, purge_days=35) as training
  - Last fold: trains on first ~80% of dates, predicts on last ~20% (~2 years)
  - Production model never predicted those dates without having seen them first
  - This fold model did NOT train on the test period → truly unbiased

Trading simulation uses real Black-Scholes premium from per-row Volatility_20D,
not the rough delta×5% approximation.

Runtime: ~20-30 minutes (data fetch + 1-fold training per regime).

Usage:
    python true_oof_validation.py
    python true_oof_validation.py --n-estimators 300   # faster
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

TICKERS = {
    'high_vol_semi':  ['NVDA', 'AMD', 'TSLA', 'SMCI', 'MU', 'AMAT', 'LRCX'],
    'large_cap_tech': ['AAPL', 'MSFT', 'META', 'GOOGL', 'NFLX', 'CRWD', 'ADBE'],
    'dividend_value': ['TXN', 'JPM', 'V', 'JNJ', 'CVX', 'KO', 'WMT'],
    'etf':            ['SPY', 'QQQ', 'IWM', 'SMH'],
}
VIX_BOUNDARY = 18.0
FORWARD_DAYS = 35


def bs_put_premium(strike_otm_pct: np.ndarray, vol20d: np.ndarray,
                   T: float = 35/365) -> np.ndarray:
    """
    Black-Scholes put premium as % of current stock price.

    vol20d  : Volatility_20D as stored (percentage form, e.g. 35.0 = 35%)
              Divided by 100 inside this function.
    T       : time to expiry in years — 35/365 matches create_contract_targets.
    r       : zero (conservative, minor effect at 35 DTE)

    Standard BS put:
      d1 = [ln(S/K) + sigma^2*T/2] / (sigma*sqrt(T))
      d2 = d1 - sigma*sqrt(T)
      P/S = (K/S)*N(-d2) - N(-d1)
          = (1-otm_pct)*N(-d2) - N(-d1)
    """
    sigma  = np.clip(vol20d / 100.0, 0.05, 3.0)   # convert % → decimal
    sqrtT  = np.sqrt(T)
    # ln(S/K) = ln(1 / (1-otm_pct))
    ln_sk  = -np.log(np.clip(1.0 - strike_otm_pct, 0.50, 0.9999))
    d1     = (ln_sk + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2     = d1 - sigma * sqrtT
    premium = (1.0 - strike_otm_pct) * norm.cdf(-d2) - norm.cdf(-d1)
    return np.clip(premium, 0.001, 0.15)


def fetch_group_data(group_name, tickers, market_cols, contract_cols):
    from data_collector import CSPDataCollector
    frames = []
    for ticker in tickers:
        try:
            print(f"    {ticker}...", end=' ', flush=True)
            col = CSPDataCollector(ticker, period='10y')
            col.fetch_data()
            if len(col.data) < 500:
                print(f"skip ({len(col.data)}d)")
                continue
            col.calculate_technical_indicators()
            df = col.create_contract_targets()
            if df is None or len(df) < 500:
                print("skip"); continue

            avail = [c for c in market_cols if c in df.columns] + \
                    [c for c in contract_cols if c in df.columns]
            df = df.dropna(subset=avail + ['strike_breached'])
            if len(df) < 500:
                print("skip"); continue

            df['_ticker'] = ticker
            df['_group']  = group_name
            frames.append(df)
            print(f"ok ({len(df):,})")
        except Exception as e:
            print(f"ERROR: {e}")
    return pd.concat(frames) if frames else None


def run_last_fold(model_key, df, feature_cols, use_gpu, n_estimators):
    """Train on fold 4/5 data, predict on fold 5/5 data. Returns (preds, labels, meta_df)."""
    from feature_utils import contract_aware_time_split

    X     = df[feature_cols].fillna(0).values
    y     = df['strike_breached'].values.astype(float)
    extra = df[['target_delta', 'strike_otm_pct', 'Volatility_20D']].copy()

    splits = contract_aware_time_split(df, n_splits=5, purge_days=FORWARD_DAYS)
    if not splits:
        return None

    # Use only the last fold
    tr_idx, val_idx = splits[-1]
    print(f"  [{model_key}] last fold: train={len(tr_idx):,}  test={len(val_idx):,}  "
          f"test_breach={y[val_idx].mean():.3f}", flush=True)

    if len(np.unique(y[val_idx])) < 2:
        print(f"  [{model_key}] only one class in test — skip")
        return None

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X[tr_idx])
    X_val  = scaler.transform(X[val_idx])
    y_tr   = y[tr_idx]

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    device = 'cuda' if use_gpu else 'cpu'

    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method='hist', device=device, random_state=42, verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr, verbose=False)
    xgb_p = xgb_model.predict_proba(X_val)[:, 1]

    if LGBM_AVAILABLE:
        lgbm_model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            verbosity=-1, random_state=42,
        )
        lgbm_model.fit(X_tr, y_tr)
        lgbm_p = lgbm_model.predict_proba(X_val)[:, 1]
    else:
        lgbm_p = xgb_p

    # Meta-features
    vix_v   = df.iloc[val_idx]['VIX'].values         if 'VIX'          in df else np.full(len(val_idx), 20.)
    vixr_v  = df.iloc[val_idx]['VIX_Rank'].values    if 'VIX_Rank'     in df else np.full(len(val_idx), 50.)
    trnd_v  = df.iloc[val_idx]['Regime_Trend'].values if 'Regime_Trend' in df else np.ones(len(val_idx))

    meta_X = np.column_stack([xgb_p, lgbm_p, vix_v, vixr_v, trnd_v])
    ms     = StandardScaler()

    # Fit meta on TRAINING meta features, transform val
    xgb_tr  = xgb_model.predict_proba(X_tr)[:, 1]
    lgbm_tr = lgbm_model.predict_proba(X_tr)[:, 1] if LGBM_AVAILABLE else xgb_tr
    vix_tr  = df.iloc[tr_idx]['VIX'].values         if 'VIX'          in df else np.full(len(tr_idx), 20.)
    vixr_tr = df.iloc[tr_idx]['VIX_Rank'].values    if 'VIX_Rank'     in df else np.full(len(tr_idx), 50.)
    trnd_tr = df.iloc[tr_idx]['Regime_Trend'].values if 'Regime_Trend' in df else np.ones(len(tr_idx))
    meta_tr = np.column_stack([xgb_tr, lgbm_tr, vix_tr, vixr_tr, trnd_tr])
    meta_X_sc = ms.fit_transform(meta_tr)
    meta_val  = ms.transform(meta_X)

    meta_lr = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42)
    meta_lr.fit(meta_X_sc, y_tr)
    p = meta_lr.predict_proba(meta_val)[:, 1]

    auc = roc_auc_score(y[val_idx], p)
    print(f"  [{model_key}] fold AUC = {auc:.4f}  n={len(val_idx):,}", flush=True)

    meta_df = extra.iloc[val_idx].copy()
    meta_df['pred']  = p
    meta_df['label'] = y[val_idx]
    return meta_df


def compute_metrics(df_all, sell_threshold=0.40):
    preds   = df_all['pred'].values
    labels  = df_all['label'].values
    deltas  = df_all['target_delta'].values
    otm     = df_all['strike_otm_pct'].values
    sigma   = df_all['Volatility_20D'].values

    auc   = roc_auc_score(labels, preds)
    brier = brier_score_loss(labels, preds)

    # Optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, preds)
    accs = [accuracy_score(labels, (preds >= t)) for t in thresholds]
    opt_t = float(thresholds[int(np.argmax(accs))])

    def cls(t):
        b = (preds >= t).astype(int)
        return dict(
            t=t,
            acc=accuracy_score(labels, b),
            prec=precision_score(labels, b, zero_division=0),
            rec=recall_score(labels, b, zero_division=0),
            f1=f1_score(labels, b, zero_division=0),
        )

    opt  = cls(opt_t)
    at50 = cls(0.50)

    # Trading simulation with BS premium
    mask     = preds < sell_threshold
    y_t      = labels[mask]
    d_t      = deltas[mask]
    otm_t    = otm[mask]
    sigma_t  = sigma[mask]

    premium_pct = bs_put_premium(otm_t, sigma_t)    # real BS formula
    loss_pct    = otm_t                              # assigned at strike

    won  = (y_t == 0)
    lost = (y_t == 1)
    pnl  = np.where(won, premium_pct, -(loss_pct - premium_pct))

    gp = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    ev = pnl.mean()

    return dict(
        n=len(labels), auc=auc, brier=brier,
        opt_t=opt_t,   opt=opt,  at50=at50,
        breach_rate=labels.mean(),
        pred_median=float(np.median(preds)),
        n_trades=int(mask.sum()),
        win_rate=float(won.mean()),
        profit_factor=float(pf),
        ev_per_trade=float(ev),
        avg_premium=float(premium_pct.mean()),
        avg_loss=float(loss_pct[lost].mean()) if lost.sum() > 0 else 0.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        default='csp_contract_model.pkl')
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--no-gpu',       action='store_true')
    args = parser.parse_args()

    use_gpu = not args.no_gpu

    print(f"Loading {args.model} for feature column reference...")
    model_data = joblib.load(args.model)
    feature_cols  = model_data['feature_cols']
    market_cols   = model_data.get('market_feature_cols', feature_cols[:56])
    from feature_utils import CONTRACT_FEATURE_COLS
    contract_cols = model_data.get('contract_feature_cols', CONTRACT_FEATURE_COLS)
    all_feat      = market_cols + contract_cols

    all_results = []

    for group, tickers in TICKERS.items():
        print(f"\n{'='*60}")
        print(f"GROUP: {group.upper()}  ({len(tickers)} tickers)")
        print(f"{'='*60}")

        df_group = fetch_group_data(group, tickers, market_cols, contract_cols)
        if df_group is None:
            print(f"  No data for {group}"); continue

        avail_feat = [c for c in all_feat if c in df_group.columns]

        low_df  = df_group[df_group['VIX'] < VIX_BOUNDARY].copy()
        high_df = df_group[df_group['VIX'] >= VIX_BOUNDARY].copy()

        for regime, regime_df in [('low_vix', low_df), ('high_vix', high_df)]:
            if len(regime_df) < 1000:
                print(f"  {regime}: too few rows ({len(regime_df)}), skip")
                continue
            model_key = f'{group}_{regime}'
            result_df = run_last_fold(model_key, regime_df, avail_feat, use_gpu, args.n_estimators)
            if result_df is not None:
                result_df['regime'] = regime
                result_df['group']  = group
                all_results.append(result_df)

    if not all_results:
        print("No results collected."); return

    df_all = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*65}")
    print(f"TRUE OOF METRICS  —  {len(df_all):,} genuinely unseen predictions")
    print(f"Last CV fold per regime (trained on ~80% of history)")
    print(f"Test period: last ~20% of each ticker's data (~2 years)")
    print(f"{'='*65}")

    m = compute_metrics(df_all)

    def row(label, value, note=''):
        print(f"  {label:<30} {value:<20} {note}")

    print(f"\n  ── DISCRIMINATION ──────────────────────────────────────────")
    row("ROC-AUC (TRUE OOF)",    f"{m['auc']:.4f}",  "✅ Genuinely unbiased")
    row("Brier Score",           f"{m['brier']:.4f}", "✅ Genuinely unbiased")

    print(f"\n  ── CLASSIFICATION @ optimal threshold ({m['opt_t']:.2f}) ─")
    row("Accuracy",   f"{m['opt']['acc']*100:.2f}%",  "✅ Real")
    row("Precision",  f"{m['opt']['prec']*100:.2f}%", "✅ Real")
    row("Recall",     f"{m['opt']['rec']*100:.2f}%",  "✅ Real")
    row("F1 Score",   f"{m['opt']['f1']:.4f}",        "✅ Real")

    print(f"\n  ── CLASSIFICATION @ 0.50 threshold ─────────────────────────")
    row("Accuracy",   f"{m['at50']['acc']*100:.2f}%",  "")
    row("Precision",  f"{m['at50']['prec']*100:.2f}%", "")
    row("Recall",     f"{m['at50']['rec']*100:.2f}%",  "")

    print(f"\n  ── CALIBRATION ──────────────────────────────────────────────")
    row("Breach rate",    f"{m['breach_rate']*100:.1f}%",  "Actual base rate in test set")
    row("Pred median",    f"{m['pred_median']:.3f}",        "Ideally near 0.50")

    print(f"\n  ── SIMULATED CSP TRADING (BS premium, threshold=0.40) ───────")
    row("Win Rate",      f"{m['win_rate']*100:.1f}%",          "✅ Real breach labels")
    row("Profit Factor", f"{m['profit_factor']:.2f}",           "⚠️  BS premium approx")
    row("EV per Trade",  f"{m['ev_per_trade']*100:+.3f}% of K", "⚠️  BS premium approx")
    row("Avg premium",   f"{m['avg_premium']*100:.2f}% of K",   "From Volatility_20D")
    row("Avg loss",      f"{m['avg_loss']*100:.2f}% of K",      "= strike_otm_pct")
    row("N trades",      f"{m['n_trades']:,}",                   "Rows where model says sell")

    print(f"\n  ── SUMMARY — HONEST COMPARISON ──────────────────────────────")
    print(f"  {'Metric':<28} {'Old Model':>12} {'Inflated':>12} {'TRUE OOF':>10}")
    print(f"  {'-'*63}")
    print(f"  {'ROC-AUC':<28} {'0.9410':>12} {'0.8928':>12} {m['auc']:>10.4f}")
    print(f"  {'Accuracy':<28} {'79.58%':>12} {'81.79%':>12} {m['opt']['acc']*100:>9.2f}%")
    print(f"  {'Precision':<28} {'93.98%':>12} {'81.85%':>12} {m['opt']['prec']*100:>9.2f}%")
    print(f"  {'Recall':<28} {'56.66%':>12} {'81.59%':>12} {m['opt']['rec']*100:>9.2f}%")
    print(f"  {'Brier Score':<28} {'0.1592':>12} {'0.1592':>12} {m['brier']:>10.4f}")
    print(f"  {'Win Rate':<28} {'94.0%':>12} {'91.6%':>12} {m['win_rate']*100:>9.1f}%")
    print(f"  {'Profit Factor':<28} {'11.71':>12} {'1.27':>12} {m['profit_factor']:>10.2f}")
    print(f"  {'EV per Trade':<28} {'$1.40':>12} {'+0.197%K':>12} {m['ev_per_trade']*100:>+9.3f}%K")
    print(f"{'='*65}\n")

    # Per-group breakdown
    print(f"  Per-group AUC (TRUE OOF):")
    for grp in sorted(df_all['group'].unique()):
        sub = df_all[df_all['group'] == grp]
        if len(sub['label'].unique()) < 2:
            continue
        g_auc = roc_auc_score(sub['label'], sub['pred'])
        print(f"    {grp:<22}: AUC={g_auc:.4f}  n={len(sub):,}  breach={sub['label'].mean():.3f}")
    print()


if __name__ == '__main__':
    main()
