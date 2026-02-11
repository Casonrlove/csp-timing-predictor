"""
CSP Timing Model V3 — Regime-Separated Training
================================================
Trains 2 XGBoost models per ticker group (8 models total):
  {group}_low_vix   — trained only on rows where VIX < 18
  {group}_high_vix  — trained only on rows where VIX >= 18

If a VIX-regime slice has < MIN_REGIME_SAMPLES training samples for a group,
the script falls back to a combined (non-regime-split) model for that group.

Model dict saved to csp_timing_model_v3.pkl includes:
  'regime_models': True
  'models':   {'{group}_low_vix': ..., '{group}_high_vix': ..., ...}
  'scalers':  same key structure
  'thresholds': same
  'group_mapping', 'feature_cols', 'ticker_groups'
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from data_collector import CSPDataCollector
from feature_utils import apply_rolling_threshold

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

MIN_REGIME_SAMPLES = 400   # Minimum training rows per regime slice before falling back

TICKER_GROUPS = {
    'high_vol':     ['TSLA', 'NVDA', 'AMD', 'MSTR', 'COIN', 'PLTR'],
    'tech_growth':  ['META', 'GOOGL', 'AMZN', 'NFLX', 'CRM'],
    'tech_stable':  ['AAPL', 'MSFT', 'V', 'MA'],
    'etf':          ['SPY', 'QQQ', 'IWM'],
}

VIX_BOUNDARY = 18.0


class RegimeTimingTrainer:
    """Train regime-split CSP timing models."""

    def __init__(self, use_gpu: bool = True):
        self.models: dict = {}
        self.scalers: dict = {}
        self.thresholds: dict = {}
        self.group_mapping: dict = {}
        self.group_dfs: dict = {}
        self.feature_cols = None
        self.use_gpu = use_gpu

        for group, tickers in TICKER_GROUPS.items():
            for ticker in tickers:
                self.group_mapping[ticker] = group

        if self.use_gpu:
            try:
                import subprocess
                res = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if res.returncode != 0:
                    self.use_gpu = False
            except Exception:
                self.use_gpu = False
        print(f"GPU: {'enabled' if self.use_gpu else 'disabled'}")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_data(self, period: str = '10y') -> None:
        print(f"\n{'='*70}")
        print(f"COLLECTING DATA ({period})")
        print('='*70)

        for group, tickers in TICKER_GROUPS.items():
            print(f"\n[{group.upper()}]: {', '.join(tickers)}")
            frames = []
            for ticker in tickers:
                try:
                    print(f"  Fetching {ticker}...", end=' ', flush=True)
                    col = CSPDataCollector(ticker, period=period)
                    df, cols = col.get_training_data()
                    if self.feature_cols is None:
                        self.feature_cols = cols
                    df['ticker'] = ticker
                    df['group'] = group
                    frames.append(df)
                    print(f"ok ({len(df)} rows, {df['Good_CSP_Time'].mean():.1%} pos)")
                except Exception as exc:
                    print(f"FAILED: {exc}")

            if frames:
                self.group_dfs[group] = pd.concat(frames, ignore_index=False).sort_index()
                print(f"  Group total: {len(self.group_dfs[group])} rows")

        if not self.group_dfs:
            raise RuntimeError("No data collected from any group.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_xgb_params(self, scale_pos_weight: float) -> dict:
        params = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'eval_metric': 'auc',
            'tree_method': 'hist',
        }
        if self.use_gpu:
            params['device'] = 'cuda'
        else:
            params['n_jobs'] = -1
        return params

    def _train_single(
        self,
        model_key: str,
        df_slice: pd.DataFrame,
    ) -> tuple:
        """Train + calibrate one XGBoost model on df_slice. Returns (model, scaler, threshold, auc)."""
        # Apply rolling threshold for label alignment
        df_slice = apply_rolling_threshold(df_slice, window=90, quantile=0.60)
        threshold_ref = float(df_slice['Max_Drawdown_35D'].quantile(0.60))
        self.thresholds[model_key] = threshold_ref

        X = df_slice[self.feature_cols].values
        y = df_slice['target'].values

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        self.scalers[model_key] = scaler

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos if pos > 0 else 1.0

        device = 'cuda' if self.use_gpu else 'cpu'
        print(f"    [{model_key}] Device={device}  n_train={len(X_train)}  n_test={len(X_test)}  "
              f"pos_rate={y_train.mean():.1%}  spw={spw:.2f}")

        t0 = time.time()
        model = xgb.XGBClassifier(**self._build_xgb_params(spw))
        model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
        xgb_time = time.time() - t0
        print(f"    [{model_key}] XGBoost training: {xgb_time:.1f}s")

        t1 = time.time()
        cal = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
        cal.fit(X_test_s, y_test)
        cal_time = time.time() - t1
        self.models[model_key] = cal

        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, cal.predict_proba(X_test_s)[:, 1])
        else:
            auc = float('nan')

        print(f"    [{model_key}] Calibration: {cal_time:.1f}s  AUC={auc:.4f}  total={xgb_time+cal_time:.1f}s")
        return cal, scaler, threshold_ref, auc

    def train_all_groups(self) -> None:
        print(f"\n{'='*70}")
        print("TRAINING REGIME-SPLIT MODELS (V3)")
        print('='*70)

        for group, df in self.group_dfs.items():
            print(f"\n--- {group.upper()} ---")

            low_mask = df['VIX'] < VIX_BOUNDARY
            high_mask = df['VIX'] >= VIX_BOUNDARY

            df_low = df[low_mask].copy()
            df_high = df[high_mask].copy()

            n_low_train = int(len(df_low) * 0.8)
            n_high_train = int(len(df_high) * 0.8)

            print(f"  low_vix  rows: {len(df_low):>5}  (train ~{n_low_train})")
            print(f"  high_vix rows: {len(df_high):>5}  (train ~{n_high_train})")

            # Low-VIX model
            if n_low_train >= MIN_REGIME_SAMPLES and len(np.unique(df_low['Good_CSP_Time'])) == 2:
                self._train_single(f'{group}_low_vix', df_low)
            else:
                print(f"  ⚠ Not enough low-VIX data ({n_low_train} < {MIN_REGIME_SAMPLES}), using combined fallback")
                self._train_single(f'{group}_low_vix', df)

            # High-VIX model
            if n_high_train >= MIN_REGIME_SAMPLES and len(np.unique(df_high['Good_CSP_Time'])) == 2:
                self._train_single(f'{group}_high_vix', df_high)
            else:
                print(f"  ⚠ Not enough high-VIX data ({n_high_train} < {MIN_REGIME_SAMPLES}), using combined fallback")
                self._train_single(f'{group}_high_vix', df)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print('='*70)
        for key in sorted(self.models.keys()):
            print(f"  {key}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str = 'csp_timing_model_v3.pkl') -> None:
        data = {
            'regime_models': True,
            'models': self.models,
            'scalers': self.scalers,
            'thresholds': self.thresholds,
            'group_mapping': self.group_mapping,
            'feature_cols': self.feature_cols,
            'ticker_groups': TICKER_GROUPS,
            'vix_boundary': VIX_BOUNDARY,
            'version': 'regime_v3_xgb',
            'use_gpu': self.use_gpu,
        }
        joblib.dump(data, filename)
        print(f"\n✓ Saved {len(self.models)} models to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Train Regime-Split CSP Timing Model V3')
    parser.add_argument('--period', default='10y')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--output', default='csp_timing_model_v3.pkl')
    args = parser.parse_args()

    print('='*70)
    print('REGIME-SPLIT CSP TIMING MODEL V3')
    print('='*70)

    trainer = RegimeTimingTrainer(use_gpu=not args.no_gpu)
    trainer.collect_data(period=args.period)
    trainer.train_all_groups()
    trainer.save(args.output)

    print('\n✅ Done. Reload API server to use the new model.')


if __name__ == '__main__':
    main()
