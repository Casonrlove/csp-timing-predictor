"""
CSP Timing Ensemble V3 — XGBoost + LightGBM + Stacking Meta-Learner
====================================================================
Combines:
  1. Regime-separated training (Phase 1) — low_vix / high_vix per group
  2. Two base learners: XGBoost + LightGBM (DART)
  3. Stacking meta-learner: LogisticRegression(C=1.0) trained on 5-fold
     time-series OOF predictions [xgb_prob, lgbm_prob, VIX, VIX_Rank, Regime_Trend]

Saved to csp_timing_ensemble.pkl:
  {
    'ensemble': True, 'regime_models': True,
    'base_models':   {'{group}_{regime}': {'xgb': ..., 'lgbm': ...}},
    'meta_learners': {'{group}_{regime}': LogisticRegression},
    'scalers':       {'{group}_{regime}': StandardScaler},
    'thresholds':    {'{group}_{regime}': float},
    'feature_cols':  [...],
    'group_mapping': {...},
    'ticker_groups': {...},
    'vix_boundary':  18.0,
    'version':       'ensemble_v3',
  }

If optuna_params_v3.json exists it is loaded and applied to the base learners.

Usage:
  python train_ensemble_v3.py
  python train_ensemble_v3.py --no-gpu
  python train_ensemble_v3.py --use-optuna-params   # load optuna_params_v3.json
"""

import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    # LightGBM v4 warns when predicting with numpy arrays if fitted with numpy (internal name tracking)
    warnings.filterwarnings('ignore', message='.*feature names.*', category=UserWarning)
except ImportError:
    LGBM_AVAILABLE = False
    print("WARNING: lightgbm not installed — run 'pip install lightgbm'. Ensemble will use XGB only.")

from data_collector import CSPDataCollector
from feature_utils import apply_rolling_threshold

MIN_REGIME_SAMPLES = 400
VIX_BOUNDARY = 18.0
N_META_FOLDS = 5
OPTUNA_PARAMS_FILE = 'optuna_params_v3.json'

TICKER_GROUPS = {
    'high_vol':     ['TSLA', 'NVDA', 'AMD', 'MSTR', 'COIN', 'PLTR'],
    'tech_growth':  ['META', 'GOOGL', 'AMZN', 'NFLX', 'CRM'],
    'tech_stable':  ['AAPL', 'MSFT', 'V', 'MA'],
    'etf':          ['SPY', 'QQQ', 'IWM'],
}

# Meta-learner feature indices in the OOF feature matrix
META_FEATURE_NAMES = ['xgb_prob', 'lgbm_prob', 'VIX', 'VIX_Rank', 'Regime_Trend']


# ---------------------------------------------------------------------------
# Default hyperparameters (overridden by Optuna params if present)
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'auc',
    'tree_method': 'hist',
}

DEFAULT_LGBM_PARAMS = {
    'boosting_type': 'dart',
    'num_leaves': 63,
    'min_child_samples': 20,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 500,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}


class EnsembleTimingTrainer:
    """Train XGB + LGBM stacking ensemble with regime separation."""

    def __init__(self, use_gpu: bool = True, use_optuna_params: bool = False,
                 recency_halflife: float = 2.0):
        self.base_models: dict = {}    # {key: {'xgb': ..., 'lgbm': ...}}
        self.meta_learners: dict = {}  # {key: LogisticRegression}
        self.scalers: dict = {}
        self.thresholds: dict = {}
        self.group_mapping: dict = {}
        self.group_dfs: dict = {}
        self.feature_cols = None
        self.use_gpu = use_gpu
        self.optuna_params: dict = {}
        self.recency_halflife = recency_halflife

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

        if use_optuna_params and os.path.exists(OPTUNA_PARAMS_FILE):
            with open(OPTUNA_PARAMS_FILE) as f:
                self.optuna_params = json.load(f)
            print(f"Loaded Optuna params from {OPTUNA_PARAMS_FILE} "
                  f"({len(self.optuna_params)} entries)")
        elif use_optuna_params:
            print(f"WARNING: {OPTUNA_PARAMS_FILE} not found — using defaults")

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
                    print(f"ok ({len(df)} rows)")
                except Exception as exc:
                    print(f"FAILED: {exc}")

            if frames:
                self.group_dfs[group] = pd.concat(frames, ignore_index=False).sort_index()

        if not self.group_dfs:
            raise RuntimeError("No data collected.")

    # ------------------------------------------------------------------
    # Hyperparameter helpers
    # ------------------------------------------------------------------

    def _xgb_params(self, model_key: str, scale_pos_weight: float) -> dict:
        p = dict(DEFAULT_XGB_PARAMS)
        optuna_key = f'{model_key}_xgb'
        if optuna_key in self.optuna_params:
            p.update(self.optuna_params[optuna_key])
        p['scale_pos_weight'] = scale_pos_weight
        if self.use_gpu:
            p['device'] = 'cuda'
        else:
            p['n_jobs'] = -1
        return p

    def _lgbm_params(self, model_key: str, scale_pos_weight: float) -> dict:
        p = dict(DEFAULT_LGBM_PARAMS)
        optuna_key = f'{model_key}_lgbm'
        if optuna_key in self.optuna_params:
            p.update(self.optuna_params[optuna_key])
        # LightGBM pip package uses OpenCL (not CUDA) — always run on CPU
        p['scale_pos_weight'] = scale_pos_weight
        p['n_jobs'] = -1
        return p

    # ------------------------------------------------------------------
    # Core training: one regime slice
    # ------------------------------------------------------------------

    def _get_meta_feature_indices(self, feature_cols: list) -> list:
        """Return column indices for VIX, VIX_Rank, Regime_Trend in feature_cols."""
        meta_cols = ['VIX', 'VIX_Rank', 'Regime_Trend']
        return [feature_cols.index(c) if c in feature_cols else -1 for c in meta_cols]

    def _compute_sample_weights(self, df_slice: pd.DataFrame) -> np.ndarray:
        """Exponential recency weighting: recent samples get more weight."""
        if self.recency_halflife <= 0:
            return np.ones(len(df_slice))
        dates = pd.to_datetime(df_slice.index)
        days_ago = (dates.max() - dates).days.values
        halflife_days = self.recency_halflife * 365.25
        return np.exp(-np.log(2) * days_ago / halflife_days)

    def _train_regime_slice(self, model_key: str, df_slice: pd.DataFrame) -> float:
        """
        Train XGB + LGBM base learners via 5-fold time-series OOF stacking,
        then train meta-learner on OOF predictions.

        Returns mean OOF AUC across folds.
        """
        # Label alignment: rolling 90D P60 threshold
        df_slice = apply_rolling_threshold(df_slice, window=90, quantile=0.60)
        self.thresholds[model_key] = float(df_slice['Max_Drawdown_35D'].quantile(0.60))

        X = df_slice[self.feature_cols].values
        y = df_slice['target'].values

        # Exponential recency sample weights
        sample_weights = self._compute_sample_weights(df_slice)

        neg = (y == 0).sum()
        pos = (y == 1).sum()
        spw = neg / pos if pos > 0 else 1.0

        # Meta-feature column indices in raw X
        meta_idx = self._get_meta_feature_indices(self.feature_cols)

        # 5-fold time-series OOF
        tscv = TimeSeriesSplit(n_splits=N_META_FOLDS)
        oof_xgb = np.zeros(len(X))
        oof_lgbm = np.zeros(len(X))
        oof_meta = [np.zeros(len(X)) for _ in meta_idx]
        valid_oof_mask = np.zeros(len(X), dtype=bool)
        oof_aucs = []

        device = 'cuda' if self.use_gpu else 'cpu'
        print(f"  [{model_key}] Device={device}  n={len(X)}  pos={y.mean():.1%}  spw={spw:.2f}")
        t_slice_start = time.time()
        print(f"  [{model_key}] OOF folds:", end=' ', flush=True)
        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            # Fold-local scaling prevents leakage from future validation rows.
            fold_scaler = StandardScaler()
            X_tr = fold_scaler.fit_transform(X[tr_idx])
            X_val = fold_scaler.transform(X[val_idx])
            y_tr, y_val = y[tr_idx], y[val_idx]
            sw_tr = sample_weights[tr_idx]

            if len(np.unique(y_val)) < 2:
                continue

            valid_oof_mask[val_idx] = True

            # XGBoost
            xgb_model = xgb.XGBClassifier(**self._xgb_params(model_key, spw))
            xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
            oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

            # LightGBM (if available)
            if LGBM_AVAILABLE:
                lgbm_model = lgb.LGBMClassifier(**self._lgbm_params(model_key, spw))
                lgbm_model.fit(X_tr, y_tr, sample_weight=sw_tr)
                oof_lgbm[val_idx] = lgbm_model.predict_proba(X_val)[:, 1]
            else:
                oof_lgbm[val_idx] = oof_xgb[val_idx]  # fallback: duplicate XGB

            # Meta side-features use raw values to match inference path.
            for j, idx in enumerate(meta_idx):
                if idx >= 0:
                    oof_meta[j][val_idx] = X[val_idx, idx]

            # Per-fold AUC (ensemble average)
            fold_prob = (oof_xgb[val_idx] + oof_lgbm[val_idx]) / 2.0
            fold_auc = roc_auc_score(y_val, fold_prob)
            oof_aucs.append(fold_auc)
            print(f"{fold_auc:.3f}", end=' ', flush=True)

        mean_oof_auc = float(np.mean(oof_aucs)) if oof_aucs else float('nan')
        print(f"  → mean OOF AUC = {mean_oof_auc:.4f}")

        # Train meta-learner on OOF predictions
        # Meta-features: [xgb_prob, lgbm_prob, VIX, VIX_Rank, Regime_Trend]
        meta_cols_data = []
        meta_cols_data.append(oof_xgb.reshape(-1, 1))
        meta_cols_data.append(oof_lgbm.reshape(-1, 1))
        for col in oof_meta:
            meta_cols_data.append(col.reshape(-1, 1))

        meta_X = np.hstack(meta_cols_data)
        meta_fit_mask = valid_oof_mask
        if meta_fit_mask.sum() < 200:
            raise RuntimeError(f"[{model_key}] Not enough OOF rows to fit meta-learner: {meta_fit_mask.sum()}")
        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_learner.fit(meta_X[meta_fit_mask], y[meta_fit_mask])
        self.meta_learners[model_key] = meta_learner

        # Train final base learners on full slice (with a full-data scaler for inference).
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[model_key] = scaler

        print(f"  [{model_key}] Training final base learners on full data...", flush=True)
        final_xgb = xgb.XGBClassifier(**self._xgb_params(model_key, spw))
        final_xgb.fit(X_scaled, y, sample_weight=sample_weights, verbose=False)

        if LGBM_AVAILABLE:
            final_lgbm = lgb.LGBMClassifier(**self._lgbm_params(model_key, spw))
            final_lgbm.fit(X_scaled, y, sample_weight=sample_weights)
        else:
            final_lgbm = None

        self.base_models[model_key] = {'xgb': final_xgb, 'lgbm': final_lgbm}
        total_time = time.time() - t_slice_start
        print(f"  [{model_key}] Slice total time: {total_time:.1f}s  mean OOF AUC={mean_oof_auc:.4f}")
        return mean_oof_auc

    # ------------------------------------------------------------------
    # Train all groups
    # ------------------------------------------------------------------

    def train_all_groups(self) -> None:
        print(f"\n{'='*70}")
        print("TRAINING ENSEMBLE V3 (REGIME-SPLIT + XGB + LGBM + STACKING)")
        print('='*70)

        for group, df in self.group_dfs.items():
            print(f"\n--- {group.upper()} ---")

            low_df = df[df['VIX'] < VIX_BOUNDARY].copy()
            high_df = df[df['VIX'] >= VIX_BOUNDARY].copy()

            for regime, regime_df in [('low_vix', low_df), ('high_vix', high_df)]:
                model_key = f'{group}_{regime}'
                n_train = int(len(regime_df) * 0.8)
                print(f"\n  Regime {regime}: {len(regime_df)} rows (train ~{n_train})")

                if n_train < MIN_REGIME_SAMPLES:
                    print(f"  ⚠ Insufficient data, using combined fallback")
                    auc = self._train_regime_slice(model_key, df.copy())
                else:
                    auc = self._train_regime_slice(model_key, regime_df)

                print(f"  {model_key}: mean OOF AUC = {auc:.4f}")

        print(f"\n{'='*70}")
        print(f"Trained {len(self.base_models)} regime-specific ensembles")
        print('='*70)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str = 'csp_timing_ensemble.pkl') -> None:
        data = {
            'ensemble': True,
            'regime_models': True,
            'base_models': self.base_models,
            'meta_learners': self.meta_learners,
            'scalers': self.scalers,
            'thresholds': self.thresholds,
            'group_mapping': self.group_mapping,
            'feature_cols': self.feature_cols,
            'ticker_groups': TICKER_GROUPS,
            'vix_boundary': VIX_BOUNDARY,
            'meta_feature_names': META_FEATURE_NAMES,
            'version': 'ensemble_v3',
            'use_gpu': self.use_gpu,
            'lgbm_available': LGBM_AVAILABLE,
        }
        joblib.dump(data, filename)
        print(f"\n✓ Saved ensemble to {filename}")
        print(f"  Keys: {sorted(self.base_models.keys())}")
        print(f"  Features: {len(self.feature_cols)}")


def main():
    parser = argparse.ArgumentParser(description='Train CSP Timing Ensemble V3')
    parser.add_argument('--period', default='10y')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--output', default='csp_timing_ensemble.pkl')
    parser.add_argument('--use-optuna-params', action='store_true',
                        help=f'Load hyperparameters from {OPTUNA_PARAMS_FILE}')
    parser.add_argument('--recency-halflife', type=float, default=2.0,
                        help='Half-life in years for exponential recency weighting (0 = uniform)')
    args = parser.parse_args()

    print('='*70)
    print('CSP TIMING ENSEMBLE V3')
    print('='*70)
    print(f"Optuna params: {'yes' if args.use_optuna_params else 'no'}")
    print(f"LightGBM:      {'available' if LGBM_AVAILABLE else 'NOT AVAILABLE — pip install lightgbm'}")

    trainer = EnsembleTimingTrainer(
        use_gpu=not args.no_gpu,
        use_optuna_params=args.use_optuna_params,
        recency_halflife=args.recency_halflife,
    )
    trainer.collect_data(period=args.period)
    trainer.train_all_groups()
    trainer.save(args.output)

    print('\n✅ Done.')


if __name__ == '__main__':
    main()
