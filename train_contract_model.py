"""
CSP Contract Model — Unified Strike-Breach Prediction
======================================================
Predicts P(strike_breach | ticker, date, market_features, strike_otm_pct)
for multiple delta buckets simultaneously.

Architecture mirrors train_ensemble_v3.py:
  - 4 ticker groups, VIX regime split at 18.0
  - XGB + LGBM DART base learners, 5-fold OOF stacking
  - LogisticRegression meta-learner on [xgb_prob, lgbm_prob, VIX, VIX_Rank, Regime_Trend]
  - StandardScaler for meta-features, isotonic calibration on tail 20% OOF
  - Recency weighting (2yr half-life), expanding window

Key differences from ensemble V3:
  1. Target: strike_breached (direct physical outcome)
  2. Features: 56 market + 7 contract = 63 total
  3. CV: contract_aware_time_split() (date-level splits, not row-level)
  4. Data: multi-strike expanded rows from create_contract_targets()

Saves to csp_contract_model.pkl (coexists with existing timing model).

Usage:
  python train_contract_model.py
  python train_contract_model.py --use-optuna-params --recency-halflife 2.0
  python train_contract_model.py --no-gpu --n-estimators 300
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    warnings.filterwarnings('ignore', message='.*feature names.*', category=UserWarning)
except ImportError:
    LGBM_AVAILABLE = False
    print("WARNING: lightgbm not installed — ensemble will use XGB only.")

from data_collector import CSPDataCollector
from feature_utils import (
    CONTRACT_FEATURE_COLS,
    contract_aware_time_split,
    create_breach_target,
)

MIN_REGIME_SAMPLES = 400
VIX_BOUNDARY = 18.0
N_META_FOLDS = 5
FORWARD_DAYS = 35
OPTUNA_PARAMS_FILE = 'optuna_params_v3.json'
OPTUNA_CONTRACT_PARAMS_FILE = 'optuna_contract_params.json'

TICKER_GROUPS = {
    'high_vol':     ['TSLA', 'NVDA', 'AMD', 'MSTR', 'COIN', 'PLTR'],
    'tech_growth':  ['META', 'GOOGL', 'AMZN', 'NFLX', 'CRM'],
    'tech_stable':  ['AAPL', 'MSFT', 'V', 'MA'],
    'etf':          ['SPY', 'QQQ', 'IWM'],
}

META_FEATURE_NAMES = ['xgb_prob', 'lgbm_prob', 'VIX', 'VIX_Rank', 'Regime_Trend']

# Default hyperparameters (same as ensemble V3)
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


class ContractModelTrainer:
    """Train XGB + LGBM stacking ensemble for contract-level breach prediction."""

    def __init__(self, use_gpu: bool = True, use_optuna_params: bool = False,
                 recency_halflife: float = 2.0, n_estimators: int | None = None):
        self.base_models: dict = {}
        self.meta_learners: dict = {}
        self.meta_scalers: dict = {}
        self.calibrators: dict = {}
        self.scalers: dict = {}
        self.thresholds: dict = {}
        self.group_mapping: dict = {}
        self.group_dfs: dict = {}
        self.feature_cols = None  # 56 market + 7 contract
        self.market_feature_cols = None  # 56 market only
        self.use_gpu = use_gpu
        self.optuna_params: dict = {}
        self.recency_halflife = recency_halflife
        self.n_estimators_override = n_estimators

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

        if use_optuna_params:
            # Load base params
            if os.path.exists(OPTUNA_PARAMS_FILE):
                with open(OPTUNA_PARAMS_FILE) as f:
                    self.optuna_params = json.load(f)
                print(f"Loaded base Optuna params from {OPTUNA_PARAMS_FILE}")
            # Contract-specific params override
            if os.path.exists(OPTUNA_CONTRACT_PARAMS_FILE):
                with open(OPTUNA_CONTRACT_PARAMS_FILE) as f:
                    contract_params = json.load(f)
                self.optuna_params.update(contract_params)
                print(f"Loaded contract Optuna params from {OPTUNA_CONTRACT_PARAMS_FILE}")
            if not self.optuna_params:
                print(f"WARNING: No Optuna params files found — using defaults")

    # ------------------------------------------------------------------
    # Data collection — multi-strike expansion
    # ------------------------------------------------------------------

    def collect_data(self, period: str = '10y') -> None:
        print(f"\n{'='*70}")
        print(f"COLLECTING CONTRACT DATA ({period})")
        print('='*70)

        for group, tickers in TICKER_GROUPS.items():
            print(f"\n[{group.upper()}]: {', '.join(tickers)}")
            frames = []
            for ticker in tickers:
                try:
                    print(f"  Fetching {ticker}...", end=' ', flush=True)
                    col = CSPDataCollector(ticker, period=period)
                    col.fetch_data()

                    if len(col.data) < 250:
                        print(f"SKIP (only {len(col.data)} days)")
                        continue

                    col.calculate_technical_indicators()
                    col.create_target_variable()  # for Max_Drawdown_35D and base features

                    # Get market feature columns from prepare_features
                    _, market_cols = col.prepare_features()
                    if self.market_feature_cols is None:
                        self.market_feature_cols = market_cols

                    # Multi-strike expansion
                    expanded = col.create_contract_targets()
                    expanded['ticker'] = ticker
                    expanded['group'] = group
                    frames.append(expanded)
                    print(f"ok ({len(expanded)} expanded rows)")
                except Exception as exc:
                    print(f"FAILED: {exc}")

            if frames:
                self.group_dfs[group] = pd.concat(frames, ignore_index=False).sort_index()

        if not self.group_dfs:
            raise RuntimeError("No data collected.")

        # Set feature columns: market features + contract interaction features
        self.feature_cols = list(self.market_feature_cols) + CONTRACT_FEATURE_COLS
        total_rows = sum(len(df) for df in self.group_dfs.values())
        print(f"\nTotal: {total_rows} expanded rows across {len(self.group_dfs)} groups")
        print(f"Features: {len(self.market_feature_cols)} market + {len(CONTRACT_FEATURE_COLS)} contract = {len(self.feature_cols)}")

    # ------------------------------------------------------------------
    # Hyperparameter helpers
    # ------------------------------------------------------------------

    def _xgb_params(self, model_key: str, scale_pos_weight: float) -> dict:
        p = dict(DEFAULT_XGB_PARAMS)
        optuna_key = f'{model_key}_xgb'
        if optuna_key in self.optuna_params:
            p.update(self.optuna_params[optuna_key])
        p['scale_pos_weight'] = scale_pos_weight
        if self.n_estimators_override:
            p['n_estimators'] = self.n_estimators_override
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
        p['scale_pos_weight'] = scale_pos_weight
        if self.n_estimators_override:
            p['n_estimators'] = self.n_estimators_override
        p['n_jobs'] = -1
        return p

    # ------------------------------------------------------------------
    # Core training: one regime slice
    # ------------------------------------------------------------------

    def _get_meta_feature_indices(self, feature_cols: list) -> list:
        meta_cols = ['VIX', 'VIX_Rank', 'Regime_Trend']
        return [feature_cols.index(c) if c in feature_cols else -1 for c in meta_cols]

    def _compute_sample_weights(self, df_slice: pd.DataFrame) -> np.ndarray:
        if self.recency_halflife <= 0:
            return np.ones(len(df_slice))
        dates = pd.to_datetime(df_slice.index)
        days_ago = (dates.max() - dates).days.values
        halflife_days = self.recency_halflife * 365.25
        return np.exp(-np.log(2) * days_ago / halflife_days)

    def _train_regime_slice(self, model_key: str, df_slice: pd.DataFrame) -> float:
        """Train XGB + LGBM via contract-aware OOF stacking.

        Returns mean OOF AUC across folds.
        """
        # Apply breach target (direct physical outcome)
        df_slice = create_breach_target(df_slice)

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df_slice.columns:
                df_slice[col] = 0.0

        X = df_slice[self.feature_cols].values
        y = df_slice['target'].values

        # Replace inf/nan in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        sample_weights = self._compute_sample_weights(df_slice)

        neg = (y == 0).sum()
        pos = (y == 1).sum()
        spw = neg / pos if pos > 0 else 1.0

        meta_idx = self._get_meta_feature_indices(self.feature_cols)

        # Contract-aware time split (splits on unique dates, not rows)
        splits = contract_aware_time_split(
            df_slice,
            n_splits=N_META_FOLDS,
            purge_days=FORWARD_DAYS,
        )

        oof_xgb = np.zeros(len(X))
        oof_lgbm = np.zeros(len(X))
        oof_meta = [np.zeros(len(X)) for _ in meta_idx]
        valid_oof_mask = np.zeros(len(X), dtype=bool)
        oof_aucs = []

        device = 'cuda' if self.use_gpu else 'cpu'
        print(f"  [{model_key}] Device={device}  n={len(X)}  breach_rate={y.mean():.1%}  spw={spw:.2f}")
        t_slice_start = time.time()
        print(f"  [{model_key}] OOF folds:", end=' ', flush=True)

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
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

            # LightGBM
            if LGBM_AVAILABLE:
                lgbm_model = lgb.LGBMClassifier(**self._lgbm_params(model_key, spw))
                lgbm_model.fit(X_tr, y_tr, sample_weight=sw_tr)
                oof_lgbm[val_idx] = lgbm_model.predict_proba(X_val)[:, 1]
            else:
                oof_lgbm[val_idx] = oof_xgb[val_idx]

            for j, idx in enumerate(meta_idx):
                if idx >= 0:
                    oof_meta[j][val_idx] = X[val_idx, idx]

            fold_prob = (oof_xgb[val_idx] + oof_lgbm[val_idx]) / 2.0
            fold_auc = roc_auc_score(y_val, fold_prob)
            oof_aucs.append(fold_auc)
            print(f"{fold_auc:.3f}", end=' ', flush=True)

        mean_oof_auc = float(np.mean(oof_aucs)) if oof_aucs else float('nan')
        print(f"  -> mean OOF AUC = {mean_oof_auc:.4f}")

        # Brier score on OOF
        if valid_oof_mask.sum() > 0:
            oof_avg = (oof_xgb[valid_oof_mask] + oof_lgbm[valid_oof_mask]) / 2.0
            brier = brier_score_loss(y[valid_oof_mask], oof_avg)
            print(f"  [{model_key}] OOF Brier score = {brier:.4f}")

        # Train meta-learner on OOF predictions
        meta_cols_data = [oof_xgb.reshape(-1, 1), oof_lgbm.reshape(-1, 1)]
        for col in oof_meta:
            meta_cols_data.append(col.reshape(-1, 1))
        meta_X = np.hstack(meta_cols_data)

        meta_fit_mask = valid_oof_mask
        if meta_fit_mask.sum() < 200:
            raise RuntimeError(f"[{model_key}] Not enough OOF rows: {meta_fit_mask.sum()}")

        meta_scaler = StandardScaler()
        meta_X_scaled = np.copy(meta_X)
        meta_X_scaled[meta_fit_mask] = meta_scaler.fit_transform(meta_X[meta_fit_mask])
        self.meta_scalers[model_key] = meta_scaler

        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_learner.fit(meta_X_scaled[meta_fit_mask], y[meta_fit_mask])
        self.meta_learners[model_key] = meta_learner

        # Isotonic calibration on held-out tail of OOF
        valid_indices = np.where(meta_fit_mask)[0]
        cal_start = int(len(valid_indices) * 0.8)
        cal_indices = valid_indices[cal_start:]
        if len(cal_indices) >= 50 and len(np.unique(y[cal_indices])) > 1:
            cal = CalibratedClassifierCV(FrozenEstimator(meta_learner), method='isotonic')
            cal.fit(meta_X_scaled[cal_indices], y[cal_indices])
            self.calibrators[model_key] = cal
            print(f"  [{model_key}] Isotonic calibrator fitted on {len(cal_indices)} OOF rows")

        # Train final base learners on full slice
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
        print(f"  [{model_key}] Slice total: {total_time:.1f}s  mean OOF AUC={mean_oof_auc:.4f}")
        return mean_oof_auc

    # ------------------------------------------------------------------
    # Train all groups
    # ------------------------------------------------------------------

    def train_all_groups(self) -> None:
        print(f"\n{'='*70}")
        print("TRAINING CONTRACT MODEL (REGIME-SPLIT + XGB + LGBM + STACKING)")
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
                    print(f"  Insufficient data, using combined fallback")
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

    def save(self, filename: str = 'csp_contract_model.pkl') -> None:
        data = {
            'contract_model': True,
            'ensemble': True,
            'regime_models': True,
            'base_models': self.base_models,
            'meta_learners': self.meta_learners,
            'meta_scalers': self.meta_scalers,
            'calibrators': self.calibrators,
            'scalers': self.scalers,
            'thresholds': self.thresholds,
            'group_mapping': self.group_mapping,
            'feature_cols': self.feature_cols,
            'market_feature_cols': self.market_feature_cols,
            'contract_feature_cols': CONTRACT_FEATURE_COLS,
            'ticker_groups': TICKER_GROUPS,
            'vix_boundary': VIX_BOUNDARY,
            'forward_days': FORWARD_DAYS,
            'meta_feature_names': META_FEATURE_NAMES,
            'version': 'contract_v1',
            'use_gpu': self.use_gpu,
            'lgbm_available': LGBM_AVAILABLE,
        }
        joblib.dump(data, filename)
        print(f"\nSaved contract model to {filename}")
        print(f"  Keys: {sorted(self.base_models.keys())}")
        print(f"  Features: {len(self.feature_cols)} ({len(self.market_feature_cols)} market + {len(CONTRACT_FEATURE_COLS)} contract)")


def main():
    parser = argparse.ArgumentParser(description='Train CSP Contract Model')
    parser.add_argument('--period', default='10y')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--output', default='csp_contract_model.pkl')
    parser.add_argument('--use-optuna-params', action='store_true',
                        help=f'Load hyperparameters from {OPTUNA_PARAMS_FILE}')
    parser.add_argument('--recency-halflife', type=float, default=2.0,
                        help='Half-life in years for exponential recency weighting (0 = uniform)')
    parser.add_argument('--n-estimators', type=int, default=None,
                        help='Override n_estimators for base learners (e.g. 300 for faster training)')
    args = parser.parse_args()

    print('='*70)
    print('CSP CONTRACT MODEL TRAINER')
    print('='*70)
    print(f"Optuna params: {'yes' if args.use_optuna_params else 'no'}")
    print(f"Recency halflife: {args.recency_halflife}yr")
    print(f"n_estimators: {args.n_estimators or 'default (500)'}")
    print(f"LightGBM: {'available' if LGBM_AVAILABLE else 'NOT AVAILABLE'}")

    trainer = ContractModelTrainer(
        use_gpu=not args.no_gpu,
        use_optuna_params=args.use_optuna_params,
        recency_halflife=args.recency_halflife,
        n_estimators=args.n_estimators,
    )
    trainer.collect_data(period=args.period)
    trainer.train_all_groups()
    trainer.save(args.output)

    print('\nDone.')


if __name__ == '__main__':
    main()
