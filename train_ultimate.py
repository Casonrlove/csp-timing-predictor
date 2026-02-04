"""
ULTIMATE MODEL TRAINING - Maximum Accuracy
Combines: More data, ensemble models, extensive tuning, walk-forward validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

from data_collector import CSPDataCollector

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Install optuna: pip install optuna")

class UltimateTrainer:
    """Maximum accuracy model training"""

    # Extended ticker list for more training data
    EXTENDED_TICKERS = [
        # Tech giants
        'NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN',
        # More tech
        'INTC', 'QCOM', 'AVGO', 'CRM', 'ADBE', 'NFLX', 'PYPL',
        # Financials
        'JPM', 'BAC', 'GS', 'V', 'MA',
        # Other sectors
        'XOM', 'CVX', 'JNJ', 'PFE', 'UNH',
        # ETFs
        'SPY', 'QQQ', 'IWM'
    ]

    def __init__(self, tickers=None, use_gpu=True):
        if tickers is None:
            self.tickers = self.EXTENDED_TICKERS
        else:
            self.tickers = tickers

        self.use_gpu = use_gpu
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_cols = None
        self.best_params = {}

        # Check GPU
        if self.use_gpu:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode != 0:
                    self.use_gpu = False
            except:
                self.use_gpu = False

        print(f"GPU: {'Enabled' if self.use_gpu else 'Disabled'}")

    def collect_data(self, period='10y'):
        """Collect data from all tickers"""
        print(f"\n{'='*70}")
        print(f"COLLECTING DATA: {len(self.tickers)} tickers, {period} history")
        print("="*70)

        all_data = []
        successful = 0

        for i, ticker in enumerate(self.tickers):
            try:
                print(f"  [{i+1}/{len(self.tickers)}] {ticker}...", end=" ", flush=True)
                collector = CSPDataCollector(ticker, period=period)
                df, feature_cols = collector.get_training_data()

                if self.feature_cols is None:
                    self.feature_cols = feature_cols

                df['ticker'] = ticker
                all_data.append(df)
                successful += 1
                print(f"✓ {len(df)} samples")

            except Exception as e:
                print(f"✗ {str(e)[:40]}")
                continue

        if not all_data:
            raise ValueError("No data collected")

        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"\n{'='*70}")
        print(f"Total: {len(combined_df)} samples from {successful} tickers")
        print(f"Positive class: {combined_df['Good_CSP_Time'].mean():.1%}")
        print("="*70)

        return combined_df

    def prepare_data(self, df, test_size=0.2):
        """Prepare train/test split with proper time ordering"""
        # Sort by date if available
        if 'Date' in df.columns:
            df = df.sort_values('Date')

        X = df[self.feature_cols].values
        y = df['Good_CSP_Time'].values

        # Chronological split
        split_idx = int(len(X) * (1 - test_size))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples ({y_train.mean():.1%} positive)")
        print(f"  Test:  {len(X_test)} samples ({y_test.mean():.1%} positive)")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def tune_xgboost(self, X_train, y_train, n_trials=100):
        """Tune XGBoost with extensive search"""
        print(f"\n{'='*70}")
        print(f"TUNING XGBOOST ({n_trials} trials)")
        print("="*70)

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'max_depth': trial.suggest_int('max_depth', 4, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 0, 2),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'eval_metric': 'auc',
            }

            if self.use_gpu:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda'
            else:
                params['n_jobs'] = -1

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

                pred = model.predict_proba(X_v)[:, 1]
                scores.append(roc_auc_score(y_v, pred))

            return np.mean(scores)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params['xgboost'] = study.best_params
        print(f"\n✓ Best ROC-AUC: {study.best_value:.4f}")

        return study.best_params

    def tune_lightgbm(self, X_train, y_train, n_trials=100):
        """Tune LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM not available")
            return None

        print(f"\n{'='*70}")
        print(f"TUNING LIGHTGBM ({n_trials} trials)")
        print("="*70)

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'max_depth': trial.suggest_int('max_depth', 4, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
            }

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)])

                pred = model.predict_proba(X_v)[:, 1]
                scores.append(roc_auc_score(y_v, pred))

            return np.mean(scores)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params['lightgbm'] = study.best_params
        print(f"\n✓ Best ROC-AUC: {study.best_value:.4f}")

        return study.best_params

    def tune_random_forest(self, X_train, y_train, n_trials=50):
        """Tune Random Forest"""
        print(f"\n{'='*70}")
        print(f"TUNING RANDOM FOREST ({n_trials} trials)")
        print("="*70)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]

                model = RandomForestClassifier(**params)
                model.fit(X_t, y_t)

                pred = model.predict_proba(X_v)[:, 1]
                scores.append(roc_auc_score(y_v, pred))

            return np.mean(scores)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params['random_forest'] = study.best_params
        print(f"\n✓ Best ROC-AUC: {study.best_value:.4f}")

        return study.best_params

    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train ensemble of best models"""
        print(f"\n{'='*70}")
        print("TRAINING ENSEMBLE MODEL")
        print("="*70)

        estimators = []

        # XGBoost
        if 'xgboost' in self.best_params:
            xgb_params = self.best_params['xgboost'].copy()
            xgb_params['random_state'] = 42
            xgb_params['eval_metric'] = 'auc'
            if self.use_gpu:
                xgb_params['tree_method'] = 'hist'
                xgb_params['device'] = 'cuda'
            xgb_model = xgb.XGBClassifier(**xgb_params)
            estimators.append(('xgb', xgb_model))
            print("  + XGBoost")

        # LightGBM
        if LIGHTGBM_AVAILABLE and 'lightgbm' in self.best_params:
            lgb_params = self.best_params['lightgbm'].copy()
            lgb_params['random_state'] = 42
            lgb_params['verbose'] = -1
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            estimators.append(('lgb', lgb_model))
            print("  + LightGBM")

        # Random Forest
        if 'random_forest' in self.best_params:
            rf_params = self.best_params['random_forest'].copy()
            rf_params['random_state'] = 42
            rf_params['class_weight'] = 'balanced'
            rf_params['n_jobs'] = -1
            rf_model = RandomForestClassifier(**rf_params)
            estimators.append(('rf', rf_model))
            print("  + Random Forest")

        if len(estimators) < 2:
            print("Not enough models for ensemble, using best single model")
            if estimators:
                self.model = estimators[0][1]
                self.model.fit(X_train, y_train)
            return

        # Stacking ensemble with logistic regression meta-learner
        print("\n  Building stacking ensemble...")

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,
            n_jobs=-1,
            passthrough=False
        )

        start = time.time()
        self.model.fit(X_train, y_train)
        print(f"  ✓ Ensemble trained in {time.time()-start:.1f}s")

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print(f"\n{'='*70}")
        print("ENSEMBLE RESULTS")
        print("="*70)
        print(classification_report(y_test, y_pred))
        print(f"\n📊 ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    def train_single_best(self, X_train, y_train, X_test, y_test, model_type='xgboost'):
        """Train a single best model without ensemble"""
        print(f"\n{'='*70}")
        print(f"TRAINING BEST {model_type.upper()}")
        print("="*70)

        if model_type == 'xgboost' and 'xgboost' in self.best_params:
            params = self.best_params['xgboost'].copy()
            params['random_state'] = 42
            params['eval_metric'] = 'auc'
            if self.use_gpu:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda'
            self.model = xgb.XGBClassifier(**params)

        elif model_type == 'lightgbm' and 'lightgbm' in self.best_params:
            params = self.best_params['lightgbm'].copy()
            params['random_state'] = 42
            params['verbose'] = -1
            self.model = lgb.LGBMClassifier(**params)

        elif model_type == 'random_forest' and 'random_forest' in self.best_params:
            params = self.best_params['random_forest'].copy()
            params['random_state'] = 42
            params['class_weight'] = 'balanced'
            params['n_jobs'] = -1
            self.model = RandomForestClassifier(**params)

        else:
            raise ValueError(f"No tuned params for {model_type}")

        start = time.time()
        self.model.fit(X_train, y_train)
        print(f"✓ Trained in {time.time()-start:.1f}s")

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print(f"\n{'='*70}")
        print(f"{model_type.upper()} RESULTS")
        print("="*70)
        print(classification_report(y_test, y_pred))
        print(f"\n📊 ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    def save_model(self, filename='csp_model_multi.pkl'):
        """Save model"""
        package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': type(self.model).__name__,
            'best_params': self.best_params,
            'tickers': self.tickers
        }
        joblib.dump(package, filename)
        print(f"\n✓ Model saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Ultimate CSP Model Training')

    parser.add_argument('--tickers', type=str, nargs='+', default=None,
                        help='Tickers (default: 28 tickers)')
    parser.add_argument('--period', type=str, default='10y',
                        help='Data period')
    parser.add_argument('--trials', type=int, default=100,
                        help='Tuning trials per model')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of all models')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer tickers, fewer trials')
    parser.add_argument('--output', type=str, default='csp_model_multi.pkl',
                        help='Output filename')

    args = parser.parse_args()

    print("="*70)
    print("🚀 ULTIMATE MODEL TRAINING")
    print("="*70)

    # Quick mode settings
    if args.quick:
        args.trials = 30
        tickers = ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
    else:
        tickers = args.tickers

    print(f"\nSettings:")
    print(f"  Trials per model: {args.trials}")
    print(f"  Ensemble: {args.ensemble}")
    print(f"  Period: {args.period}")

    # Initialize
    trainer = UltimateTrainer(tickers=tickers, use_gpu=not args.no_gpu)

    # Collect data
    df = trainer.collect_data(period=args.period)

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)

    # Tune all models
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*70)

    trainer.tune_xgboost(X_train, y_train, n_trials=args.trials)

    if LIGHTGBM_AVAILABLE:
        trainer.tune_lightgbm(X_train, y_train, n_trials=args.trials)

    trainer.tune_random_forest(X_train, y_train, n_trials=max(30, args.trials // 2))

    # Train final model
    if args.ensemble:
        trainer.train_ensemble(X_train, y_train, X_test, y_test)
    else:
        # Use best single model (XGBoost usually)
        trainer.train_single_best(X_train, y_train, X_test, y_test, 'xgboost')

    # Save
    trainer.save_model(args.output)

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nRestart API server to use new model:")
    print("  pkill -f simple_api_server && python simple_api_server.py --port 8000 &")


if __name__ == "__main__":
    main()
