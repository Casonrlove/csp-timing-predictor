"""
GPU-Accelerated Model Training with Hyperparameter Tuning
Uses NVIDIA GPU for faster training and Optuna for optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb
import joblib
import argparse
import time
from data_collector import CSPDataCollector

# Check for Optuna (for hyperparameter tuning)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Note: Install optuna for hyperparameter tuning: pip install optuna")


class GPUTrainer:
    """GPU-accelerated model training with hyperparameter tuning"""

    def __init__(self, tickers=None, use_gpu=True):
        if tickers is None:
            self.tickers = ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
        else:
            self.tickers = tickers

        self.use_gpu = use_gpu
        self.scaler = StandardScaler()
        self.model = None
        self.feature_cols = None
        self.best_params = None

        # Check GPU availability
        if self.use_gpu:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ GPU detected - will use CUDA acceleration")
                else:
                    print("⚠ GPU not available - falling back to CPU")
                    self.use_gpu = False
            except:
                print("⚠ GPU check failed - falling back to CPU")
                self.use_gpu = False

    def collect_data(self, period='10y'):
        """Collect data from all tickers"""
        print(f"\n{'='*70}")
        print(f"Collecting data from {len(self.tickers)} tickers ({period})...")
        print("="*70)

        all_data = []

        for ticker in self.tickers:
            try:
                print(f"  Fetching {ticker}...", end=" ")
                collector = CSPDataCollector(ticker, period=period)
                df, feature_cols = collector.get_training_data()

                if self.feature_cols is None:
                    self.feature_cols = feature_cols

                df['ticker'] = ticker
                all_data.append(df)
                print(f"✓ {len(df)} samples ({df['Good_CSP_Time'].mean():.1%} positive)")

            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                continue

        if not all_data:
            raise ValueError("No data collected from any ticker")

        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"\n{'='*70}")
        print(f"Total: {len(combined_df)} samples | Positive: {combined_df['Good_CSP_Time'].mean():.1%}")
        print("="*70)

        return combined_df

    def prepare_data(self, df):
        """Prepare train/test split"""
        X = df[self.feature_cols]
        y = df['Good_CSP_Time']

        # Chronological split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

    def train_xgboost_gpu(self, X_train, y_train, X_test, y_test, n_estimators=500, tune=False, n_trials=50):
        """Train XGBoost with GPU acceleration"""
        print(f"\n{'='*70}")
        print("TRAINING XGBOOST WITH GPU ACCELERATION")
        print("="*70)

        # Calculate class weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        if tune and OPTUNA_AVAILABLE:
            print(f"\n🔍 Hyperparameter tuning with {n_trials} trials...")
            self.best_params = self._tune_xgboost(X_train, y_train, n_trials)
            params = self.best_params
        else:
            # Default optimized params
            params = {
                'n_estimators': n_estimators,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }

        # GPU settings
        if self.use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            print("🚀 Using GPU acceleration (CUDA)")
        else:
            params['tree_method'] = 'hist'
            params['n_jobs'] = -1
            print("💻 Using CPU (multi-threaded)")

        params['scale_pos_weight'] = scale_pos_weight
        params['random_state'] = 42
        params['eval_metric'] = 'auc'

        print(f"\nTraining with {params.get('n_estimators', 500)} trees, max_depth={params.get('max_depth', 8)}...")

        start_time = time.time()

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.1f} seconds")

        return self.model

    def _tune_xgboost(self, X_train, y_train, n_trials=50):
        """Tune XGBoost hyperparameters using Optuna"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
                'random_state': 42,
                'eval_metric': 'auc',
            }

            if self.use_gpu:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda'
            else:
                params['n_jobs'] = -1

            model = xgb.XGBClassifier(**params)

            # Cross-validation score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=1)
            return scores.mean()

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\n✓ Best ROC-AUC: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        return study.best_params

    def train_random_forest(self, X_train, y_train, n_estimators=500, tune=False, n_trials=50):
        """Train Random Forest (CPU-based but highly parallel)"""
        print(f"\n{'='*70}")
        print("TRAINING RANDOM FOREST")
        print("="*70)

        if tune and OPTUNA_AVAILABLE:
            print(f"\n🔍 Hyperparameter tuning with {n_trials} trials...")
            self.best_params = self._tune_random_forest(X_train, y_train, n_trials)
            params = self.best_params
        else:
            params = {
                'n_estimators': n_estimators,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
            }

        params['class_weight'] = 'balanced'
        params['random_state'] = 42
        params['n_jobs'] = -1

        print(f"\nTraining with {params['n_estimators']} trees, max_depth={params.get('max_depth', 15)}...")

        start_time = time.time()

        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.1f} seconds")

        return self.model

    def _tune_random_forest(self, X_train, y_train, n_trials=50):
        """Tune Random Forest hyperparameters"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }

            model = RandomForestClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=1)
            return scores.mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\n✓ Best ROC-AUC: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        return study.best_params

    def evaluate(self, X_test, y_test, df_test=None):
        """Evaluate model performance"""
        print(f"\n{'='*70}")
        print("MODEL EVALUATION")
        print("="*70)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")

        # Per-ticker evaluation if available
        if df_test is not None and 'ticker' in df_test.columns:
            print(f"\n{'='*70}")
            print("PER-TICKER PERFORMANCE")
            print("="*70)

            # Reset indices to align
            y_test_reset = y_test.reset_index(drop=True)

            for ticker in self.tickers:
                ticker_mask = df_test['ticker'] == ticker
                if ticker_mask.sum() < 10:
                    continue

                # Use .values to avoid index alignment issues
                ticker_indices = ticker_mask[ticker_mask].index.tolist()
                ticker_y = y_test_reset.iloc[ticker_indices]
                ticker_proba = y_pred_proba[ticker_indices]

                try:
                    ticker_roc = roc_auc_score(ticker_y, ticker_proba)
                    print(f"  {ticker:6s}: {ticker_mask.sum():4d} samples | ROC-AUC: {ticker_roc:.4f}")
                except:
                    pass

        return roc_auc

    def save_model(self, filename='csp_model_multi.pkl'):
        """Save the trained model"""
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': type(self.model).__name__,
            'best_params': self.best_params,
            'tickers': self.tickers
        }

        joblib.dump(model_package, filename)
        print(f"\n✓ Model saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated CSP Model Training')

    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'random_forest'],
                        help='Model type (default: xgboost for GPU)')

    parser.add_argument('--tickers', type=str, nargs='+',
                        default=['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
                        help='Tickers to train on')

    parser.add_argument('--period', type=str, default='10y',
                        help='Data period: 1y, 2y, 5y, 10y')

    parser.add_argument('--trees', type=int, default=500,
                        help='Number of trees/estimators (default: 500)')

    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning with Optuna')

    parser.add_argument('--trials', type=int, default=50,
                        help='Number of tuning trials (default: 50)')

    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU even if available')

    parser.add_argument('--output', type=str, default='csp_model_multi.pkl',
                        help='Output filename')

    args = parser.parse_args()

    print("="*70)
    print("🚀 GPU-ACCELERATED CSP MODEL TRAINING")
    print("="*70)
    print(f"\nModel:    {args.model.upper()}")
    print(f"Tickers:  {', '.join(args.tickers)}")
    print(f"Period:   {args.period}")
    print(f"Trees:    {args.trees}")
    print(f"Tuning:   {'Yes (' + str(args.trials) + ' trials)' if args.tune else 'No'}")
    print(f"GPU:      {'Disabled' if args.no_gpu else 'Auto-detect'}")

    # Initialize trainer
    trainer = GPUTrainer(tickers=args.tickers, use_gpu=not args.no_gpu)

    # Collect data
    df = trainer.collect_data(period=args.period)

    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = trainer.prepare_data(df)

    # Get test dataframe for per-ticker eval
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    # Train
    if args.model == 'xgboost':
        trainer.train_xgboost_gpu(X_train_scaled, y_train, X_test_scaled, y_test,
                                   n_estimators=args.trees, tune=args.tune, n_trials=args.trials)
        trainer.evaluate(X_test_scaled, y_test, df_test)
    else:
        trainer.train_random_forest(X_train, y_train,
                                     n_estimators=args.trees, tune=args.tune, n_trials=args.trials)
        trainer.evaluate(X_test, y_test, df_test)

    # Save
    trainer.save_model(args.output)

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)

    if args.tune:
        print("\nBest hyperparameters saved with model.")

    print(f"\nTo use: Restart the API server to load the new model")
    print(f"        pkill -f simple_api_server && python simple_api_server.py --port 8000 &")


if __name__ == "__main__":
    main()
