"""
STRIKE PROBABILITY MODEL V3 - Two-Stage Residual Architecture
- Uses delta as baseline, model predicts adjustment (residual)
- No StandardScaler (tree-based models don't need it)
- Delta interaction features (Delta × IV, Delta × VIX, Delta²)
- XGBRegressor for continuous residual prediction
- Removes need for hybrid blend hack
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_collector import CSPDataCollector
from scipy.stats import norm

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")


class ResidualStrikeModel:
    """
    Two-stage residual model:
    Stage 1: Delta provides baseline probability
    Stage 2: Model predicts adjustment based on market conditions
    
    Final prediction = delta + model_adjustment
    """

    # Ticker groups by volatility profile (same as V2)
    TICKER_GROUPS = {
        'high_vol': ['TSLA', 'NVDA', 'AMD', 'COIN', 'MSTR', 'PLTR', 'MRVL', 'MU'],
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX', 'CRM', 'ADBE'],
        'semis': ['INTC', 'QCOM', 'AVGO', 'TXN'],
        'finance': ['JPM', 'BAC', 'GS', 'V', 'MA', 'C', 'WFC'],
        'healthcare': ['UNH', 'JNJ', 'PFE', 'LLY', 'MRK', 'ABBV'],
        'consumer': ['COST', 'WMT', 'HD', 'LOW', 'SBUX', 'MCD', 'NKE'],
        'energy': ['XOM', 'CVX', 'COP'],
        'industrial': ['CAT', 'DE', 'HON', 'UPS', 'RTX'],
        'etf': ['SPY', 'QQQ', 'IWM', 'TQQQ', 'SCHD'],
    }

    # Delta buckets for training
    DELTA_BUCKETS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    def __init__(self, forward_days=35, use_gpu=True):
        self.forward_days = forward_days
        self.models = {}  # {group_name: model}
        self.feature_cols = None
        self.group_mapping = {}

        # Build reverse mapping
        for group, tickers in self.TICKER_GROUPS.items():
            for ticker in tickers:
                self.group_mapping[ticker] = group

        # Check GPU
        self.use_gpu = use_gpu and self._check_gpu()
        if self.use_gpu:
            print("GPU acceleration enabled")
        else:
            print("Using CPU")

    def _check_gpu(self):
        try:
            test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
            test_model.fit([[1, 2], [3, 4]], [1, 2])
            return True
        except:
            return False

    def get_ticker_group(self, ticker):
        """Get the volatility group for a ticker"""
        return self.group_mapping.get(ticker, 'tech')

    def collect_data(self, period='3y'):
        """Collect data and create residual training set"""
        all_tickers = []
        for tickers in self.TICKER_GROUPS.values():
            all_tickers.extend(tickers)
        all_tickers = list(set(all_tickers))

        group_data = {group: [] for group in self.TICKER_GROUPS.keys()}

        for i, ticker in enumerate(all_tickers):
            print(f"[{i+1}/{len(all_tickers)}] Collecting {ticker}...")
            try:
                collector = CSPDataCollector(ticker, period=period)
                collector.fetch_data()
                collector.calculate_technical_indicators()

                df = collector.data.copy()
                group = self.get_ticker_group(ticker)

                # Calculate forward breach/drop outcomes for label construction.
                # We align with timing target semantics: strike breach anytime in
                # the forward window (not only terminal-day outcome).
                forward_breach_drops = []
                for j in range(len(df)):
                    if j + self.forward_days < len(df):
                        current = df['Close'].iloc[j]
                        future_window = df['Close'].iloc[j:j + self.forward_days + 1]
                        min_future = future_window.min()
                        breach_drop = ((current - min_future) / current) * 100
                        forward_breach_drops.append(breach_drop)
                    else:
                        forward_breach_drops.append(np.nan)

                df['Forward_Drop_Pct'] = forward_breach_drops
                df['Ticker'] = ticker
                df['Group'] = group

                # Create training rows with residual target
                expanded_rows = []
                for idx in range(len(df) - self.forward_days):
                    row = df.iloc[idx]
                    forward_breach_drop = row['Forward_Drop_Pct']

                    if pd.isna(forward_breach_drop):
                        continue

                    for target_delta in self.DELTA_BUCKETS:
                        # Use IV estimate to calculate strike (FIXED from V2)
                        rv = row.get('Volatility_20D', 20) / 100
                        if rv <= 0:
                            rv = 0.20

                        # Use IV_RV_Ratio to scale to IV
                        iv_rv_ratio = row.get('IV_RV_Ratio', 1.2)
                        iv_rv_ratio = max(1.0, min(iv_rv_ratio, 2.5))
                        vol = rv * iv_rv_ratio

                        T = self.forward_days / 365

                        # Calculate strike OTM percentage
                        try:
                            strike_pct = vol * np.sqrt(T) * norm.ppf(1 - target_delta) * 100
                            strike_pct = max(1, min(30, strike_pct))
                        except:
                            strike_pct = target_delta * 30

                        # Did the strike get breached anytime in the holding window?
                        breached_strike = 1 if forward_breach_drop >= strike_pct else 0

                        # **KEY CHANGE**: Target is RESIDUAL (actual - delta)
                        residual = breached_strike - target_delta

                        new_row = row.copy()
                        new_row['Target_Delta'] = target_delta
                        new_row['Strike_OTM_Pct'] = strike_pct
                        new_row['Expired_ITM'] = breached_strike
                        new_row['Residual'] = residual  # This is what we train on

                        # Add delta interaction features
                        new_row['Delta_x_Vol'] = target_delta * vol
                        new_row['Delta_x_VIX'] = target_delta * row.get('VIX', 15) / 100
                        new_row['Delta_x_IV_Rank'] = target_delta * row.get('IV_Rank', 50) / 100
                        new_row['Delta_Squared'] = target_delta ** 2
                        new_row['Delta_x_RSI'] = target_delta * row.get('RSI', 50) / 100

                        expanded_rows.append(new_row)

                if expanded_rows:
                    expanded_df = pd.DataFrame(expanded_rows)
                    group_data[group].append(expanded_df)
                    print(f"    {len(expanded_df)} option outcome samples")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        # Combine data by group
        self.group_dfs = {}
        for group, dfs in group_data.items():
            if dfs:
                self.group_dfs[group] = pd.concat(dfs, ignore_index=True)
                print(f"\n{group}: {len(self.group_dfs[group])} total samples")

        return self.group_dfs

    def prepare_features(self, df):
        """Prepare features - NO SCALER needed for tree models"""
        exclude_cols = [
            'Forward_Drop_Pct', 'Expired_ITM', 'Strike_OTM_Pct', 'Residual',
            'Ticker', 'Group', 'Date', 'Good_CSP_Time',
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
            'Forward_Return', 'Max_Drawdown_35D', 'Max_Drawdown_Pct',
            'Adjusted_Threshold', 'Expected_Premium_Pct',
            'Target_Delta'  # Exclude from features, added manually
        ]

        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns
                               if c not in exclude_cols
                               and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
            print(f"  Feature columns: {len(self.feature_cols)}")

        return self.feature_cols

    def purged_time_series_split(self, n_samples, n_splits=5, gap_days=35):
        """
        Time-series cross-validation with purge gap to prevent data leakage.

        The gap ensures validation data is at least gap_days after training data,
        matching our forward_days prediction horizon.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for train_idx, val_idx in tscv.split(range(n_samples)):
            # Apply purge gap: remove validation samples too close to training
            train_end = train_idx[-1] if len(train_idx) > 0 else 0
            purged_val_idx = [i for i in val_idx if i >= train_end + gap_days]

            if len(purged_val_idx) > 100:  # Require minimum validation size
                yield train_idx, np.array(purged_val_idx)

    def optimize_hyperparameters(self, group, df, n_trials=50):
        """
        Use Optuna to find optimal XGBoost hyperparameters via time-series CV.

        Returns: best hyperparameters dict
        """
        if not OPTUNA_AVAILABLE:
            print("  Optuna not available, using default params")
            return None

        print(f"  Running Optuna hyperparameter optimization ({n_trials} trials)...")

        feature_cols = self.prepare_features(df)
        df_clean = df[feature_cols + ['Residual', 'Target_Delta', 'Expired_ITM']].dropna()

        X = df_clean[feature_cols].values
        y = df_clean['Residual'].values
        deltas = df_clean['Target_Delta'].values
        if len(deltas.shape) > 1:
            deltas = deltas[:, 0]
        actuals = df_clean['Expired_ITM'].values
        if len(actuals.shape) > 1:
            actuals = actuals[:, 0]

        # Recency weights
        n = len(y)
        decay_rate = 2.0
        sample_weights = np.exp(decay_rate * np.linspace(-1, 0, n))
        sample_weights = sample_weights / sample_weights.mean()

        device_params = {
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
        }

        def objective(trial):
            # Suggest hyperparameters
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                **device_params
            }

            # Time-series cross-validation
            cv_scores = []
            for train_idx, val_idx in self.purged_time_series_split(len(X), n_splits=3, gap_days=self.forward_days):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                w_train = sample_weights[train_idx]

                deltas_val = deltas[val_idx]
                actuals_val = actuals[val_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

                # Predict on validation
                y_pred_val = model.predict(X_val)
                predicted_probs = deltas_val + y_pred_val
                predicted_probs = np.clip(predicted_probs, 0, 1)

                # Calculate Brier score (lower is better)
                from sklearn.metrics import brier_score_loss
                brier = brier_score_loss(actuals_val, predicted_probs)
                cv_scores.append(brier)

            return np.mean(cv_scores) if cv_scores else 1.0

        # Run optimization
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

        print(f"  Best Brier score: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

        return study.best_params

    def train_group_model(self, group, df, tune_hyperparams=False, n_trials=50):
        """Train residual regression model for a group"""
        print(f"\n{'='*60}")
        print(f"Training V3 residual model for: {group}")
        print(f"{'='*60}")

        feature_cols = self.prepare_features(df)

        # Clean data - keep Target_Delta and Expired_ITM for metrics
        df_clean = df[feature_cols + ['Residual', 'Target_Delta', 'Expired_ITM']].dropna()

        X = df_clean[feature_cols].values
        y = df_clean['Residual'].values  # Train on residual, not binary

        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Mean residual: {y.mean():.4f} (negative = delta overestimates)")
        print(f"Residual std: {y.std():.4f}")

        # Recency weighting
        n = len(y)
        decay_rate = 2.0
        sample_weights = np.exp(decay_rate * np.linspace(-1, 0, n))
        sample_weights = sample_weights / sample_weights.mean()

        # GPU/CPU params
        device_params = {
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
        }

        # Hyperparameter optimization if requested
        if tune_hyperparams:
            best_params = self.optimize_hyperparameters(group, df, n_trials=n_trials)
            if best_params:
                # Use optimized params
                print(f"  Training with optimized hyperparameters...")
                model_params = {
                    'objective': 'reg:squarederror',
                    'random_state': 42,
                    **best_params,
                    **device_params
                }
            else:
                # Fallback to defaults
                model_params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 300,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    **device_params
                }
        else:
            # Use default params (fast training)
            print(f"  Training with default hyperparameters...")
            model_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                **device_params
            }

        model = xgb.XGBRegressor(**model_params)
        model.fit(X, y, sample_weight=sample_weights, verbose=False)
        self.models[group] = model

        # Evaluate on training set
        y_pred = model.predict(X)

        # Convert residuals back to probabilities for metrics
        # Get corresponding deltas for each sample
        deltas = df_clean['Target_Delta'].values
        if len(deltas.shape) > 1:
            deltas = deltas[:, 0]  # Flatten if multiple columns

        actuals = df_clean['Expired_ITM'].values
        if len(actuals.shape) > 1:
            actuals = actuals[:, 0]  # Flatten if multiple columns

        predicted_probs = deltas + y_pred
        predicted_probs = np.clip(predicted_probs, 0, 1)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Calculate Brier score on final probabilities
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(actuals, predicted_probs)
        
        print(f"  Residual MSE: {mse:.6f}")
        print(f"  Residual MAE: {mae:.6f}")
        print(f"  Brier Score (final probs): {brier:.4f}")

        return self.models[group]

    def train_all(self, period='3y', tune_hyperparams=False, n_trials=50):
        """Train models for all groups"""
        print("="*70)
        print("STRIKE MODEL V3 - TWO-STAGE RESIDUAL TRAINING")
        print("="*70)
        print("\nDelta provides baseline, model predicts market-condition adjustment")
        print("Target = Expired_ITM - Target_Delta (residual)")
        if tune_hyperparams:
            print(f"Hyperparameter tuning: ENABLED ({n_trials} trials per group)")
        else:
            print("Hyperparameter tuning: DISABLED (use --tune to enable)")
        print()

        # Collect data
        self.collect_data(period=period)

        # Train each group
        for group, df in self.group_dfs.items():
            if len(df) > 1000:
                self.train_group_model(group, df, tune_hyperparams=tune_hyperparams, n_trials=n_trials)
            else:
                print(f"\nSkipping {group} - insufficient data ({len(df)} samples)")

        return self.models

    def predict_adjustment(self, features_df, ticker):
        """
        Predict residual adjustment for given market conditions.
        
        Returns: adjustment value (typically -0.10 to +0.10)
        """
        group = self.get_ticker_group(ticker)

        if group not in self.models:
            group = list(self.models.keys())[0] if self.models else None
            if group is None:
                return 0.0

        try:
            # Prepare features (no scaling!)
            feature_cols = [c for c in self.feature_cols if c in features_df.columns]
            X = features_df[feature_cols].iloc[-1:].values

            adjustment = self.models[group].predict(X)[0]
            return float(adjustment)

        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

    def save(self, path='strike_model_v3.pkl'):
        """Save all models"""
        model_data = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'group_mapping': self.group_mapping,
            'ticker_groups': self.TICKER_GROUPS,
            'forward_days': self.forward_days,
            'version': 3
        }
        joblib.dump(model_data, path)
        print(f"\nModel V3 saved to {path}")

    def load(self, path='strike_model_v3.pkl'):
        """Load models"""
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.feature_cols = model_data['feature_cols']
        self.group_mapping = model_data['group_mapping']
        self.forward_days = model_data['forward_days']
        print(f"Model V3 loaded from {path}")
        print(f"Groups: {list(self.models.keys())}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train V3 Residual Strike Model')
    parser.add_argument('--period', default='3y', help='Data period')
    parser.add_argument('--output', default='strike_model_v3.pkl', help='Output path')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--tune', action='store_true', help='Enable Optuna hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials per group')
    args = parser.parse_args()

    model = ResidualStrikeModel(forward_days=35, use_gpu=not args.no_gpu)
    model.train_all(period=args.period, tune_hyperparams=args.tune, n_trials=args.trials)
    model.save(args.output)

    print("\n" + "="*70)
    print("V3 MODEL TRAINED - Two-Stage Residual Architecture")
    print("="*70)
    print("\nPrediction formula: final_prob = delta + model.predict_adjustment()")
    print("No hybrid blend needed - model output is directly usable!")
    if args.tune:
        print(f"\nHyperparameters optimized via Optuna ({args.trials} trials per group)")


if __name__ == "__main__":
    main()
