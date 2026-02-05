"""
STRIKE-SPECIFIC PROBABILITY MODEL
Predicts the probability distribution of maximum drawdown over 35 days.
For any strike, calculates P(stock drops below strike) - directly comparable to delta.

This is more accurate than binary classification because:
1. Predicts continuous distribution, not binary threshold
2. Works for ANY strike level, not just 5% OTM
3. Calibrated probabilities directly comparable to market delta
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_collector import CSPDataCollector

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False


class StrikeProbabilityModel:
    """
    Predicts strike-specific probabilities using quantile regression.

    Instead of binary classification (will it drop 5%?), this model predicts:
    - The expected maximum drawdown
    - Multiple quantiles of the drawdown distribution

    This allows calculating P(stock drops to any strike) for direct comparison with delta.
    """

    TICKERS = [
        # Core training tickers - diverse sectors and volatility profiles
        'NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN',
        'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'MRVL',
        'CRM', 'ADBE', 'NFLX', 'PYPL', 'NOW', 'PANW', 'CRWD',
        'JPM', 'BAC', 'GS', 'V', 'MA', 'C', 'WFC',
        'UNH', 'JNJ', 'PFE', 'LLY', 'MRK', 'ABBV',
        'COST', 'WMT', 'HD', 'LOW', 'SBUX', 'MCD', 'NKE',
        'XOM', 'CVX', 'COP',
        'CAT', 'DE', 'HON', 'UPS', 'RTX',
        'PLTR', 'COIN', 'MSTR',
        'SPY', 'QQQ', 'IWM', 'TQQQ', 'SCHD'
    ]

    # Quantiles to predict - covers the range of typical deltas
    QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    def __init__(self, forward_days=35):
        self.forward_days = forward_days
        self.scaler = StandardScaler()
        self.quantile_models = {}  # One model per quantile
        self.median_model = None   # For point estimate
        self.feature_cols = None

    def collect_data(self, tickers=None, period='10y'):
        """Collect and prepare training data from multiple tickers"""
        if tickers is None:
            tickers = self.TICKERS

        all_data = []

        for i, ticker in enumerate(tickers):
            print(f"[{i+1}/{len(tickers)}] Collecting {ticker}...")
            try:
                collector = CSPDataCollector(ticker, period=period)
                collector.fetch_data()
                collector.calculate_technical_indicators()

                df = collector.data.copy()

                # Calculate maximum drawdown (regression target)
                max_drawdown_list = []
                for j in range(len(df)):
                    if j + self.forward_days < len(df):
                        future_prices = df['Close'].iloc[j:j+self.forward_days+1]
                        current_price = df['Close'].iloc[j]
                        # Drawdown as positive percentage (how much it dropped)
                        min_price = future_prices.min()
                        drawdown_pct = ((current_price - min_price) / current_price) * 100
                        max_drawdown_list.append(drawdown_pct)
                    else:
                        max_drawdown_list.append(np.nan)

                df['Max_Drawdown_Pct'] = max_drawdown_list
                df['Ticker'] = ticker

                all_data.append(df)
                print(f"    {len(df)} samples, avg drawdown: {df['Max_Drawdown_Pct'].mean():.2f}%")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        self.data = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal: {len(self.data)} samples from {len(tickers)} tickers")

        return self.data

    def prepare_features(self):
        """Prepare feature matrix and target"""
        # Feature columns - exclude target and metadata
        exclude_cols = [
            'Max_Drawdown_Pct', 'Good_CSP_Time', 'Ticker', 'Date',
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
            'Forward_Return', 'Max_Drawdown_35D', 'Adjusted_Threshold',
            'Expected_Premium_Pct'
        ]

        self.feature_cols = [c for c in self.data.columns
                           if c not in exclude_cols
                           and self.data[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        # Drop NaN rows
        df_clean = self.data[self.feature_cols + ['Max_Drawdown_Pct']].dropna()

        X = df_clean[self.feature_cols].values
        y = df_clean['Max_Drawdown_Pct'].values

        print(f"Features: {len(self.feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Target range: {y.min():.2f}% to {y.max():.2f}%")
        print(f"Target mean: {y.mean():.2f}%, median: {np.median(y):.2f}%")

        return X, y

    def train_quantile_models(self, X, y, n_splits=5):
        """Train quantile regression models for each quantile"""

        print(f"\nTraining quantile regression models...")
        print(f"Quantiles: {self.QUANTILES}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for quantile in self.QUANTILES:
            print(f"\n  Training q={quantile:.2f} model...")

            # XGBoost with quantile loss
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=quantile,
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )

            # Cross-validation scores
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                # Quantile loss (pinball loss)
                y_pred = model.predict(X_val)
                errors = y_val - y_pred
                loss = np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors))
                cv_scores.append(loss)

            # Train final model on all data
            model.fit(X_scaled, y, verbose=False)
            self.quantile_models[quantile] = model

            print(f"    CV Pinball Loss: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Train median model for point estimates
        print(f"\n  Training median model...")
        self.median_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.median_model.fit(X_scaled, y, verbose=False)

        return self.quantile_models

    def predict_strike_probability(self, X, strike_pct):
        """
        Predict probability that stock drops to a given strike level.

        Args:
            X: Feature matrix (scaled)
            strike_pct: How far OTM the strike is as percentage (e.g., 5 for 5% OTM)

        Returns:
            Probability that max drawdown exceeds strike_pct
        """
        X_scaled = self.scaler.transform(X) if X.ndim > 1 else self.scaler.transform(X.reshape(1, -1))

        # Get predicted quantiles
        quantile_predictions = {}
        for q, model in self.quantile_models.items():
            quantile_predictions[q] = model.predict(X_scaled)

        # Interpolate to find probability that drawdown exceeds strike_pct
        # If strike_pct is between predicted quantiles, interpolate
        probabilities = []
        for i in range(len(X_scaled)):
            pred_quantiles = {q: quantile_predictions[q][i] for q in self.QUANTILES}

            # Find where strike_pct falls in the distribution
            # Quantile q means P(drawdown <= pred_q) = q
            # So P(drawdown > strike) = 1 - q where pred_q >= strike

            prob = self._interpolate_probability(pred_quantiles, strike_pct)
            probabilities.append(prob)

        return np.array(probabilities)

    def _interpolate_probability(self, pred_quantiles, strike_pct):
        """Interpolate probability from quantile predictions"""
        sorted_q = sorted(pred_quantiles.keys())
        sorted_pred = [pred_quantiles[q] for q in sorted_q]

        # If strike is below all predictions, probability is very high
        if strike_pct <= sorted_pred[0]:
            return 1.0 - sorted_q[0]  # Almost certain to hit

        # If strike is above all predictions, probability is very low
        if strike_pct >= sorted_pred[-1]:
            return 1.0 - sorted_q[-1]  # Very unlikely to hit

        # Linear interpolation
        for i in range(len(sorted_pred) - 1):
            if sorted_pred[i] <= strike_pct <= sorted_pred[i+1]:
                # Interpolate between quantiles
                frac = (strike_pct - sorted_pred[i]) / (sorted_pred[i+1] - sorted_pred[i])
                q_interp = sorted_q[i] + frac * (sorted_q[i+1] - sorted_q[i])
                return 1.0 - q_interp

        return 0.5  # Default fallback

    def save(self, path='strike_probability_model.pkl'):
        """Save the trained model"""
        model_data = {
            'quantile_models': self.quantile_models,
            'median_model': self.median_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'quantiles': self.QUANTILES,
            'forward_days': self.forward_days
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load(self, path='strike_probability_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(path)
        self.quantile_models = model_data['quantile_models']
        self.median_model = model_data['median_model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.QUANTILES = model_data['quantiles']
        self.forward_days = model_data['forward_days']
        print(f"Model loaded from {path}")

    def evaluate(self, X, y):
        """Evaluate model calibration"""
        X_scaled = self.scaler.transform(X)

        print("\nModel Calibration Check:")
        print("="*50)
        print(f"{'Quantile':<10} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
        print("-"*50)

        for q in self.QUANTILES:
            y_pred = self.quantile_models[q].predict(X_scaled)
            # What fraction of actual values fall below predicted quantile?
            actual_fraction = np.mean(y <= y_pred)
            error = actual_fraction - q
            print(f"{q:<10.2f} {q*100:<12.1f}% {actual_fraction*100:<12.1f}% {error*100:+.1f}%")

        # Overall calibration score (mean absolute error)
        calibration_errors = []
        for q in self.QUANTILES:
            y_pred = self.quantile_models[q].predict(X_scaled)
            actual_fraction = np.mean(y <= y_pred)
            calibration_errors.append(abs(actual_fraction - q))

        print("-"*50)
        print(f"Mean Calibration Error: {np.mean(calibration_errors)*100:.2f}%")

        return np.mean(calibration_errors)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Strike-Specific Probability Model')
    parser.add_argument('--tickers', nargs='+', help='Tickers to train on (default: all)')
    parser.add_argument('--period', default='10y', help='Data period (default: 10y)')
    parser.add_argument('--output', default='strike_probability_model.pkl', help='Output model path')
    args = parser.parse_args()

    print("="*70)
    print("STRIKE-SPECIFIC PROBABILITY MODEL TRAINING")
    print("="*70)
    print("\nThis model predicts P(stock drops to any strike level)")
    print("Output is directly comparable to market delta")
    print()

    model = StrikeProbabilityModel(forward_days=35)

    # Collect data
    tickers = args.tickers if args.tickers else None
    model.collect_data(tickers=tickers, period=args.period)

    # Prepare features
    X, y = model.prepare_features()

    # Train quantile models
    model.train_quantile_models(X, y)

    # Evaluate calibration
    model.evaluate(X, y)

    # Save model
    model.save(args.output)

    # Demo: predict probabilities for different strikes
    print("\n" + "="*70)
    print("DEMO: Strike Probability Predictions")
    print("="*70)

    # Use last sample as example
    X_sample = X[-1:, :]
    print(f"\nPredicted probabilities for sample (actual drawdown: {y[-1]:.2f}%):")
    for strike in [2.5, 5.0, 7.5, 10.0, 15.0, 20.0]:
        prob = model.predict_strike_probability(X_sample, strike)[0]
        print(f"  Strike {strike:5.1f}% OTM: P(breach) = {prob*100:5.1f}% (like delta {prob:.2f})")


if __name__ == "__main__":
    main()
