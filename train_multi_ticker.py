"""
Train CSP timing model on multiple tickers for better generalization
This creates a model that works well across different stocks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from data_collector import CSPDataCollector


class MultiTickerTrainer:
    """Train model on multiple tickers"""

    def __init__(self, tickers=None, model_type='random_forest'):
        """
        Initialize trainer.

        Args:
            tickers: List of stock tickers to train on
            model_type: 'random_forest' (default) or 'xgboost'
        """
        if tickers is None:
            # Default tech-heavy watchlist
            self.tickers = ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TQQQ', 'SQQQ', 'PLTR', 'VOO']
        else:
            self.tickers = tickers

        self.scaler = StandardScaler()
        self.model = None
        self.feature_cols = None
        self.model_type = model_type.lower()

    def collect_data(self, period='10y'):
        """Collect data from all tickers"""
        print(f"\nCollecting data from {len(self.tickers)} tickers...")
        print("="*70)

        all_data = []

        for ticker in self.tickers:
            try:
                print(f"\nFetching {ticker}...")
                collector = CSPDataCollector(ticker, period=period)
                df, feature_cols = collector.get_training_data()

                if self.feature_cols is None:
                    self.feature_cols = feature_cols

                # Add ticker column for reference
                df['ticker'] = ticker

                all_data.append(df)
                print(f"  ✓ {ticker}: {len(df)} samples, {df['Good_CSP_Time'].mean():.1%} positive")

            except Exception as e:
                print(f"  ✗ {ticker}: Error - {str(e)}")
                continue

        if not all_data:
            raise ValueError("No data collected from any ticker")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        print("\n" + "="*70)
        print(f"Total samples: {len(combined_df)}")
        print(f"Positive class: {combined_df['Good_CSP_Time'].mean():.1%}")
        print(f"Samples per ticker:")
        print(combined_df.groupby('ticker').size())

        return combined_df

    def train(self, df):
        """Train model on combined data"""
        print("\n" + "="*70)
        print("TRAINING MULTI-TICKER MODEL")
        print("="*70)

        X = df[self.feature_cols]
        y = df['Good_CSP_Time']

        # Chronological split (important for time series)
        # Split by date index if available, otherwise simple split
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        print(f"\nTrain: {len(X_train)} samples ({y_train.mean():.1%} positive)")
        print(f"Test: {len(X_test)} samples ({y_test.mean():.1%} positive)")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the selected model type
        print("\n" + "="*70)

        if self.model_type == 'xgboost':
            print("Training XGBoost...")
            print("="*70)

            # Calculate scale_pos_weight for imbalanced classes
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='auc'
            )

            # XGBoost works better with scaled features
            self.model.fit(X_train_scaled, y_train)

        else:  # Default: random_forest
            print("Training Random Forest...")
            print("="*70)

            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

            # Random Forest doesn't need scaled features
            self.model.fit(X_train, y_train)

        # Evaluate (use scaled features for XGBoost)
        if self.model_type == 'xgboost':
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print("\n" + "="*70)
        print("MULTI-TICKER MODEL - TEST SET RESULTS")
        print("="*70)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")

        # Test on each ticker separately
        print("\n" + "="*70)
        print("PER-TICKER PERFORMANCE")
        print("="*70)

        test_df = df.iloc[split_idx:]
        for ticker in self.tickers:
            ticker_mask = test_df['ticker'] == ticker
            if ticker_mask.sum() == 0:
                continue

            ticker_X = X_test[ticker_mask]
            ticker_y = y_test[ticker_mask]

            if len(ticker_X) < 10:
                continue

            if self.model_type == 'xgboost':
                ticker_X_scaled = self.scaler.transform(ticker_X)
                ticker_pred_proba = self.model.predict_proba(ticker_X_scaled)[:, 1]
            else:
                ticker_pred_proba = self.model.predict_proba(ticker_X)[:, 1]
            ticker_roc = roc_auc_score(ticker_y, ticker_pred_proba)

            print(f"{ticker:6s}: {len(ticker_X):4d} samples | ROC-AUC: {ticker_roc:.4f}")

        return self.model

    def save_model(self, filename='csp_model_multi.pkl'):
        """Save the trained model"""
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': type(self.model).__name__,
            'tickers': self.tickers
        }

        joblib.dump(model_package, filename)
        print(f"\n✓ Model saved to {filename}")

    def plot_feature_importance(self):
        """Plot feature importance"""
        import matplotlib.pyplot as plt

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [self.feature_cols[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances - Multi-Ticker Model')
        plt.tight_layout()
        plt.savefig('feature_importance_multi.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved to feature_importance_multi.png")
        plt.close()

        # Print top features
        print("\nTop 10 Most Important Features:")
        for i, idx in enumerate(indices[:10], 1):
            print(f"{i:2d}. {self.feature_cols[idx]:<25s}: {importances[idx]:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train CSP timing model on multiple tickers')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost'],
                        help='Model type: random_forest (default) or xgboost')
    parser.add_argument('--tickers', type=str, nargs='+',
                        default=['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
                        help='List of tickers to train on')
    parser.add_argument('--period', type=str, default='10y',
                        help='Data period: 1y, 2y, 5y, 10y (default: 10y)')
    parser.add_argument('--output', type=str, default='csp_model_multi.pkl',
                        help='Output model filename')

    args = parser.parse_args()

    print("="*70)
    print("MULTI-TICKER CSP TIMING MODEL")
    print("="*70)
    print(f"\nModel type: {args.model.upper()}")
    print(f"Training on: {', '.join(args.tickers)}")
    print(f"Data period: {args.period}")
    print("This creates a model that generalizes across different stocks")
    print("")

    # Train
    trainer = MultiTickerTrainer(args.tickers, model_type=args.model)
    df = trainer.collect_data(period=args.period)
    trainer.train(df)
    trainer.plot_feature_importance()
    trainer.save_model(args.output)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {args.output}")
    print(f"Model type: {args.model}")
    print("\nTo use the model:")
    print("  python predictor.py NVDA")
    print("  (it will automatically use csp_model_multi.pkl if available)")
    print("\nTo train with XGBoost:")
    print("  python train_multi_ticker.py --model xgboost")
    print("")
