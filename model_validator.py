"""
Model Validation and Backtesting Module
Provides forward validation, P/L tracking, and probability calibration metrics
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, precision_score, recall_score
import joblib


class ModelValidator:
    """Comprehensive model validation and backtesting"""

    def __init__(self, model_path='csp_model_multi.pkl', log_path='predictions_log.json',
                 lstm_path='csp_model_multi_lstm.pt'):
        self.model_path = model_path
        self.log_path = log_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.lstm_generator = None

        # Load model
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names') or model_data.get('feature_cols')

        # Load LSTM model if needed
        LSTM_FEATURES = ['LSTM_Return_5D', 'LSTM_Return_10D', 'LSTM_Direction_Prob']
        if self.feature_names and any(f in self.feature_names for f in LSTM_FEATURES):
            if os.path.exists(lstm_path):
                try:
                    from lstm_features import LSTMFeatureGenerator
                    self.lstm_generator = LSTMFeatureGenerator()
                    self.lstm_generator.load(lstm_path)
                    print(f"✓ LSTM model loaded for validation")
                except Exception as e:
                    print(f"⚠ Failed to load LSTM model: {e}")

        # Load or initialize prediction log
        self.predictions_log = self._load_log()

    def _load_log(self):
        """Load existing prediction log"""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except:
                return {'predictions': [], 'outcomes': []}
        return {'predictions': [], 'outcomes': []}

    def _save_log(self):
        """Save prediction log"""
        with open(self.log_path, 'w') as f:
            json.dump(self.predictions_log, f, indent=2, default=str)

    # =========================================================================
    # 1. BRIER SCORE - Probability Calibration
    # =========================================================================

    def calculate_brier_score(self, y_true, y_prob):
        """
        Calculate Brier score for probability calibration

        Brier score = mean((predicted_prob - actual_outcome)^2)
        - 0.0 = perfect calibration
        - 0.25 = random guessing (50% predictions)
        - Lower is better

        Returns dict with Brier score and interpretation
        """
        brier = brier_score_loss(y_true, y_prob)

        # Calibration buckets - are 70% predictions actually right 70% of time?
        calibration = self._calculate_calibration_curve(y_true, y_prob)

        return {
            'brier_score': round(brier, 4),
            'interpretation': self._interpret_brier(brier),
            'calibration_curve': calibration,
            'log_loss': round(log_loss(y_true, y_prob), 4)
        }

    def _interpret_brier(self, brier):
        if brier < 0.1:
            return "Excellent calibration"
        elif brier < 0.15:
            return "Good calibration"
        elif brier < 0.2:
            return "Fair calibration"
        elif brier < 0.25:
            return "Poor calibration (near random)"
        else:
            return "Very poor calibration (worse than random)"

    def _calculate_calibration_curve(self, y_true, y_prob, n_bins=10):
        """Calculate calibration curve - predicted vs actual probability by bucket"""
        bins = np.linspace(0, 1, n_bins + 1)
        calibration = []

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() > 0:
                predicted_avg = y_prob[mask].mean()
                actual_avg = y_true[mask].mean()
                count = mask.sum()
                calibration.append({
                    'bin': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    'predicted_prob': round(predicted_avg, 3),
                    'actual_rate': round(actual_avg, 3),
                    'count': int(count),
                    'gap': round(abs(predicted_avg - actual_avg), 3)
                })

        return calibration

    # =========================================================================
    # 2. WALK-FORWARD BACKTEST
    # =========================================================================

    def walk_forward_backtest(self, df, feature_cols, initial_train_pct=0.6, step_size=20):
        """
        Walk-forward validation - train on past, predict future, roll forward

        Args:
            df: DataFrame with features and 'Good_CSP_Time' target
            feature_cols: List of feature column names
            initial_train_pct: Initial training data percentage
            step_size: Number of days to step forward each iteration

        Returns:
            Dict with detailed backtest results
        """
        if 'Good_CSP_Time' not in df.columns:
            raise ValueError("DataFrame must have 'Good_CSP_Time' column")

        results = []
        n = len(df)
        initial_train_size = int(n * initial_train_pct)

        print(f"\nWalk-Forward Backtest")
        print(f"{'='*60}")
        print(f"Total samples: {n}")
        print(f"Initial train size: {initial_train_size}")
        print(f"Step size: {step_size} days")

        train_end = initial_train_size

        while train_end < n - step_size:
            # Train on data up to train_end
            X_train = df[feature_cols].iloc[:train_end].values
            y_train = df['Good_CSP_Time'].iloc[:train_end].values

            # Test on next step_size days
            test_start = train_end
            test_end = min(train_end + step_size, n)

            X_test = df[feature_cols].iloc[test_start:test_end].values
            y_test = df['Good_CSP_Time'].iloc[test_start:test_end].values

            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Predict
            y_pred = self.model.predict(X_test_scaled)
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics for this window
            accuracy = accuracy_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_prob)

            results.append({
                'train_end_idx': train_end,
                'test_start_idx': test_start,
                'test_end_idx': test_end,
                'accuracy': round(accuracy, 4),
                'brier_score': round(brier, 4),
                'n_test': len(y_test),
                'positive_rate': round(y_test.mean(), 4)
            })

            # Step forward
            train_end += step_size

        # Aggregate results
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_brier = np.mean([r['brier_score'] for r in results])

        print(f"\nResults ({len(results)} windows):")
        print(f"  Average Accuracy: {avg_accuracy:.2%}")
        print(f"  Average Brier Score: {avg_brier:.4f}")

        return {
            'windows': results,
            'summary': {
                'n_windows': len(results),
                'avg_accuracy': round(avg_accuracy, 4),
                'avg_brier': round(avg_brier, 4),
                'std_accuracy': round(np.std([r['accuracy'] for r in results]), 4),
                'min_accuracy': round(min(r['accuracy'] for r in results), 4),
                'max_accuracy': round(max(r['accuracy'] for r in results), 4)
            }
        }

    # =========================================================================
    # 3. P/L TRACKER - Simulated Trading
    # =========================================================================

    def simulate_csp_trading(self, df, predictions, probabilities,
                              premium_pct=0.02, assignment_loss_pct=0.05,
                              csp_score_threshold=0.0, confidence_threshold=0.5):
        """
        Simulate P/L from following model's CSP recommendations

        Args:
            df: DataFrame with 'Close' prices and actual outcomes
            predictions: Model predictions (0 or 1)
            probabilities: Model probability of safe outcome
            premium_pct: Average premium collected as % of strike (default 2%)
            assignment_loss_pct: Average loss when assigned (default 5%)
            csp_score_threshold: Minimum CSP score to trade
            confidence_threshold: Minimum p_safe to trade

        Returns:
            Dict with P/L results and trade log
        """
        trades = []
        total_pnl = 0
        total_premium = 0
        total_losses = 0
        n_trades = 0
        n_wins = 0
        n_losses = 0

        # Assume 'Good_CSP_Time' column exists (1 = stock didn't drop >5%)
        actuals = df['Good_CSP_Time'].values if 'Good_CSP_Time' in df.columns else None
        prices = df['Close'].values

        for i in range(len(predictions)):
            p_safe = probabilities[i]
            p_downside = 1 - p_safe
            csp_score = p_safe - p_downside

            # Only trade if meets thresholds
            if csp_score >= csp_score_threshold and p_safe >= confidence_threshold:
                # We would sell a CSP here
                strike = prices[i] * 0.95  # Assume 5% OTM
                premium = strike * premium_pct

                # Determine outcome
                if actuals is not None:
                    actual_safe = actuals[i]
                    if actual_safe == 1:
                        # CSP expired worthless - keep premium
                        pnl = premium
                        n_wins += 1
                        total_premium += premium
                    else:
                        # Got assigned - loss
                        pnl = premium - (strike * assignment_loss_pct)
                        n_losses += 1
                        total_losses += abs(pnl)

                    total_pnl += pnl
                    n_trades += 1

                    trades.append({
                        'index': i,
                        'price': round(prices[i], 2),
                        'strike': round(strike, 2),
                        'p_safe': round(p_safe, 3),
                        'csp_score': round(csp_score, 3),
                        'actual_safe': int(actual_safe),
                        'premium': round(premium, 2),
                        'pnl': round(pnl, 2),
                        'cumulative_pnl': round(total_pnl, 2)
                    })

        win_rate = n_wins / n_trades if n_trades > 0 else 0
        avg_win = total_premium / n_wins if n_wins > 0 else 0
        avg_loss = total_losses / n_losses if n_losses > 0 else 0
        profit_factor = total_premium / total_losses if total_losses > 0 else float('inf')

        return {
            'summary': {
                'total_trades': n_trades,
                'wins': n_wins,
                'losses': n_losses,
                'win_rate': round(win_rate, 4),
                'total_pnl': round(total_pnl, 2),
                'total_premium_collected': round(total_premium, 2),
                'total_assignment_losses': round(total_losses, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'expected_value_per_trade': round(total_pnl / n_trades, 2) if n_trades > 0 else 0
            },
            'trades': trades[-50:],  # Last 50 trades
            'equity_curve': [t['cumulative_pnl'] for t in trades]
        }

    # =========================================================================
    # 4. DAILY PREDICTION LOG
    # =========================================================================

    def log_prediction(self, ticker, price, p_safe, p_downside, csp_score,
                       suggested_strike=None, suggested_expiration=None):
        """
        Log a prediction for future validation

        Call this whenever a prediction is made to track accuracy over time
        """
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'price': price,
            'p_safe': p_safe,
            'p_downside': p_downside,
            'csp_score': csp_score,
            'prediction': 'SAFE' if p_safe > p_downside else 'RISKY',
            'suggested_strike': suggested_strike,
            'suggested_expiration': suggested_expiration,
            'outcome': None,  # To be filled in later
            'outcome_date': None
        }

        self.predictions_log['predictions'].append(prediction)
        self._save_log()

        return prediction

    def record_outcome(self, ticker, prediction_date, actual_outcome, actual_return=None):
        """
        Record the actual outcome for a past prediction

        Args:
            ticker: Stock ticker
            prediction_date: Date prediction was made (ISO format)
            actual_outcome: 1 if stock didn't drop >5%, 0 if it did
            actual_return: Actual return over the period (optional)
        """
        for pred in self.predictions_log['predictions']:
            if pred['ticker'] == ticker and pred['timestamp'].startswith(prediction_date):
                pred['outcome'] = actual_outcome
                pred['outcome_date'] = datetime.now().isoformat()
                pred['actual_return'] = actual_return
                self._save_log()
                return True
        return False

    def get_prediction_accuracy(self):
        """Calculate accuracy of logged predictions that have outcomes"""
        predictions_with_outcomes = [
            p for p in self.predictions_log['predictions']
            if p['outcome'] is not None
        ]

        if not predictions_with_outcomes:
            return {'message': 'No predictions with outcomes yet'}

        correct = 0
        total = len(predictions_with_outcomes)

        by_ticker = {}
        calibration_data = {'y_true': [], 'y_prob': []}

        for p in predictions_with_outcomes:
            predicted_safe = p['p_safe'] > 0.5
            actual_safe = p['outcome'] == 1

            if predicted_safe == actual_safe:
                correct += 1

            # Track by ticker
            ticker = p['ticker']
            if ticker not in by_ticker:
                by_ticker[ticker] = {'correct': 0, 'total': 0}
            by_ticker[ticker]['total'] += 1
            if predicted_safe == actual_safe:
                by_ticker[ticker]['correct'] += 1

            # For Brier score
            calibration_data['y_true'].append(p['outcome'])
            calibration_data['y_prob'].append(p['p_safe'])

        # Calculate Brier score on logged predictions
        brier = brier_score_loss(
            calibration_data['y_true'],
            calibration_data['y_prob']
        )

        return {
            'total_predictions': total,
            'correct': correct,
            'accuracy': round(correct / total, 4),
            'brier_score': round(brier, 4),
            'by_ticker': {
                k: {
                    'accuracy': round(v['correct'] / v['total'], 4),
                    'n': v['total']
                }
                for k, v in by_ticker.items()
            }
        }

    def get_recent_predictions(self, n=20):
        """Get most recent predictions"""
        return self.predictions_log['predictions'][-n:]

    # =========================================================================
    # COMPREHENSIVE VALIDATION REPORT
    # =========================================================================

    def generate_validation_report(self, df, feature_cols):
        """Generate comprehensive validation report"""
        print("\n" + "="*70)
        print("MODEL VALIDATION REPORT")
        print("="*70)

        # Prepare data
        X = df[feature_cols].values
        y = df['Good_CSP_Time'].values

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]

        # 1. Basic Metrics
        print("\n1. BASIC METRICS")
        print("-"*40)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        print(f"   Accuracy:  {accuracy:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall:    {recall:.2%}")

        # 2. Brier Score
        print("\n2. PROBABILITY CALIBRATION")
        print("-"*40)
        brier_results = self.calculate_brier_score(y, y_prob)
        print(f"   Brier Score: {brier_results['brier_score']:.4f}")
        print(f"   Log Loss:    {brier_results['log_loss']:.4f}")
        print(f"   Assessment:  {brier_results['interpretation']}")

        print("\n   Calibration by Probability Bucket:")
        for bucket in brier_results['calibration_curve']:
            gap_indicator = "✓" if bucket['gap'] < 0.1 else "⚠" if bucket['gap'] < 0.2 else "✗"
            print(f"   {bucket['bin']}: predicted={bucket['predicted_prob']:.0%}, actual={bucket['actual_rate']:.0%}, n={bucket['count']} {gap_indicator}")

        # 3. Walk-Forward Backtest
        print("\n3. WALK-FORWARD BACKTEST")
        print("-"*40)
        wf_results = self.walk_forward_backtest(df, feature_cols)

        # 4. P/L Simulation
        print("\n4. SIMULATED P/L (following model recommendations)")
        print("-"*40)
        pnl_results = self.simulate_csp_trading(df, y_pred, y_prob)
        summary = pnl_results['summary']
        print(f"   Total Trades:    {summary['total_trades']}")
        print(f"   Win Rate:        {summary['win_rate']:.1%}")
        print(f"   Total P/L:       ${summary['total_pnl']:,.2f}")
        print(f"   Profit Factor:   {summary['profit_factor']:.2f}")
        print(f"   Avg Win:         ${summary['avg_win']:.2f}")
        print(f"   Avg Loss:        ${summary['avg_loss']:.2f}")
        print(f"   EV per Trade:    ${summary['expected_value_per_trade']:.2f}")

        # 5. Prediction Log Status
        print("\n5. PREDICTION LOG STATUS")
        print("-"*40)
        log_accuracy = self.get_prediction_accuracy()
        if 'message' in log_accuracy:
            print(f"   {log_accuracy['message']}")
        else:
            print(f"   Logged Predictions: {log_accuracy['total_predictions']}")
            print(f"   Accuracy:           {log_accuracy['accuracy']:.1%}")
            print(f"   Brier Score:        {log_accuracy['brier_score']:.4f}")

        print("\n" + "="*70)

        return {
            'basic_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            },
            'brier': brier_results,
            'walk_forward': wf_results,
            'pnl_simulation': pnl_results,
            'prediction_log': log_accuracy
        }


if __name__ == "__main__":
    # Test the validator
    from data_collector import CSPDataCollector

    print("Testing Model Validator...")

    # Load some data
    collector = CSPDataCollector('NVDA', period='5y')
    collector.fetch_data()
    collector.calculate_technical_indicators()
    collector.create_target_variable()

    df = collector.data.dropna()

    # Get feature columns (exclude target and date columns)
    exclude_cols = ['Good_CSP_Time', 'Date', 'Ticker']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]

    # Run validation
    validator = ModelValidator()

    if validator.model is not None:
        report = validator.generate_validation_report(df, feature_cols[:27])  # Use first 27 features
    else:
        print("No model found. Train a model first with train_ultimate.py")
