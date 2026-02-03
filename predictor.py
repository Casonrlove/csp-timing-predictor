"""
CSP Timing Predictor - Real-time predictions for optimal CSP selling opportunities
"""

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from data_collector import CSPDataCollector


class CSPPredictor:
    def __init__(self, model_path=None):
        # Auto-detect best available model
        if model_path is None:
            if os.path.exists('csp_model_multi.pkl'):
                model_path = 'csp_model_multi.pkl'
                print("Using multi-ticker model (better generalization)")
            elif os.path.exists('csp_model.pkl'):
                model_path = 'csp_model.pkl'
            else:
                raise FileNotFoundError("No trained model found. Please run model_trainer.py first.")
        """Load trained model"""
        self.model_package = joblib.load(model_path)
        self.model = self.model_package['model']
        self.scaler = self.model_package.get('scaler')
        self.feature_cols = self.model_package['feature_cols']
        self.model_type = self.model_package['model_type']

        print(f"Loaded {self.model_type} model")
        print(f"Features: {len(self.feature_cols)}")

    def get_current_features(self, ticker='NVDA'):
        """Get current features for a ticker"""
        print(f"\nFetching current data for {ticker}...")

        # Use data collector to get recent data and calculate features
        # Need at least 2 years for 200-day indicators
        collector = CSPDataCollector(ticker, period='2y')
        collector.fetch_data()
        collector.calculate_technical_indicators()

        # Get the most recent row with all features
        df = collector.data

        # Get latest complete data point (drop any NaN)
        latest = df[self.feature_cols].iloc[-1]

        # Check for NaN and warn
        if latest.isna().any():
            print(f"WARNING: Some features have NaN values. Missing: {latest[latest.isna()].index.tolist()}")
            # Fill NaN with mean from recent data
            for col in latest[latest.isna()].index:
                latest[col] = df[col].iloc[-100:].mean()

        print(f"Data as of: {df.index[-1].strftime('%Y-%m-%d')}")

        return latest, df

    def predict(self, ticker='NVDA', show_details=True):
        """Predict if it's a good time to sell CSP on ticker"""
        # Get current features
        features, full_data = self.get_current_features(ticker)

        # Prepare features for prediction
        X = features.values.reshape(1, -1)

        # Scale if needed
        if self.scaler is not None and 'Logistic' in self.model_type:
            X = self.scaler.transform(X)

        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        # Get current price info
        current_price = full_data['Close'].iloc[-1]

        print("\n" + "="*70)
        print(f"CSP TIMING PREDICTION FOR {ticker}")
        print("="*70)

        print(f"\nCurrent Price: ${current_price:.2f}")
        print(f"Date: {full_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n{'RECOMMENDATION:':<20} {'GOOD TIME TO SELL CSP' if prediction == 1 else 'NOT IDEAL - WAIT'}")
        print(f"{'Confidence:':<20} {probability[prediction]:.1%}")
        print(f"{'Probability Good:':<20} {probability[1]:.1%}")
        print(f"{'Probability Bad:':<20} {probability[0]:.1%}")

        if show_details:
            self.show_technical_context(ticker, full_data, features)

        return {
            'ticker': ticker,
            'prediction': 'GOOD' if prediction == 1 else 'WAIT',
            'confidence': probability[prediction],
            'prob_good': probability[1],
            'prob_bad': probability[0],
            'current_price': current_price,
            'date': full_data.index[-1]
        }

    def show_technical_context(self, ticker, full_data, features):
        """Show key technical indicators providing context"""
        print("\n" + "-"*70)
        print("TECHNICAL CONTEXT")
        print("-"*70)

        latest = full_data.iloc[-1]

        # Support/Resistance Analysis
        print("\n[Support/Resistance]")
        print(f"  Distance from Support:    {features['Distance_From_Support_Pct']:>6.2f}%")
        print(f"  Distance from Resistance: {features['Distance_From_Resistance_Pct']:>6.2f}%")
        print(f"  Distance from 20D Low:    {features['Distance_From_Low_20D']:>6.2f}%")

        # Moving Average Analysis
        print("\n[Moving Averages]")
        print(f"  Price vs SMA20:  {features['Price_to_SMA20']:>6.2f}%")
        print(f"  Price vs SMA50:  {features['Price_to_SMA50']:>6.2f}%")
        print(f"  Price vs SMA200: {features['Price_to_SMA200']:>6.2f}%")

        # Momentum
        print("\n[Momentum Indicators]")
        print(f"  RSI:             {features['RSI']:>6.2f} {'(OVERSOLD)' if features['RSI'] < 30 else '(OVERBOUGHT)' if features['RSI'] > 70 else ''}")
        print(f"  Stochastic K:    {features['Stoch_K']:>6.2f}")
        print(f"  MACD:            {features['MACD']:>6.2f}")

        # Volatility
        print("\n[Volatility]")
        print(f"  ATR %:           {features['ATR_Pct']:>6.2f}%")
        print(f"  20D Volatility:  {features['Volatility_20D']:>6.2f}%")
        print(f"  BB Position:     {features['BB_Position']:>6.2f}% {'(Near Lower Band)' if features['BB_Position'] < 20 else '(Near Upper Band)' if features['BB_Position'] > 80 else ''}")

        # Recent Returns
        print("\n[Recent Performance]")
        print(f"  1-Day Return:    {features['Return_1D']:>6.2f}%")
        print(f"  5-Day Return:    {features['Return_5D']:>6.2f}%")
        print(f"  20-Day Return:   {features['Return_20D']:>6.2f}%")

        # Market Context
        print("\n[Market Context]")
        print(f"  VIX:             {features['VIX']:>6.2f}")

        # Interpretation
        print("\n" + "-"*70)
        print("KEY INSIGHTS")
        print("-"*70)

        insights = []

        # Support level check
        if features['Distance_From_Support_Pct'] < 2:
            insights.append("• STRONG: Price is very close to support level (good CSP entry)")
        elif features['Distance_From_Support_Pct'] < 5:
            insights.append("• Price is near support level")

        # RSI check
        if features['RSI'] < 35:
            insights.append("• STRONG: RSI shows oversold conditions (potential bounce)")
        elif features['RSI'] > 65:
            insights.append("• CAUTION: RSI shows overbought conditions")

        # Bollinger Band check
        if features['BB_Position'] < 20:
            insights.append("• STRONG: Price at lower Bollinger Band (support area)")
        elif features['BB_Position'] > 80:
            insights.append("• CAUTION: Price at upper Bollinger Band")

        # Moving average support
        if -2 < features['Price_to_SMA20'] < 0:
            insights.append("• Price testing SMA20 support")
        if -2 < features['Price_to_SMA50'] < 0:
            insights.append("• Price testing SMA50 support")

        # Recent selloff
        if features['Return_5D'] < -3:
            insights.append("• Recent selloff may present opportunity")

        # Trend check
        if features['Price_to_SMA200'] > 5:
            insights.append("• Strong uptrend (above 200 SMA)")
        elif features['Price_to_SMA200'] < -5:
            insights.append("• CAUTION: Downtrend (below 200 SMA)")

        if insights:
            for insight in insights:
                print(insight)
        else:
            print("• No strong signals detected")

        print()

    def scan_multiple_tickers(self, tickers):
        """Scan multiple tickers and rank by CSP opportunity"""
        print("\n" + "="*70)
        print("SCANNING MULTIPLE TICKERS FOR CSP OPPORTUNITIES")
        print("="*70)

        results = []

        for ticker in tickers:
            try:
                result = self.predict(ticker, show_details=False)
                results.append(result)
                print(f"\n{ticker}: {result['prediction']} (Confidence: {result['confidence']:.1%})")
            except Exception as e:
                print(f"\n{ticker}: ERROR - {str(e)}")

        # Sort by probability of good timing
        results.sort(key=lambda x: x['prob_good'], reverse=True)

        print("\n" + "="*70)
        print("RANKED OPPORTUNITIES")
        print("="*70)
        print(f"\n{'Rank':<6} {'Ticker':<8} {'Recommendation':<12} {'Prob Good':<12} {'Current Price':<15}")
        print("-"*70)

        for i, result in enumerate(results, 1):
            print(f"{i:<6} {result['ticker']:<8} {result['prediction']:<12} "
                  f"{result['prob_good']:<12.1%} ${result['current_price']:<14.2f}")

        return results


if __name__ == "__main__":
    import sys

    # Check if model exists
    try:
        predictor = CSPPredictor('csp_model.pkl')

        # Get ticker from command line or use default
        ticker = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'

        if ',' in ticker:
            # Multiple tickers
            tickers = [t.strip().upper() for t in ticker.split(',')]
            predictor.scan_multiple_tickers(tickers)
        else:
            # Single ticker
            predictor.predict(ticker.upper(), show_details=True)

    except FileNotFoundError:
        print("ERROR: Model file 'csp_model.pkl' not found.")
        print("Please run 'python model_trainer.py' first to train the model.")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
