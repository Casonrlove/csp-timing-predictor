#!/usr/bin/env python
"""
Run model validation report
Usage: python validate_model.py [--ticker NVDA] [--period 5y]
"""

import argparse
from model_validator import ModelValidator
from data_collector import CSPDataCollector


def main():
    parser = argparse.ArgumentParser(description='Validate CSP Timing Model')
    parser.add_argument('--ticker', default='NVDA', help='Ticker for validation data')
    parser.add_argument('--period', default='5y', help='Data period (1y, 2y, 5y, 10y)')
    parser.add_argument('--stats-only', action='store_true', help='Only show prediction log stats')
    args = parser.parse_args()

    validator = ModelValidator()

    if args.stats_only:
        print("\nPrediction Log Statistics")
        print("="*50)
        stats = validator.get_prediction_accuracy()
        if 'message' in stats:
            print(stats['message'])
        else:
            print(f"Total Predictions: {stats['total_predictions']}")
            print(f"Accuracy: {stats['accuracy']:.1%}")
            print(f"Brier Score: {stats['brier_score']:.4f}")
            print("\nBy Ticker:")
            for ticker, data in stats['by_ticker'].items():
                print(f"  {ticker}: {data['accuracy']:.1%} ({data['n']} predictions)")

        print("\nRecent Predictions:")
        for pred in validator.get_recent_predictions(5):
            outcome = "✓" if pred['outcome'] == 1 else "✗" if pred['outcome'] == 0 else "?"
            print(f"  {pred['timestamp'][:10]} {pred['ticker']}: CSP Score={pred['csp_score']:.2f} → {outcome}")
        return

    if validator.model is None:
        print("ERROR: No model found. Train a model first with: ./start --train")
        return

    print(f"\nLoading validation data for {args.ticker} ({args.period})...")

    # Collect data
    collector = CSPDataCollector(args.ticker, period=args.period)
    collector.fetch_data()
    collector.calculate_technical_indicators()
    collector.create_target_variable()

    # Generate LSTM features if validator has LSTM model
    if validator.lstm_generator is not None:
        print("Generating LSTM features...")
        try:
            lstm_features = validator.lstm_generator.generate_features(collector.data)
            for col in lstm_features.columns:
                collector.data[col] = lstm_features[col].values
            print(f"  Added {len(lstm_features.columns)} LSTM features")
        except Exception as e:
            print(f"  Failed to generate LSTM features: {e}")

    df = collector.data.dropna()

    # Get feature columns - use model's expected features
    if validator.feature_names:
        feature_cols = [c for c in validator.feature_names if c in df.columns]
    else:
        exclude_cols = ['Good_CSP_Time', 'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]

    print(f"Using {len(feature_cols)} features for validation")

    # Run full validation report
    report = validator.generate_validation_report(df, feature_cols)

    # Save report
    import json
    with open('validation_report.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(report), f, indent=2)
        print("\n✓ Report saved to validation_report.json")


if __name__ == "__main__":
    main()
