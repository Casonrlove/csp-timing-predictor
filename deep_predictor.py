"""
Predictor for GPU-accelerated deep learning models
"""

import torch
import numpy as np
import joblib
from data_collector import CSPDataCollector
from deep_learning_model import LSTMModel, TransformerModel, HybridModel
from pytorch_tabnet.tab_model import TabNetClassifier


class DeepLearningPredictor:
    """Predictor for deep learning CSP timing models"""
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_package = joblib.load(model_path)
        self.model_type = self.model_package['model_type']
        self.scaler = self.model_package['scaler']
        self.feature_cols = self.model_package['feature_cols']

        # Load model based on type
        if self.model_type == 'TabNet':
            self.model = TabNetClassifier()
            self.model.load_model(self.model_package['model_path'] + '.zip')
            self.seq_length = None
        else:
            self.seq_length = self.model_package['seq_length']
            input_size = self.model_package['input_size']

            # Create model architecture
            if self.model_type == 'lstm':
                self.model = LSTMModel(input_size)
            elif self.model_type == 'transformer':
                self.model = TransformerModel(input_size)
            elif self.model_type == 'hybrid':
                self.model = HybridModel(input_size)

            # Load weights
            self.model.load_state_dict(self.model_package['model_state'])
            self.model = self.model.to(self.device)
            self.model.eval()

        print(f"Loaded {self.model_type} model on {self.device}")

    def get_current_features(self, ticker='NVDA'):
        """Get recent features for prediction"""
        print(f"\nFetching data for {ticker}...")

        # Get more historical data for sequences
        collector = CSPDataCollector(ticker, period='2y')
        collector.fetch_data()
        collector.calculate_technical_indicators()

        df = collector.data
        features_df = df[self.feature_cols].dropna()

        if len(features_df) == 0:
            raise ValueError("No valid data available")

        print(f"Data as of: {df.index[-1].strftime('%Y-%m-%d')}")

        return features_df, df

    def predict(self, ticker='NVDA', show_details=True):
        """Make prediction for ticker"""
        features_df, full_data = self.get_current_features(ticker)

        # Scale features
        features_scaled = self.scaler.transform(features_df.values)

        # Make prediction
        if self.model_type == 'TabNet':
            # TabNet uses most recent point
            X = features_scaled[-1:, :]
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
        else:
            # Sequence models need sequence
            if len(features_scaled) < self.seq_length:
                raise ValueError(f"Need at least {self.seq_length} days of data")

            # Get most recent sequence
            X_seq = features_scaled[-self.seq_length:]
            X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(X_tensor)
                prob_good = output.cpu().numpy()[0][0]

            probability = [1 - prob_good, prob_good]
            prediction = 1 if prob_good > 0.5 else 0

        current_price = full_data['Close'].iloc[-1]

        print("\n" + "="*70)
        print(f"CSP TIMING PREDICTION FOR {ticker} ({self.model_type.upper()})")
        print("="*70)

        print(f"\nCurrent Price: ${current_price:.2f}")
        print(f"Date: {full_data.index[-1].strftime('%Y-%m-%d')}")

        print(f"\n{'RECOMMENDATION:':<20} {'GOOD TIME TO SELL CSP' if prediction == 1 else 'NOT IDEAL - WAIT'}")
        print(f"{'Confidence:':<20} {probability[prediction]:.1%}")
        print(f"{'Probability Good:':<20} {probability[1]:.1%}")
        print(f"{'Probability Bad:':<20} {probability[0]:.1%}")

        if show_details:
            self.show_technical_context(ticker, full_data, features_df.iloc[-1])

        return {
            'ticker': ticker,
            'prediction': 'GOOD' if prediction == 1 else 'WAIT',
            'confidence': probability[prediction],
            'prob_good': probability[1],
            'prob_bad': probability[0],
            'current_price': current_price,
            'date': full_data.index[-1],
            'model_type': self.model_type
        }

    def show_technical_context(self, ticker, full_data, features):
        """Show key technical indicators"""
        print("\n" + "-"*70)
        print("TECHNICAL CONTEXT")
        print("-"*70)

        # Support/Resistance
        print("\n[Support/Resistance]")
        if 'Distance_From_Support_Pct' in features.index:
            print(f"  Distance from Support:    {features['Distance_From_Support_Pct']:>6.2f}%")
        if 'Distance_From_Resistance_Pct' in features.index:
            print(f"  Distance from Resistance: {features['Distance_From_Resistance_Pct']:>6.2f}%")

        # Moving Averages
        print("\n[Moving Averages]")
        if 'Price_to_SMA20' in features.index:
            print(f"  Price vs SMA20:  {features['Price_to_SMA20']:>6.2f}%")
        if 'Price_to_SMA50' in features.index:
            print(f"  Price vs SMA50:  {features['Price_to_SMA50']:>6.2f}%")
        if 'Price_to_SMA200' in features.index:
            print(f"  Price vs SMA200: {features['Price_to_SMA200']:>6.2f}%")

        # Momentum
        print("\n[Momentum]")
        if 'RSI' in features.index:
            rsi = features['RSI']
            status = '(OVERSOLD)' if rsi < 30 else '(OVERBOUGHT)' if rsi > 70 else ''
            print(f"  RSI:             {rsi:>6.2f} {status}")

        # Volatility
        print("\n[Volatility]")
        if 'ATR_Pct' in features.index:
            print(f"  ATR %:           {features['ATR_Pct']:>6.2f}%")
        if 'VIX' in features.index:
            print(f"  VIX:             {features['VIX']:>6.2f}")

        print()

    def compare_models_prediction(self, ticker, model_paths):
        """Compare predictions from multiple models"""
        print("\n" + "="*70)
        print(f"COMPARING MODELS FOR {ticker}")
        print("="*70)

        results = []
        for model_path in model_paths:
            try:
                predictor = DeepLearningPredictor(model_path)
                result = predictor.predict(ticker, show_details=False)
                results.append(result)
            except Exception as e:
                print(f"\nERROR with {model_path}: {str(e)}")

        # Summary table
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Model':<20} {'Prediction':<12} {'Prob Good':<12} {'Confidence':<12}")
        print("-"*70)

        for result in results:
            print(f"{result['model_type']:<20} {result['prediction']:<12} "
                  f"{result['prob_good']:<12.1%} {result['confidence']:<12.1%}")

        # Ensemble prediction (average probabilities)
        if len(results) > 1:
            avg_prob_good = np.mean([r['prob_good'] for r in results])
            ensemble_pred = 'GOOD' if avg_prob_good > 0.5 else 'WAIT'

            print("\n" + "-"*70)
            print(f"{'ENSEMBLE (Average)':<20} {ensemble_pred:<12} {avg_prob_good:<12.1%}")

        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python deep_predictor.py <model_path> <ticker>")
        print("Example: python deep_predictor.py csp_hybrid_model.pkl NVDA")
        sys.exit(1)

    model_path = sys.argv[1]
    ticker = sys.argv[2].upper()

    try:
        predictor = DeepLearningPredictor(model_path)
        predictor.predict(ticker, show_details=True)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
