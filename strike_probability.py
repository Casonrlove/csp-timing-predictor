"""
Strike Probability Calculator

Uses the trained quantile regression model to predict strike-specific probabilities.
Provides direct comparison to market delta for edge calculation.
"""

import numpy as np
import joblib
import os


class StrikeProbabilityCalculator:
    """
    Calculate probability of stock reaching any strike level.

    Uses quantile regression model trained on historical drawdown data.
    Output is directly comparable to option delta.
    """

    def __init__(self, model_path='strike_probability_model.pkl'):
        self.model_loaded = False
        self.quantile_models = None
        self.median_model = None
        self.scaler = None
        self.feature_cols = None
        self.quantiles = None

        if os.path.exists(model_path):
            self.load(model_path)

    def load(self, path):
        """Load trained model"""
        try:
            model_data = joblib.load(path)
            self.quantile_models = model_data['quantile_models']
            self.median_model = model_data['median_model']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            self.quantiles = model_data['quantiles']
            self.model_loaded = True
            print(f"[StrikeProb] Model loaded from {path}")
        except Exception as e:
            print(f"[StrikeProb] Failed to load model: {e}")
            self.model_loaded = False

    def predict_strike_probability(self, features_df, strike_otm_pct):
        """
        Predict probability that stock drops to a given strike level.

        Args:
            features_df: DataFrame with feature columns (from CSPDataCollector)
            strike_otm_pct: How far OTM the strike is as percentage (e.g., 5.0 for 5% OTM)

        Returns:
            float: Probability (0-1) that max drawdown exceeds strike_otm_pct
                   This is directly comparable to option delta
        """
        if not self.model_loaded:
            return None

        # Extract features in correct order
        try:
            X = features_df[self.feature_cols].iloc[-1:].values
            X_scaled = self.scaler.transform(X)
        except KeyError as e:
            print(f"[StrikeProb] Missing features: {e}")
            return None

        # Get predicted quantiles
        quantile_predictions = {}
        for q, model in self.quantile_models.items():
            quantile_predictions[q] = model.predict(X_scaled)[0]

        # Interpolate to find probability
        prob = self._interpolate_probability(quantile_predictions, strike_otm_pct)
        return prob

    def predict_all_strikes(self, features_df, strike_otm_pcts):
        """
        Predict probabilities for multiple strikes at once.

        Args:
            features_df: DataFrame with feature columns
            strike_otm_pcts: List of OTM percentages

        Returns:
            dict: {strike_pct: probability}
        """
        if not self.model_loaded:
            return {}

        try:
            X = features_df[self.feature_cols].iloc[-1:].values
            X_scaled = self.scaler.transform(X)
        except KeyError as e:
            print(f"[StrikeProb] Missing features: {e}")
            return {}

        # Get all quantile predictions once
        quantile_predictions = {}
        for q, model in self.quantile_models.items():
            quantile_predictions[q] = model.predict(X_scaled)[0]

        # Calculate probability for each strike
        results = {}
        for strike_pct in strike_otm_pcts:
            results[strike_pct] = self._interpolate_probability(quantile_predictions, strike_pct)

        return results

    def _interpolate_probability(self, pred_quantiles, strike_pct):
        """Interpolate probability from quantile predictions"""
        sorted_q = sorted(pred_quantiles.keys())
        sorted_pred = [pred_quantiles[q] for q in sorted_q]

        # If strike is below all predictions, probability is very high
        if strike_pct <= sorted_pred[0]:
            # Extrapolate: very likely to hit this small drawdown
            return min(0.99, 1.0 - sorted_q[0] * (strike_pct / sorted_pred[0]))

        # If strike is above all predictions, probability is very low
        if strike_pct >= sorted_pred[-1]:
            # Extrapolate: unlikely to hit this large drawdown
            return max(0.01, (1.0 - sorted_q[-1]) * (sorted_pred[-1] / strike_pct))

        # Linear interpolation between quantiles
        for i in range(len(sorted_pred) - 1):
            if sorted_pred[i] <= strike_pct <= sorted_pred[i+1]:
                frac = (strike_pct - sorted_pred[i]) / (sorted_pred[i+1] - sorted_pred[i])
                q_interp = sorted_q[i] + frac * (sorted_q[i+1] - sorted_q[i])
                return 1.0 - q_interp

        return 0.5  # Fallback

    def get_expected_drawdown(self, features_df):
        """Get expected (median) maximum drawdown"""
        if not self.model_loaded or self.median_model is None:
            return None

        try:
            X = features_df[self.feature_cols].iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            return self.median_model.predict(X_scaled)[0]
        except Exception as e:
            print(f"[StrikeProb] Error predicting drawdown: {e}")
            return None


def calculate_edge(delta, model_probability):
    """
    Calculate edge: difference between market's view and model's view.

    Args:
        delta: Market delta (implied probability of ITM)
        model_probability: Model's predicted probability of reaching strike

    Returns:
        Edge in percentage points (positive = favorable for selling)
    """
    # Delta is typically negative for puts, we use absolute value
    market_prob = abs(delta)
    edge = (market_prob - model_probability) * 100
    return edge


# Convenience function for API integration
def get_strike_probability(features_df, current_price, strike_price, model=None):
    """
    Get model probability for a specific strike.

    Args:
        features_df: DataFrame with features
        current_price: Current stock price
        strike_price: Option strike price
        model: StrikeProbabilityCalculator instance (or loads default)

    Returns:
        tuple: (model_probability, edge_vs_delta)
    """
    if model is None:
        model = StrikeProbabilityCalculator()

    if not model.model_loaded:
        return None, None

    # Calculate how far OTM the strike is
    otm_pct = ((current_price - strike_price) / current_price) * 100

    # Get model probability
    model_prob = model.predict_strike_probability(features_df, otm_pct)

    return model_prob, otm_pct
