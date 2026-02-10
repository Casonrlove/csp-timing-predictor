"""
Strike Probability Calculator

Uses the trained quantile regression model to predict strike-specific probabilities.
Provides direct comparison to market delta for edge calculation.

V2: Now supports ticker-specific models trained on actual option outcomes.
"""

import numpy as np
import pandas as pd
import joblib
import os


# Try to load V2 model
try:
    from train_strike_model_v2 import TickerSpecificStrikeModel
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False


class StrikeProbabilityCalculatorV2:
    """
    V2 Calculator using ticker-specific models trained on option outcomes.
    """

    def __init__(self, model_path='strike_model_v2.pkl'):
        self.model_loaded = False
        self.model = None

        if os.path.exists(model_path):
            self.load(model_path)

    def load(self, path):
        """Load V2 model"""
        try:
            model_data = joblib.load(path)
            if model_data.get('version') == 2:
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.feature_cols = model_data['feature_cols']
                self.group_mapping = model_data['group_mapping']
                self.ticker_groups = model_data['ticker_groups']
                self.model_loaded = True
                print(f"[StrikeProb V2] Loaded ticker-specific models: {list(self.models.keys())}")
            else:
                print(f"[StrikeProb V2] Not a V2 model file")
        except Exception as e:
            print(f"[StrikeProb V2] Failed to load: {e}")
            self.model_loaded = False

    def get_ticker_group(self, ticker):
        """Get volatility group for ticker"""
        return self.group_mapping.get(ticker, 'tech')

    def predict_strike_probability(self, features_df, strike_otm_pct, ticker='NVDA'):
        """
        Predict probability of expiring ITM.

        Args:
            features_df: DataFrame with feature columns
            strike_otm_pct: How far OTM (e.g., 5 for 5% OTM)
            ticker: Stock ticker for group selection

        Returns:
            Probability (0-1) of expiring ITM
        """
        if not self.model_loaded:
            return None

        group = self.get_ticker_group(ticker)

        if group not in self.models or group not in self.scalers:
            group = list(self.models.keys())[0] if self.models else None
            if group is None:
                return None

        try:
            # Convert strike_otm_pct to approximate delta
            # FIX: Use IV estimate instead of realized vol (same as training fix)
            rv = features_df.get('Volatility_20D', pd.Series([20])).iloc[-1] / 100
            if rv <= 0:
                rv = 0.20

            # Use IV_RV_Ratio to estimate IV
            iv_rv_ratio = features_df.get('IV_RV_Ratio', pd.Series([1.2])).iloc[-1]
            iv_rv_ratio = max(1.0, min(iv_rv_ratio, 2.5))
            vol = rv * iv_rv_ratio  # IV estimate

            T = 35 / 365

            from scipy.stats import norm
            # Approximate delta from strike OTM percentage using IV
            target_delta = 1 - norm.cdf(strike_otm_pct / (vol * np.sqrt(T) * 100))
            target_delta = max(0.05, min(0.50, target_delta))

            # Prepare features
            row = features_df.iloc[-1:].copy()
            row['Target_Delta'] = target_delta

            # Build feature list matching training: base features + Target_Delta
            feature_cols_full = self.feature_cols + ['Target_Delta']

            # Check for missing features
            missing = [c for c in feature_cols_full if c not in row.columns]
            if missing:
                for col in missing:
                    row[col] = 0

            X = row[feature_cols_full].values
            X_scaled = self.scalers[group].transform(X)

            # Use probability model
            prob_model = self.models[group].get('prob')
            if prob_model:
                prob = prob_model.predict_proba(X_scaled)[0, 1]
                return float(np.clip(prob, 0.01, 0.99))

            return None

        except Exception as e:
            print(f"[StrikeProb V2] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_for_delta(self, features_df, target_delta, ticker='NVDA'):
        """
        Predict ITM probability for a given delta directly.

        Uses a hybrid approach:
        - Model predicts market-condition adjustment factor
        - Blended with delta as baseline to get realistic probabilities

        Args:
            features_df: DataFrame with feature columns
            target_delta: The delta (0.1 to 0.5)
            ticker: Stock ticker

        Returns:
            Probability of expiring ITM
        """
        if not self.model_loaded:
            return target_delta  # Fallback to delta itself

        group = self.get_ticker_group(ticker)

        if group not in self.models or group not in self.scalers:
            group = list(self.models.keys())[0] if self.models else None
            if group is None:
                return target_delta

        try:
            row = features_df.iloc[-1:].copy()
            row['Target_Delta'] = target_delta

            # Build feature list matching training: base features + Target_Delta
            feature_cols_full = self.feature_cols + ['Target_Delta']

            # Check for missing features
            missing = [c for c in feature_cols_full if c not in row.columns]
            if missing:
                for col in missing:
                    row[col] = 0

            X = row[feature_cols_full].values
            X_scaled = self.scalers[group].transform(X)

            prob_model = self.models[group].get('prob')
            if prob_model:
                raw_prob = prob_model.predict_proba(X_scaled)[0, 1]

                # Hybrid approach: Blend model with delta as baseline
                # Use weighted average to prevent extreme predictions
                #
                # If raw_prob is reasonable (close to delta), trust it more
                # If raw_prob is extreme, blend toward delta

                # Scale raw_prob by delta to get relative adjustment
                # A raw_prob of 0.15 at delta 0.30 means model thinks 50% of market estimate
                if raw_prob < 0.01:
                    # Model is extremely confident (near 0) - blend heavily with delta
                    model_weight = 0.3
                elif raw_prob > 0.5:
                    # Model predicts high risk - blend moderately with delta
                    model_weight = 0.5
                else:
                    # Model in reasonable range - trust it more
                    model_weight = 0.6

                # Weighted blend: heavier weight on delta to anchor predictions
                delta_weight = 1.0 - model_weight
                blended_prob = (model_weight * raw_prob) + (delta_weight * target_delta)

                # Cap edge to reasonable range: +/- 15% of delta
                max_edge = target_delta * 0.15  # Max 15% edge relative to delta
                min_prob = target_delta - max_edge
                max_prob = target_delta + max_edge

                final_prob = float(np.clip(blended_prob, max(0.02, min_prob), min(0.95, max_prob)))
                return final_prob

            return target_delta

        except Exception as e:
            print(f"[StrikeProb V2] Prediction error for {ticker} delta={target_delta}: {e}")
            import traceback
            traceback.print_exc()
            return target_delta


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
        self.calibrator = None
        self.calibration_method = None
        self.is_calibrated = False

        # Try calibrated model first, fall back to uncalibrated
        calibrated_path = model_path.replace('.pkl', '_calibrated.pkl')
        if os.path.exists(calibrated_path):
            self.load(calibrated_path)
        elif os.path.exists(model_path):
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

            # Load calibration if present
            if model_data.get('is_calibrated'):
                self.calibrator = model_data['calibrator']
                self.calibration_method = model_data['calibration_method']
                self.is_calibrated = True
                print(f"[StrikeProb] Calibrated model loaded from {path} ({self.calibration_method})")
            else:
                print(f"[StrikeProb] Model loaded from {path} (uncalibrated)")

            self.model_loaded = True
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

        # Apply calibration if available
        prob = self._apply_calibration(prob)

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
            prob = self._interpolate_probability(quantile_predictions, strike_pct)
            prob = self._apply_calibration(prob)
            results[strike_pct] = prob

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

    def _apply_calibration(self, prob):
        """Apply calibration function to raw probability"""
        # No bias correction - return raw model probability
        # Model should be accurate from training
        return prob

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


class StrikeProbabilityCalculatorV3:
    """
    V3 Calculator using two-stage residual model.
    
    Prediction: final_prob = delta + model_adjustment
    - Delta provides baseline (market's risk-neutral probability)
    - Model predicts adjustment based on market conditions
    - No hybrid blend hack needed
    """

    def __init__(self, model_path='strike_model_v3.pkl'):
        self.model_loaded = False
        self.max_edge_cap = 0.10  # Safety cap: max 10% edge (can increase with confidence)

        if os.path.exists(model_path):
            self.load(model_path)

    def load(self, path):
        """Load V3 model"""
        try:
            model_data = joblib.load(path)
            if model_data.get('version') == 3:
                self.models = model_data['models']
                self.feature_cols = model_data['feature_cols']
                self.group_mapping = model_data['group_mapping']
                self.ticker_groups = model_data['ticker_groups']
                self.model_loaded = True
                print(f"[StrikeProb V3] Loaded residual models: {list(self.models.keys())}")
            else:
                print(f"[StrikeProb V3] Not a V3 model file")
        except Exception as e:
            print(f"[StrikeProb V3] Failed to load: {e}")

    def get_ticker_group(self, ticker):
        """Get the volatility group for a ticker"""
        return self.group_mapping.get(ticker, 'tech')

    def predict_for_delta(self, features_df, target_delta, ticker='NVDA'):
        """
        Predict ITM probability using two-stage approach.
        
        Stage 1: Delta is the baseline
        Stage 2: Model predicts adjustment
        
        Args:
            features_df: DataFrame with feature columns
            target_delta: The delta (0.1 to 0.5)
            ticker: Stock ticker
            
        Returns:
            Probability of expiring ITM
        """
        if not self.model_loaded:
            return target_delta  # Fallback to delta

        group = self.get_ticker_group(ticker)

        if group not in self.models:
            group = list(self.models.keys())[0] if self.models else None
            if group is None:
                return target_delta

        try:
            # Add delta interaction features
            row = features_df.iloc[-1:].copy()
            
            # Calculate IV for interactions
            rv = row.get('Volatility_20D', pd.Series([20])).iloc[0] / 100
            iv_rv_ratio = row.get('IV_RV_Ratio', pd.Series([1.2])).iloc[0]
            vol = rv * max(1.0, min(iv_rv_ratio, 2.5))
            
            # Add interaction features (must match training)
            row['Target_Delta'] = target_delta
            row['Delta_x_Vol'] = target_delta * vol
            row['Delta_x_VIX'] = target_delta * row.get('VIX', pd.Series([15])).iloc[0] / 100
            row['Delta_x_IV_Rank'] = target_delta * row.get('IV_Rank', pd.Series([50])).iloc[0] / 100
            row['Delta_Squared'] = target_delta ** 2
            row['Delta_x_RSI'] = target_delta * row.get('RSI', pd.Series([50])).iloc[0] / 100

            # Get features (no scaling!)
            feature_cols_available = [c for c in self.feature_cols if c in row.columns]
            
            # Check for missing
            missing = [c for c in self.feature_cols if c not in row.columns]
            if missing:
                for col in missing:
                    row[col] = 0

            X = row[self.feature_cols].values

            # Get model adjustment
            adjustment = self.models[group].predict(X)[0]

            # Apply safety cap
            adjustment = np.clip(adjustment, -self.max_edge_cap, self.max_edge_cap)

            # Final probability = delta + adjustment
            final_prob = target_delta + adjustment

            # Ensure valid probability range
            final_prob = float(np.clip(final_prob, 0.01, 0.99))

            return final_prob

        except Exception as e:
            print(f"[StrikeProb V3] Prediction error for {ticker} delta={target_delta}: {e}")
            import traceback
            traceback.print_exc()
            return target_delta

    def predict_strike_probability(self, features_df, strike_otm_pct, ticker='NVDA'):
        """
        Predict probability from strike OTM percentage.
        Converts strike to delta, then uses predict_for_delta.
        """
        if not self.model_loaded:
            return None

        try:
            # Convert strike_otm_pct to delta using IV estimate
            rv = features_df.get('Volatility_20D', pd.Series([20])).iloc[-1] / 100
            if rv <= 0:
                rv = 0.20

            iv_rv_ratio = features_df.get('IV_RV_Ratio', pd.Series([1.2])).iloc[-1]
            iv_rv_ratio = max(1.0, min(iv_rv_ratio, 2.5))
            vol = rv * iv_rv_ratio

            T = 35 / 365

            from scipy.stats import norm
            target_delta = 1 - norm.cdf(strike_otm_pct / (vol * np.sqrt(T) * 100))
            target_delta = max(0.05, min(0.50, target_delta))

            # Use predict_for_delta
            return self.predict_for_delta(features_df, target_delta, ticker)

        except Exception as e:
            print(f"[StrikeProb V3] Strike prediction error: {e}")
            return None
