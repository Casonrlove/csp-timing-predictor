"""
Simple FastAPI server for CSP timing predictions
Supports both Schwab API (preferred) and Yahoo Finance (fallback) for options data
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import joblib
import os
from datetime import datetime, timedelta
from data_collector import CSPDataCollector
from options_analyzer import get_best_csp_option
from options_analyzer_multi import get_all_csp_options

# Try to import Schwab client (optional)
try:
    from schwab_client import (
        get_csp_options_schwab,
        is_market_open,
        get_market_hours,
        get_stock_price,
        get_quotes
    )
    SCHWAB_AVAILABLE = True
    print("✓ Schwab API client loaded")
except ImportError as e:
    SCHWAB_AVAILABLE = False
    print(f"⚠ Schwab API not available: {e}")
    print("  Using Yahoo Finance for market data")

app = FastAPI(
    title="CSP Timing API",
    description="Predict optimal timing for selling Cash Secured Puts",
    version="1.0.0"
)

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    ticker: str
    min_delta: Optional[float] = 0.10
    max_delta: Optional[float] = 0.60


class PredictionResponse(BaseModel):
    ticker: str
    prediction: str
    confidence: float  # |p_safe - p_downside| - model decisiveness (0=unsure, 1=decisive)
    p_safe: float  # Probability stock won't drop >5% (good for CSP)
    p_downside: float  # Probability of >5% drop (bad for CSP)
    csp_score: float  # p_safe - p_downside (-1 to +1, higher = better trade)
    # Legacy fields for backward compatibility
    prob_good: float
    prob_bad: float
    current_price: float
    date: str
    model_type: str
    technical_context: dict
    options_data: Optional[dict] = None
    all_options: Optional[list] = None
    options_by_expiration: Optional[dict] = None  # Options grouped by monthly expiration
    available_expirations: Optional[list] = None  # List of available monthly expirations
    data_source: Optional[dict] = None  # Shows where data came from (Schwab vs Yahoo)
    market_status: Optional[dict] = None  # Market open/closed status from Schwab
    price_history: Optional[list] = None  # Historical price data for charting


class PriceHistoryRequest(BaseModel):
    ticker: str
    days: Optional[int] = 90  # Default 90 days of history


class MultiTickerRequest(BaseModel):
    tickers: List[str]


# Load model at startup
MODEL_PATH = None
MODEL = None
SCALER = None
FEATURE_NAMES = None
LSTM_GENERATOR = None

for path in ['csp_model_multi.pkl', 'csp_model.pkl']:
    if os.path.exists(path):
        MODEL_PATH = path
        print(f"Loading model from {MODEL_PATH}...")
        model_data = joblib.load(MODEL_PATH)
        MODEL = model_data['model']
        SCALER = model_data['scaler']
        FEATURE_NAMES = model_data.get('feature_names') or model_data.get('feature_cols')
        print(f"✓ Model loaded: {MODEL}")
        break

if MODEL is None:
    print("ERROR: No trained model found!")

# Load LSTM model if it exists and features require it
LSTM_FEATURES = ['LSTM_Return_5D', 'LSTM_Return_10D', 'LSTM_Direction_Prob']
if FEATURE_NAMES and any(f in FEATURE_NAMES for f in LSTM_FEATURES):
    lstm_path = 'csp_model_multi_lstm.pt'
    if os.path.exists(lstm_path):
        try:
            from lstm_features import LSTMFeatureGenerator
            LSTM_GENERATOR = LSTMFeatureGenerator()
            LSTM_GENERATOR.load(lstm_path)
            print(f"✓ LSTM model loaded from {lstm_path}")
        except Exception as e:
            print(f"⚠ Failed to load LSTM model: {e}")
            LSTM_GENERATOR = None
    else:
        print(f"⚠ LSTM features required but {lstm_path} not found")


@app.get("/")
def read_root():
    """Health check"""
    return {
        "status": "running",
        "model": str(MODEL) if MODEL else "No model loaded",
        "model_file": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if MODEL else "error",
        "model_loaded": MODEL is not None,
        "model_file": MODEL_PATH,
        "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/reload")
def reload_model():
    """Reload the model from disk without restarting the server"""
    global MODEL, SCALER, FEATURE_NAMES, MODEL_PATH

    for path in ['csp_model_multi.pkl', 'csp_model.pkl']:
        if os.path.exists(path):
            MODEL_PATH = path
            print(f"Reloading model from {MODEL_PATH}...")
            model_data = joblib.load(MODEL_PATH)
            MODEL = model_data['model']
            SCALER = model_data['scaler']
            FEATURE_NAMES = model_data.get('feature_names') or model_data.get('feature_cols')
            print(f"✓ Model reloaded: {MODEL}")
            return {
                "status": "success",
                "message": f"Model reloaded from {MODEL_PATH}",
                "model_type": type(MODEL).__name__,
                "timestamp": datetime.now().isoformat()
            }

    raise HTTPException(status_code=404, detail="No model file found")


@app.get("/market/hours")
def market_hours():
    """Get market hours and check if market is open"""
    if not SCHWAB_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Schwab API not configured. Market hours not available.",
            "timestamp": datetime.now().isoformat()
        }

    try:
        hours = get_market_hours(['equity', 'option'])
        equity_open = is_market_open('equity')
        option_open = is_market_open('option')

        return {
            "status": "success",
            "equity_market_open": equity_open,
            "option_market_open": option_open,
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/token/status")
def token_status():
    """Check Schwab token expiration status"""
    if not SCHWAB_AVAILABLE:
        return {"status": "unavailable", "message": "Schwab API not configured"}

    try:
        import json
        from datetime import datetime
        config_path = os.path.join(os.path.dirname(__file__), 'schwab_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        if not config.get('token_expires_at'):
            return {"status": "unknown", "message": "No token expiration info"}

        # Access token expiry (30 min)
        access_expires = datetime.fromisoformat(config['token_expires_at'])
        access_remaining = (access_expires - datetime.now()).total_seconds()

        # Refresh token expires 7 days after last refresh
        # We estimate based on access token (refresh happens when access expires)
        refresh_expires_approx = access_expires + timedelta(days=7)
        refresh_remaining = (refresh_expires_approx - datetime.now()).total_seconds()
        refresh_days = refresh_remaining / 86400

        return {
            "status": "ok" if refresh_days > 1 else "expiring_soon",
            "access_token_expires_in_minutes": round(access_remaining / 60, 1),
            "refresh_token_expires_in_days": round(refresh_days, 1),
            "warning": "Re-authorize soon!" if refresh_days < 2 else None,
            "reauth_url": "https://api.schwabapi.com/v1/oauth/authorize?response_type=code&client_id=" + config['client_id'] + "&scope=readonly&redirect_uri=https%3A%2F%2Fdeveloper.schwab.com%2Foauth2-redirect.html" if refresh_days < 2 else None
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/market/status")
def market_status():
    """Quick check if markets are open"""
    if not SCHWAB_AVAILABLE:
        # Fallback: estimate based on time (rough check)
        now = datetime.now()
        # Market hours: 9:30 AM - 4:00 PM ET (roughly)
        hour = now.hour
        weekday = now.weekday()
        is_weekday = weekday < 5
        is_market_hours = 9 <= hour < 16

        return {
            "equity_open": is_weekday and is_market_hours,
            "option_open": is_weekday and is_market_hours,
            "source": "estimated",
            "note": "Schwab API not configured, using time-based estimate"
        }

    try:
        return {
            "equity_open": is_market_open('equity'),
            "option_open": is_market_open('option'),
            "source": "schwab"
        }
    except Exception as e:
        return {
            "equity_open": False,
            "option_open": False,
            "source": "error",
            "error": str(e)
        }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make CSP timing prediction for a single ticker"""
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        ticker = request.ticker.upper()

        # Collect data and features
        collector = CSPDataCollector(ticker, period='10y')
        collector.fetch_data()
        collector.calculate_technical_indicators()
        collector.create_target_variable(forward_days=35, threshold_pct=-5)

        # Generate LSTM features if model requires them
        if LSTM_GENERATOR is not None:
            try:
                lstm_features = LSTM_GENERATOR.generate_features(collector.data)
                for col in lstm_features.columns:
                    collector.data[col] = lstm_features[col].values
                print(f"[LSTM] Generated features for {ticker}")
            except Exception as lstm_err:
                print(f"[LSTM] Feature generation failed: {lstm_err}")
                # Fill with zeros if LSTM fails
                for col in LSTM_FEATURES:
                    if col not in collector.data.columns:
                        collector.data[col] = 0.0

        features_df = collector.data[FEATURE_NAMES].copy()

        if len(features_df) == 0:
            raise HTTPException(status_code=400, detail=f"No data available for {ticker}")

        # Get most recent features
        X_current = features_df.iloc[-1:].values
        X_scaled = SCALER.transform(X_current)

        # Make prediction
        prediction = MODEL.predict(X_scaled)[0]
        probabilities = MODEL.predict_proba(X_scaled)[0]

        # Calculate CSP metrics early (needed for trade gating)
        p_safe = float(probabilities[1])
        p_downside = float(probabilities[0])
        csp_score = p_safe - p_downside

        # Get current data - try Schwab first for real-time price
        current_data = collector.data.iloc[-1]
        price_source = "Yahoo"

        if SCHWAB_AVAILABLE:
            try:
                schwab_quote = get_stock_price(ticker)
                current_price = float(schwab_quote['price'])
                price_source = "Schwab"
                print(f"[Schwab] Current price for {ticker}: ${current_price:.2f}")
            except Exception as e:
                print(f"[Schwab] Price fetch failed: {e}, using Yahoo Finance")
                current_price = float(current_data['Close'])
        else:
            current_price = float(current_data['Close'])

        # Get ALL options data in the delta range
        # Try Schwab API first (real Greeks), fall back to Yahoo Finance + Black-Scholes
        all_options = []
        options_data = None
        options_source = "Yahoo+BS"
        try:
            print(f"Fetching options for {ticker} with delta range {request.min_delta}-{request.max_delta}...")

            if SCHWAB_AVAILABLE:
                try:
                    all_options = get_csp_options_schwab(ticker, min_delta=request.min_delta, max_delta=request.max_delta, max_dte=200)
                    if all_options:
                        options_source = "Schwab"
                        print(f"[Schwab] Found {len(all_options)} options")
                except Exception as schwab_err:
                    print(f"[Schwab] Failed: {schwab_err}, falling back to Yahoo Finance")
                    all_options = []

            # Fallback to Yahoo Finance + Black-Scholes if Schwab failed or not available
            if not all_options:
                all_options = get_all_csp_options(ticker, min_delta=request.min_delta, max_delta=request.max_delta)
                options_source = "Yahoo+BS"

            print(f"Found {len(all_options)} options via {options_source}")

            # Add edge calculation with SCALED model risk per strike
            model_prob_bad = float(probabilities[0])  # Model's prediction (prob of >5% drop)
            model_threshold_pct = 0.05  # Model threshold is 5% drop

            # Find the strike closest to 5% OTM to calibrate
            threshold_strike = current_price * (1 - model_threshold_pct)
            threshold_delta = None
            for opt in all_options:
                otm_pct = (current_price - opt['strike']) / current_price
                if abs(otm_pct - model_threshold_pct) < 0.02:
                    threshold_delta = abs(opt['delta'])
                    break

            # Risk factor: how model's view differs from market at threshold
            # e.g., model=21%, market delta at 5% OTM=16% → factor=1.31 (model more bearish)
            if threshold_delta and threshold_delta > 0:
                risk_factor = model_prob_bad / threshold_delta
            else:
                risk_factor = 1.0

            for option in all_options:
                market_delta = abs(option['delta'])

                # Scale each strike's risk by the factor
                model_risk_at_strike = market_delta * risk_factor
                model_risk_at_strike = max(0.01, min(0.60, model_risk_at_strike))

                # Edge = market - model (positive = good for selling)
                edge = (market_delta - model_risk_at_strike) * 100

                option['market_delta'] = round(market_delta, 3)
                option['model_prob_assignment'] = round(model_risk_at_strike, 3)
                option['edge'] = round(edge, 2)
                option['edge_signal'] = 'GOOD EDGE' if edge > 0 else 'NEGATIVE EDGE'
                option['is_suggested'] = False

                # Calculate Sharpe ratio: ROR / risk (model probability)
                # Higher Sharpe = better risk-adjusted return
                ror = option.get('ror', 0)
                if model_risk_at_strike > 0 and ror > 0:
                    option['sharpe'] = round(ror / (model_risk_at_strike * 100), 3)
                else:
                    option['sharpe'] = 0

            # Select best option - require BOTH CSP Score > 0 AND positive edge
            if all_options:
                # Gate 1: CSP Score must be positive (model favors safe outcome)
                if csp_score <= 0:
                    options_data = None
                    print(f"CSP Score {csp_score:.2f} <= 0 - no suggestion (model favors downside)")
                else:
                    # Gate 2: Find option with best positive edge
                    positive_edge_options = [opt for opt in all_options if opt.get('edge', 0) > 0]
                    if positive_edge_options:
                        best_edge_option = max(positive_edge_options, key=lambda x: x.get('edge', 0))
                        best_edge_option['is_suggested'] = True
                        options_data = best_edge_option
                        print(f"Best Edge: ${best_edge_option['strike']} strike, Edge={best_edge_option['edge']:.1f}, CSP Score={csp_score:.2f}")
                    else:
                        # No positive edge - don't suggest any option
                        options_data = None
                        print(f"CSP Score {csp_score:.2f} > 0 but no positive edge options - no suggestion")
            else:
                options_data = None
            print(f"Options data prepared: {len(all_options)} total, returning top 5")
        except Exception as e:
            print(f"ERROR fetching options data: {e}")
            import traceback
            traceback.print_exc()
            all_options = []
            options_data = None

        # Build technical context
        technical_context = {
            'support_resistance': {
                'distance_from_support': float(current_data.get('Distance_From_Support_Pct', 0)),
                'distance_from_resistance': float(current_data.get('Distance_From_Resistance_Pct', 0))
            },
            'moving_averages': {
                'price_to_sma20': float(current_data.get('Price_to_SMA20', 0)),
                'price_to_sma50': float(current_data.get('Price_to_SMA50', 0)),
                'price_to_sma200': float(current_data.get('Price_to_SMA200', 0))
            },
            'momentum': {
                'rsi': float(current_data.get('RSI', 0)),
                'macd': float(current_data.get('MACD', 0)),
                'macd_signal': float(current_data.get('MACD_Signal', 0))
            },
            'volatility': {
                'atr_pct': float(current_data.get('ATR_Pct', 0)),
                'bb_position': float(current_data.get('BB_Position', 0)),
                'iv_rank': float(current_data.get('IV_Rank', 0)),
                'vix': float(current_data.get('VIX', 15.0))
            },
            'earnings': {
                'days_to_earnings': float(current_data.get('Days_To_Earnings', 999)),
                'near_earnings': bool(current_data.get('Near_Earnings', False))
            }
        }

        # Group options by monthly expiration and return next 2 monthlies
        options_by_expiration = {}
        available_expirations = []
        display_options = []

        if all_options:
            # Group options by expiration
            for opt in all_options:
                exp = opt['expiration']
                if exp not in options_by_expiration:
                    options_by_expiration[exp] = {
                        'dte': opt['dte'],
                        'expiration': exp,
                        'options': []
                    }
                options_by_expiration[exp]['options'].append(opt)

            # Sort expirations by date and take next 3
            sorted_exps = sorted(options_by_expiration.keys())
            available_expirations = sorted_exps[:6]  # Next 6 monthly expirations

            # Sort strikes within each expiration (low to high = low delta at top, high delta at bottom)
            for exp in options_by_expiration:
                options_by_expiration[exp]['options'] = sorted(
                    options_by_expiration[exp]['options'],
                    key=lambda x: x['strike']
                )

            # For backward compatibility, use first expiration as default display
            if available_expirations:
                first_exp = available_expirations[0]
                display_options = options_by_expiration[first_exp]['options']

            # Only include the available monthly expirations in the response
            options_by_expiration = {
                exp: options_by_expiration[exp]
                for exp in available_expirations
                if exp in options_by_expiration
            }

        # Check market status if no options data available
        market_status = None
        if not all_options and SCHWAB_AVAILABLE:
            try:
                equity_open = is_market_open('equity')
                option_open = is_market_open('option')
                market_status = {
                    "equity_open": equity_open,
                    "option_open": option_open,
                    "message": "Market is OPEN" if option_open else "Market is CLOSED"
                }
                print(f"Market status: equity={equity_open}, options={option_open}")
            except Exception as e:
                print(f"Could not check market status: {e}")
                market_status = {"message": "Could not determine market status"}

        # Build price history for charting (last 180 days for 6-month view)
        price_history = []
        try:
            hist_df = collector.data.tail(180)
            for idx, row in hist_df.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
                price_history.append({
                    'date': date_str,
                    'close': round(float(row['Close']), 2),
                    'high': round(float(row['High']), 2),
                    'low': round(float(row['Low']), 2),
                    'sma20': round(float(row.get('SMA_20', row['Close'])), 2),
                    'sma50': round(float(row.get('SMA_50', row['Close'])), 2),
                    'sma200': round(float(row.get('SMA_200', row['Close'])), 2)
                })
        except Exception as hist_err:
            print(f"Failed to build price history: {hist_err}")

        # Calculate confidence (decisiveness) - p_safe, p_downside, csp_score already calculated above
        confidence = abs(csp_score)  # Model decisiveness (0=unsure, 1=decisive)

        # Log prediction for forward validation
        try:
            validator = get_validator()
            if validator:
                validator.log_prediction(
                    ticker=ticker,
                    price=current_price,
                    p_safe=p_safe,
                    p_downside=p_downside,
                    csp_score=csp_score,
                    suggested_strike=options_data['strike'] if options_data else None,
                    suggested_expiration=options_data['expiration'] if options_data else None
                )
        except Exception as log_err:
            print(f"Failed to log prediction: {log_err}")

        return PredictionResponse(
            ticker=ticker,
            prediction="GOOD TIME TO SELL CSP" if prediction == 1 else "WAIT - NOT OPTIMAL",
            confidence=confidence,
            p_safe=p_safe,
            p_downside=p_downside,
            csp_score=csp_score,
            # Legacy fields
            prob_good=p_safe,
            prob_bad=p_downside,
            current_price=current_price,
            date=datetime.now().strftime('%Y-%m-%d'),
            model_type=f"{type(MODEL).__name__} (Multi-Ticker)" if "multi" in MODEL_PATH else type(MODEL).__name__,
            technical_context=technical_context,
            options_data=options_data,
            all_options=display_options if display_options else None,
            options_by_expiration=options_by_expiration if options_by_expiration else None,
            available_expirations=available_expirations if available_expirations else None,
            data_source={
                "price": price_source,
                "options": options_source,
                "greeks": "Schwab API" if options_source == "Schwab" else "Black-Scholes (calculated)"
            },
            market_status=market_status,
            price_history=price_history if price_history else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test/options/{ticker}")
def test_options(ticker: str, min_delta: float = 0.25, max_delta: float = 0.35, debug: bool = True):
    """Debug endpoint to test options fetching"""
    try:
        print(f"TEST: Fetching options for {ticker}...")
        from options_analyzer_multi import get_all_csp_options
        options = get_all_csp_options(ticker, min_delta=min_delta, max_delta=max_delta, debug=debug)
        print(f"TEST: Found {len(options)} options")
        return {
            "status": "success",
            "ticker": ticker,
            "count": len(options),
            "first_3": options[:3] if options else []
        }
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"TEST ERROR: {error_detail}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": error_detail
        }


@app.post("/predict/multi")
def predict_multi(request: MultiTickerRequest):
    """Make predictions for multiple tickers"""
    results = []

    for ticker in request.tickers:
        try:
            pred_request = PredictionRequest(ticker=ticker)
            result = predict(pred_request)
            results.append(result.dict())
        except Exception as e:
            results.append({
                "ticker": ticker.upper(),
                "error": str(e)
            })

    # Sort by probability of good timing
    results_with_prob = [r for r in results if 'prob_good' in r]
    results_with_errors = [r for r in results if 'error' in r]

    results_with_prob.sort(key=lambda x: x['prob_good'], reverse=True)

    return {
        "results": results_with_prob + results_with_errors,
        "count": len(results),
        "timestamp": datetime.now().isoformat()
    }


# =========================================================================
# VALIDATION ENDPOINTS
# =========================================================================

# Initialize validator (lazy load)
_validator = None

def get_validator():
    global _validator
    if _validator is None:
        try:
            from model_validator import ModelValidator
            _validator = ModelValidator()
        except ImportError:
            return None
    return _validator


@app.get("/validation/stats")
def get_validation_stats():
    """Get prediction accuracy stats from logged predictions"""
    validator = get_validator()
    if validator is None:
        return {"status": "error", "message": "Validator not available"}

    accuracy = validator.get_prediction_accuracy()
    recent = validator.get_recent_predictions(10)

    return {
        "status": "success",
        "accuracy_stats": accuracy,
        "recent_predictions": recent,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/validation/record_outcome")
def record_outcome(ticker: str, prediction_date: str, outcome: int, actual_return: float = None):
    """
    Record the actual outcome for a past prediction

    Args:
        ticker: Stock ticker
        prediction_date: Date prediction was made (YYYY-MM-DD)
        outcome: 1 if stock didn't drop >5%, 0 if it did
        actual_return: Actual return over the period (optional)
    """
    validator = get_validator()
    if validator is None:
        return {"status": "error", "message": "Validator not available"}

    success = validator.record_outcome(ticker, prediction_date, outcome, actual_return)

    return {
        "status": "success" if success else "not_found",
        "message": f"Outcome recorded for {ticker}" if success else f"No prediction found for {ticker} on {prediction_date}"
    }


@app.get("/validation/recent")
def get_recent_predictions(n: int = 20):
    """Get recent logged predictions"""
    validator = get_validator()
    if validator is None:
        return {"status": "error", "message": "Validator not available"}

    return {
        "status": "success",
        "predictions": validator.get_recent_predictions(n),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run CSP Timing API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()

    print("="*70)
    print("CSP TIMING API SERVER (Random Forest)")
    print("="*70)
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Model: {MODEL_PATH}")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("="*70)

    uvicorn.run(
        "simple_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
