"""
Simple FastAPI server for CSP timing predictions using Random Forest model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import joblib
import os
from datetime import datetime
from data_collector import CSPDataCollector
from options_analyzer import get_best_csp_option
from options_analyzer_multi import get_all_csp_options

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
    max_delta: Optional[float] = 0.40


class PredictionResponse(BaseModel):
    ticker: str
    prediction: str
    confidence: float
    prob_good: float
    prob_bad: float
    current_price: float
    date: str
    model_type: str
    technical_context: dict
    options_data: Optional[dict] = None
    all_options: Optional[list] = None


class MultiTickerRequest(BaseModel):
    tickers: List[str]


# Load model at startup
MODEL_PATH = None
MODEL = None
SCALER = None
FEATURE_NAMES = None

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

        features_df = collector.data[FEATURE_NAMES].copy()

        if len(features_df) == 0:
            raise HTTPException(status_code=400, detail=f"No data available for {ticker}")

        # Get most recent features
        X_current = features_df.iloc[-1:].values
        X_scaled = SCALER.transform(X_current)

        # Make prediction
        prediction = MODEL.predict(X_scaled)[0]
        probabilities = MODEL.predict_proba(X_scaled)[0]

        # Get current data
        current_data = collector.data.iloc[-1]
        current_price = float(current_data['Close'])

        # Get ALL options data in the delta range
        all_options = []
        options_data = None
        try:
            print(f"Fetching options for {ticker} with delta range {request.min_delta}-{request.max_delta}...")
            all_options = get_all_csp_options(ticker, min_delta=request.min_delta, max_delta=request.max_delta)
            print(f"Found {len(all_options)} options")

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
                option['is_closest_atm'] = False

            if all_options:
                all_options[0]['is_closest_atm'] = True

            # Get the best one for backward compatibility
            options_data = all_options[0] if all_options else None
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

        # Pick ONE expiration (closest to 37 DTE) and show all strikes sorted high→low
        display_options = []
        if all_options:
            # Group options by expiration
            expirations = {}
            for opt in all_options:
                exp = opt['expiration']
                if exp not in expirations:
                    expirations[exp] = {'dte': opt['dte'], 'options': []}
                expirations[exp]['options'].append(opt)

            # Pick the expiration closest to 37 DTE (sweet spot for CSPs)
            best_exp = min(expirations.keys(), key=lambda x: abs(expirations[x]['dte'] - 37))

            # Get all strikes for that expiration, sorted by strike (high to low = high delta to low)
            display_options = sorted(expirations[best_exp]['options'],
                                    key=lambda x: -x['strike'])[:8]

        return PredictionResponse(
            ticker=ticker,
            prediction="GOOD TIME TO SELL CSP" if prediction == 1 else "WAIT - NOT OPTIMAL",
            confidence=float(max(probabilities)),
            prob_good=float(probabilities[1]),
            prob_bad=float(probabilities[0]),
            current_price=current_price,
            date=datetime.now().strftime('%Y-%m-%d'),
            model_type="Random Forest (Multi-Ticker)" if "multi" in MODEL_PATH else "Random Forest",
            technical_context=technical_context,
            options_data=options_data,
            all_options=display_options if display_options else None  # All strikes for best expiration
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
