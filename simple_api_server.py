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

        # Get options data (will be None if market closed)
        options_data = get_best_csp_option(ticker)

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
            options_data=options_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
