"""
FastAPI backend server for CSP timing predictions
Runs locally on your PC with GPU, exposed via ngrok
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import joblib
import torch
from datetime import datetime
import os
from deep_predictor import DeepLearningPredictor


app = FastAPI(
    title="CSP Timing API",
    description="Predict optimal timing for selling Cash Secured Puts",
    version="1.0.0"
)

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    ticker: str
    model_type: Optional[str] = "hybrid"


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


class MultiTickerRequest(BaseModel):
    tickers: List[str]
    model_type: Optional[str] = "hybrid"


# Global model cache
MODELS = {}


def load_model(model_type='hybrid'):
    """Load model into cache"""
    if model_type in MODELS:
        return MODELS[model_type]

    model_files = {
        'hybrid': 'csp_hybrid_model.pkl',
        'lstm': 'csp_lstm_model.pkl',
        'transformer': 'csp_transformer_model.pkl',
        'tabnet': 'csp_tabnet_nvda.pkl',
        'tabnet_multi': 'csp_tabnet_multi.pkl'
    }

    model_path = model_files.get(model_type)
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")

    try:
        predictor = DeepLearningPredictor(model_path)
        MODELS[model_type] = predictor
        return predictor
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/")
def read_root():
    """Health check"""
    return {
        "status": "running",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": list(MODELS.keys()),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models")
def list_models():
    """List available models"""
    available = []
    model_files = {
        'hybrid': 'csp_hybrid_model.pkl',
        'lstm': 'csp_lstm_model.pkl',
        'transformer': 'csp_transformer_model.pkl',
        'tabnet': 'csp_tabnet_nvda.pkl',
        'tabnet_multi': 'csp_tabnet_multi.pkl'
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            available.append({
                "name": name,
                "path": path,
                "loaded": name in MODELS
            })

    return {"models": available}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make CSP timing prediction for a single ticker"""
    try:
        ticker = request.ticker.upper()
        model_type = request.model_type

        # Load model
        predictor = load_model(model_type)

        # Get features
        features_df, full_data = predictor.get_current_features(ticker)
        features_scaled = predictor.scaler.transform(features_df.values)

        # Make prediction
        if predictor.model_type == 'TabNet':
            X = features_scaled[-1:, :]
            prediction = predictor.model.predict(X)[0]
            probability = predictor.model.predict_proba(X)[0]
        else:
            if len(features_scaled) < predictor.seq_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"Need at least {predictor.seq_length} days of data"
                )

            X_seq = features_scaled[-predictor.seq_length:]
            X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(predictor.device)

            with torch.no_grad():
                output = predictor.model(X_tensor)
                prob_good = output.cpu().numpy()[0][0]

            probability = [1 - prob_good, prob_good]
            prediction = 1 if prob_good > 0.5 else 0

        current_price = float(full_data['Close'].iloc[-1])
        latest_features = features_df.iloc[-1]

        # Build technical context
        technical_context = {
            'support_resistance': {
                'distance_from_support': float(latest_features.get('Distance_From_Support_Pct', 0)),
                'distance_from_resistance': float(latest_features.get('Distance_From_Resistance_Pct', 0))
            },
            'moving_averages': {
                'price_to_sma20': float(latest_features.get('Price_to_SMA20', 0)),
                'price_to_sma50': float(latest_features.get('Price_to_SMA50', 0)),
                'price_to_sma200': float(latest_features.get('Price_to_SMA200', 0))
            },
            'momentum': {
                'rsi': float(latest_features.get('RSI', 0)),
                'macd': float(latest_features.get('MACD', 0))
            },
            'volatility': {
                'atr_pct': float(latest_features.get('ATR_Pct', 0)),
                'vix': float(latest_features.get('VIX', 0))
            }
        }

        return PredictionResponse(
            ticker=ticker,
            prediction='GOOD' if prediction == 1 else 'WAIT',
            confidence=float(probability[prediction]),
            prob_good=float(probability[1]),
            prob_bad=float(probability[0]),
            current_price=current_price,
            date=full_data.index[-1].strftime('%Y-%m-%d'),
            model_type=predictor.model_type,
            technical_context=technical_context
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/multi")
def predict_multi(request: MultiTickerRequest):
    """Make predictions for multiple tickers"""
    results = []

    for ticker in request.tickers:
        try:
            pred_request = PredictionRequest(ticker=ticker, model_type=request.model_type)
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


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gpu": {
            "available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
        },
        "models_cached": len(MODELS),
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
    print("CSP TIMING API SERVER")
    print("="*70)
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("="*70)

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
