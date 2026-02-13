"""
Simple FastAPI server for CSP timing predictions
Supports both Schwab API (preferred) and Yahoo Finance (fallback) for options data
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import uvicorn
import joblib
import json
import os
import threading
import warnings
import numpy as np
from datetime import datetime, timedelta

# Suppress LightGBM v4 feature-name mismatch warning (cosmetic, not a correctness issue)
warnings.filterwarnings('ignore', message='.*feature names.*', category=UserWarning)
from data_collector import CSPDataCollector

# Lazy imports — options_analyzer depends on advanced_options_pricing which may
# not be present.  These are only needed for the Yahoo Finance fallback path.
def get_best_csp_option(*args, **kwargs):
    from options_analyzer import get_best_csp_option as _fn
    return _fn(*args, **kwargs)

def get_all_csp_options(*args, **kwargs):
    from options_analyzer_multi import get_all_csp_options as _fn
    return _fn(*args, **kwargs)

# Try to import Schwab client (optional)
try:
    from schwab_client import (
        get_csp_options_schwab,
        is_market_open,
        get_market_hours,
        get_stock_price,
        get_quotes,
        get_price_history,
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
PROB_CALIBRATOR = None
PROB_CALIBRATION_METHOD = None

# Per-ticker model components
TIMING_MODELS = None
TIMING_SCALERS = None
TIMING_THRESHOLDS = None
TIMING_GROUP_MAPPING = None
PER_TICKER_MODE = False

# Contract model components (highest priority — unified strike-breach prediction)
CONTRACT_MODE = False
CONTRACT_BASE_MODELS = None
CONTRACT_META_LEARNERS = None
CONTRACT_META_SCALERS = None
CONTRACT_CALIBRATORS = None
CONTRACT_SCALERS = None
CONTRACT_GROUP_MAPPING = None
CONTRACT_FEATURE_COLS = None      # 56 market + 7 contract
CONTRACT_MARKET_FEATURE_COLS = None
CONTRACT_CONTRACT_FEATURE_COLS = None
CONTRACT_VIX_BOUNDARY = 18.0
CONTRACT_FORWARD_DAYS = 35
MIN_EDGE_PP = 1.0
MIN_EDGE_PROB = MIN_EDGE_PP / 100.0

# Ensemble V3 components (loaded preferentially over per-ticker model)
ENSEMBLE_MODE = False
ENSEMBLE_BASE_MODELS = None    # {'{group}_{regime}': {'xgb': ..., 'lgbm': ...}}
ENSEMBLE_META_LEARNERS = None  # {'{group}_{regime}': LogisticRegression}
ENSEMBLE_META_SCALERS = None   # {'{group}_{regime}': StandardScaler for meta-features}
ENSEMBLE_CALIBRATORS = None    # {'{group}_{regime}': CalibratedClassifierCV}
ENSEMBLE_SCALERS = None
ENSEMBLE_THRESHOLDS = None
ENSEMBLE_GROUP_MAPPING = None
ENSEMBLE_VIX_BOUNDARY = 18.0
ENSEMBLE_FEATURE_COLS = None

# Try contract model first (highest priority — unified breach prediction)
contract_path = 'csp_contract_model.pkl'
if os.path.exists(contract_path):
    try:
        print(f"Loading contract model from {contract_path}...")
        contract_data = joblib.load(contract_path)
        if contract_data.get('contract_model'):
            CONTRACT_BASE_MODELS = contract_data['base_models']
            CONTRACT_META_LEARNERS = contract_data['meta_learners']
            CONTRACT_META_SCALERS = contract_data.get('meta_scalers', {})
            CONTRACT_CALIBRATORS = contract_data.get('calibrators', {})
            CONTRACT_SCALERS = contract_data['scalers']
            CONTRACT_GROUP_MAPPING = contract_data['group_mapping']
            CONTRACT_VIX_BOUNDARY = contract_data.get('vix_boundary', 18.0)
            CONTRACT_FORWARD_DAYS = int(contract_data.get('forward_days', 35))
            CONTRACT_FEATURE_COLS = contract_data['feature_cols']
            CONTRACT_MARKET_FEATURE_COLS = contract_data.get('market_feature_cols', [])
            CONTRACT_CONTRACT_FEATURE_COLS = contract_data.get('contract_feature_cols', [])
            FEATURE_NAMES = CONTRACT_MARKET_FEATURE_COLS  # For prepare_inference_features
            CONTRACT_MODE = True
            MODEL_PATH = contract_path
            n_cal = len(CONTRACT_CALIBRATORS)
            print(f"  Contract model loaded: {sorted(CONTRACT_BASE_MODELS.keys())}")
            print(f"  Features: {len(CONTRACT_FEATURE_COLS)} ({len(CONTRACT_MARKET_FEATURE_COLS)} market + {len(CONTRACT_CONTRACT_FEATURE_COLS)} contract)")
            if n_cal:
                print(f"  Calibrators: {n_cal}, Meta-scalers: {len(CONTRACT_META_SCALERS)}")
    except Exception as e:
        print(f"  Failed to load contract model: {e}")
        CONTRACT_MODE = False

# Try ensemble model next (second priority)
ensemble_path = 'csp_timing_ensemble.pkl'
if not CONTRACT_MODE and os.path.exists(ensemble_path):
    try:
        print(f"Loading ensemble model from {ensemble_path}...")
        ens_data = joblib.load(ensemble_path)
        if ens_data.get('ensemble'):
            ENSEMBLE_BASE_MODELS = ens_data['base_models']
            ENSEMBLE_META_LEARNERS = ens_data['meta_learners']
            ENSEMBLE_META_SCALERS = ens_data.get('meta_scalers', {})
            ENSEMBLE_CALIBRATORS = ens_data.get('calibrators', {})
            ENSEMBLE_SCALERS = ens_data['scalers']
            ENSEMBLE_THRESHOLDS = ens_data['thresholds']
            ENSEMBLE_GROUP_MAPPING = ens_data['group_mapping']
            ENSEMBLE_VIX_BOUNDARY = ens_data.get('vix_boundary', 18.0)
            ENSEMBLE_FEATURE_COLS = ens_data['feature_cols']
            FEATURE_NAMES = ENSEMBLE_FEATURE_COLS
            ENSEMBLE_MODE = True
            MODEL_PATH = ensemble_path
            n_cal = len(ENSEMBLE_CALIBRATORS)
            print(f"✓ Ensemble V3 loaded: {sorted(ENSEMBLE_BASE_MODELS.keys())}")
            if n_cal:
                print(f"  Calibrators: {n_cal}, Meta-scalers: {len(ENSEMBLE_META_SCALERS)}")
    except Exception as e:
        print(f"⚠ Failed to load ensemble: {e}")
        ENSEMBLE_MODE = False

# Try v3 regime model next
if not CONTRACT_MODE and not ENSEMBLE_MODE:
    v3_path = 'csp_timing_model_v3.pkl'
    if os.path.exists(v3_path):
        try:
            print(f"Loading V3 regime model from {v3_path}...")
            v3_data = joblib.load(v3_path)
            if v3_data.get('regime_models'):
                TIMING_MODELS = v3_data['models']
                TIMING_SCALERS = v3_data['scalers']
                TIMING_THRESHOLDS = v3_data['thresholds']
                TIMING_GROUP_MAPPING = v3_data['group_mapping']
                FEATURE_NAMES = v3_data['feature_cols']
                ENSEMBLE_VIX_BOUNDARY = v3_data.get('vix_boundary', 18.0)
                PER_TICKER_MODE = True
                MODEL_PATH = v3_path
                print(f"✓ V3 regime models loaded: {sorted(TIMING_MODELS.keys())}")
        except Exception as e:
            print(f"⚠ Failed to load V3 model: {e}")

# Try per-ticker model next (preferred over global) — skip if higher-priority model loaded
per_ticker_path = 'csp_timing_model_per_ticker.pkl'
if not CONTRACT_MODE and not ENSEMBLE_MODE and os.path.exists(per_ticker_path):
    try:
        print(f"Loading per-ticker timing model from {per_ticker_path}...")
        timing_data = joblib.load(per_ticker_path)
        TIMING_MODELS = timing_data['models']  # Dict of group models
        TIMING_SCALERS = timing_data['scalers']  # Dict of group scalers
        TIMING_THRESHOLDS = timing_data['thresholds']
        TIMING_GROUP_MAPPING = timing_data['group_mapping']
        FEATURE_NAMES = timing_data['feature_cols']
        PER_TICKER_MODE = True
        MODEL_PATH = per_ticker_path
        print(f"✓ Per-ticker timing models loaded:")
        for group, model in TIMING_MODELS.items():
            threshold = TIMING_THRESHOLDS[group]
            print(f"  {group:15s}: {type(model).__name__}, threshold={threshold:+.1f}%")
    except Exception as e:
        print(f"⚠ Failed to load per-ticker model: {e}")
        PER_TICKER_MODE = False

# Fallback to global model
if not CONTRACT_MODE and not ENSEMBLE_MODE and not PER_TICKER_MODE:
    for path in ['csp_model_multi.pkl', 'csp_model.pkl']:
        if os.path.exists(path):
            MODEL_PATH = path
            print(f"Loading global model from {MODEL_PATH}...")
            model_data = joblib.load(MODEL_PATH)
            MODEL = model_data['model']
            SCALER = model_data['scaler']
            FEATURE_NAMES = model_data.get('feature_names') or model_data.get('feature_cols')
            print(f"✓ Global model loaded: {MODEL}")
            break

if MODEL is None and not PER_TICKER_MODE:
    print("ERROR: No trained model found!")

# Optional probability calibrator trained from settled prediction logs.
calibrator_path = 'timing_probability_calibrator.pkl'
if os.path.exists(calibrator_path):
    try:
        cal_data = joblib.load(calibrator_path)
        PROB_CALIBRATOR = cal_data.get('calibrator')
        PROB_CALIBRATION_METHOD = cal_data.get('method')
        if PROB_CALIBRATOR is not None:
            print(f"✓ Probability calibrator loaded ({PROB_CALIBRATION_METHOD})")
    except Exception as e:
        print(f"⚠ Failed to load probability calibrator: {e}")
        PROB_CALIBRATOR = None
        PROB_CALIBRATION_METHOD = None

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

# Load strike probability model for accurate edge calculation
STRIKE_PROB_MODEL = None
strike_prob_path_v3 = 'strike_model_v3.pkl'
strike_prob_path_v2 = 'strike_model_v2.pkl'
strike_prob_path = 'strike_probability_model.pkl'

# Try V3 model first (two-stage residual, best accuracy)
if os.path.exists(strike_prob_path_v3):
    try:
        from strike_probability import StrikeProbabilityCalculatorV3
        STRIKE_PROB_MODEL = StrikeProbabilityCalculatorV3(strike_prob_path_v3)
        STRIKE_PROB_VERSION = 3
        print("✓ Strike probability model V3 loaded (two-stage residual, ROC-AUC 0.76)")
    except Exception as e:
        print(f"⚠ Failed to load V3 model: {e}")
        STRIKE_PROB_MODEL = None
        STRIKE_PROB_VERSION = 0

# Fall back to V2 (ticker-specific, trained on option outcomes)
if STRIKE_PROB_MODEL is None and os.path.exists(strike_prob_path_v2):
    try:
        from strike_probability import StrikeProbabilityCalculatorV2
        STRIKE_PROB_MODEL = StrikeProbabilityCalculatorV2(strike_prob_path_v2)
        STRIKE_PROB_VERSION = 2
        print("✓ Strike probability model V2 loaded (ticker-specific)")
    except Exception as e:
        print(f"⚠ Failed to load V2 model: {e}")
        STRIKE_PROB_MODEL = None
        STRIKE_PROB_VERSION = 0

# Fall back to V1
if STRIKE_PROB_MODEL is None and os.path.exists(strike_prob_path):
    try:
        from strike_probability import StrikeProbabilityCalculator
        STRIKE_PROB_MODEL = StrikeProbabilityCalculator(strike_prob_path)
        STRIKE_PROB_VERSION = 1
        print(f"✓ Strike probability model V1 loaded from {strike_prob_path}")
    except Exception as e:
        print(f"⚠ Failed to load strike probability model: {e}")
        STRIKE_PROB_MODEL = None
        STRIKE_PROB_VERSION = 0

if STRIKE_PROB_MODEL is None:
    STRIKE_PROB_VERSION = 0
    print(f"⚠ No strike probability model available")


def apply_probability_calibration(raw_p_safe: float) -> float:
    """
    Apply optional post-hoc probability calibration to p_safe.
    Falls back to raw probability when no calibrator is available.
    """
    if PROB_CALIBRATOR is None:
        return float(np.clip(raw_p_safe, 0.001, 0.999))

    try:
        if PROB_CALIBRATION_METHOD == 'isotonic':
            calibrated = float(PROB_CALIBRATOR.predict([raw_p_safe])[0])
        elif PROB_CALIBRATION_METHOD == 'platt':
            calibrated = float(PROB_CALIBRATOR.predict_proba([[raw_p_safe]])[0, 1])
        else:
            calibrated = raw_p_safe
        return float(np.clip(calibrated, 0.001, 0.999))
    except Exception as e:
        print(f"[Calibration] Failed to apply calibrator: {e}")
        return float(np.clip(raw_p_safe, 0.001, 0.999))


def prepare_inference_features(df, feature_names):
    """
    Align live data to training feature contract.
    Returns aligned feature DataFrame and parity diagnostics.
    """
    if feature_names is None:
        raise ValueError("Feature names not loaded")

    aligned = df.copy()
    missing = [c for c in feature_names if c not in aligned.columns]
    for col in missing:
        aligned[col] = 0.0

    extra = [c for c in aligned.columns if c not in feature_names]
    features_df = aligned[feature_names].copy()
    features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    parity = {
        "missing_features": missing,
        "missing_count": len(missing),
        "extra_count": len(extra),
    }
    return features_df, parity


@app.get("/")
def read_root():
    """Health check"""
    if PER_TICKER_MODE:
        model_status = f"per-ticker ({', '.join(TIMING_MODELS.keys())})" if TIMING_MODELS else "No model loaded"
    else:
        model_status = str(MODEL) if MODEL else "No model loaded"
    return {
        "status": "running",
        "model": model_status,
        "model_file": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    if CONTRACT_MODE:
        return {
            "status": "healthy",
            "model_type": "contract_v1",
            "model_loaded": True,
            "model_file": MODEL_PATH,
            "groups": sorted(CONTRACT_BASE_MODELS.keys()) if CONTRACT_BASE_MODELS else [],
            "feature_count": len(CONTRACT_FEATURE_COLS) if CONTRACT_FEATURE_COLS else 0,
            "market_features": len(CONTRACT_MARKET_FEATURE_COLS) if CONTRACT_MARKET_FEATURE_COLS else 0,
            "contract_features": len(CONTRACT_CONTRACT_FEATURE_COLS) if CONTRACT_CONTRACT_FEATURE_COLS else 0,
            "calibrators": len(CONTRACT_CALIBRATORS) if CONTRACT_CALIBRATORS else 0,
            "timestamp": datetime.now().isoformat()
        }
    elif ENSEMBLE_MODE:
        return {
            "status": "healthy",
            "model_type": "ensemble_v3",
            "model_loaded": True,
            "model_file": MODEL_PATH,
            "groups": sorted(ENSEMBLE_BASE_MODELS.keys()) if ENSEMBLE_BASE_MODELS else [],
            "feature_count": len(ENSEMBLE_FEATURE_COLS) if ENSEMBLE_FEATURE_COLS else 0,
            "calibrators": len(ENSEMBLE_CALIBRATORS) if ENSEMBLE_CALIBRATORS else 0,
            "timestamp": datetime.now().isoformat()
        }
    elif PER_TICKER_MODE:
        return {
            "status": "healthy",
            "model_type": "per_ticker",
            "model_loaded": True,
            "model_file": MODEL_PATH,
            "groups": list(TIMING_MODELS.keys()) if TIMING_MODELS else [],
            "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "healthy" if MODEL else "error",
            "model_type": "global",
            "model_loaded": MODEL is not None,
            "model_file": MODEL_PATH,
            "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
            "timestamp": datetime.now().isoformat()
        }


@app.post("/reload")
def reload_model():
    """Reload the model from disk without restarting the server"""
    global MODEL, SCALER, FEATURE_NAMES, MODEL_PATH
    global TIMING_MODELS, TIMING_SCALERS, TIMING_THRESHOLDS, TIMING_GROUP_MAPPING, PER_TICKER_MODE
    global ENSEMBLE_MODE, ENSEMBLE_BASE_MODELS, ENSEMBLE_META_LEARNERS
    global ENSEMBLE_META_SCALERS, ENSEMBLE_CALIBRATORS
    global ENSEMBLE_SCALERS, ENSEMBLE_THRESHOLDS, ENSEMBLE_GROUP_MAPPING, ENSEMBLE_FEATURE_COLS
    global CONTRACT_MODE, CONTRACT_BASE_MODELS, CONTRACT_META_LEARNERS
    global CONTRACT_META_SCALERS, CONTRACT_CALIBRATORS, CONTRACT_SCALERS
    global CONTRACT_GROUP_MAPPING, CONTRACT_FEATURE_COLS, CONTRACT_MARKET_FEATURE_COLS
    global CONTRACT_CONTRACT_FEATURE_COLS, CONTRACT_VIX_BOUNDARY, CONTRACT_FORWARD_DAYS

    # Try contract model first (highest priority)
    contract_path = 'csp_contract_model.pkl'
    if os.path.exists(contract_path):
        try:
            cdata = joblib.load(contract_path)
            if cdata.get('contract_model'):
                CONTRACT_BASE_MODELS = cdata['base_models']
                CONTRACT_META_LEARNERS = cdata['meta_learners']
                CONTRACT_META_SCALERS = cdata.get('meta_scalers', {})
                CONTRACT_CALIBRATORS = cdata.get('calibrators', {})
                CONTRACT_SCALERS = cdata['scalers']
                CONTRACT_GROUP_MAPPING = cdata['group_mapping']
                CONTRACT_VIX_BOUNDARY = cdata.get('vix_boundary', 18.0)
                CONTRACT_FORWARD_DAYS = int(cdata.get('forward_days', 35))
                CONTRACT_FEATURE_COLS = cdata['feature_cols']
                CONTRACT_MARKET_FEATURE_COLS = cdata.get('market_feature_cols', [])
                CONTRACT_CONTRACT_FEATURE_COLS = cdata.get('contract_feature_cols', [])
                FEATURE_NAMES = CONTRACT_MARKET_FEATURE_COLS
                CONTRACT_MODE = True
                ENSEMBLE_MODE = False
                PER_TICKER_MODE = True  # suppress global fallback
                MODEL_PATH = contract_path
                return {
                    "status": "success",
                    "message": f"Contract model reloaded from {contract_path}",
                    "model_type": "Contract V1 (XGB+LGBM+Meta, breach prediction)",
                    "keys": sorted(CONTRACT_BASE_MODELS.keys()),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"  Failed to reload contract model: {e}")

    # Try ensemble next (second priority)
    ensemble_path = 'csp_timing_ensemble.pkl'
    if os.path.exists(ensemble_path):
        try:
            ens_data = joblib.load(ensemble_path)
            if ens_data.get('ensemble'):
                ENSEMBLE_BASE_MODELS = ens_data['base_models']
                ENSEMBLE_META_LEARNERS = ens_data['meta_learners']
                ENSEMBLE_META_SCALERS = ens_data.get('meta_scalers', {})
                ENSEMBLE_CALIBRATORS = ens_data.get('calibrators', {})
                ENSEMBLE_SCALERS = ens_data['scalers']
                ENSEMBLE_THRESHOLDS = ens_data['thresholds']
                ENSEMBLE_GROUP_MAPPING = ens_data['group_mapping']
                ENSEMBLE_FEATURE_COLS = ens_data['feature_cols']
                FEATURE_NAMES = ENSEMBLE_FEATURE_COLS
                ENSEMBLE_MODE = True
                MODEL_PATH = ensemble_path
                return {
                    "status": "success",
                    "message": f"Ensemble V3 reloaded from {ensemble_path}",
                    "model_type": "Ensemble V3 (XGB+LGBM+Meta)",
                    "keys": sorted(ENSEMBLE_BASE_MODELS.keys()),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"⚠ Failed to reload ensemble: {e}")

    # Try per-ticker model next
    per_ticker_path = 'csp_timing_model_per_ticker.pkl'
    if os.path.exists(per_ticker_path):
        try:
            print(f"Reloading per-ticker model from {per_ticker_path}...")
            timing_data = joblib.load(per_ticker_path)
            TIMING_MODELS = timing_data['models']
            TIMING_SCALERS = timing_data['scalers']
            TIMING_THRESHOLDS = timing_data['thresholds']
            TIMING_GROUP_MAPPING = timing_data['group_mapping']
            FEATURE_NAMES = timing_data['feature_cols']
            PER_TICKER_MODE = True
            MODEL_PATH = per_ticker_path
            print(f"✓ Per-ticker models reloaded: {list(TIMING_MODELS.keys())}")
            return {
                "status": "success",
                "message": f"Per-ticker models reloaded from {per_ticker_path}",
                "model_type": "Per-Ticker XGBoost",
                "groups": list(TIMING_MODELS.keys()),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"⚠ Failed to reload per-ticker model: {e}")

    # Fallback to global model
    for path in ['csp_model_multi.pkl', 'csp_model.pkl']:
        if os.path.exists(path):
            MODEL_PATH = path
            print(f"Reloading model from {MODEL_PATH}...")
            model_data = joblib.load(MODEL_PATH)
            MODEL = model_data['model']
            SCALER = model_data['scaler']
            FEATURE_NAMES = model_data.get('feature_names') or model_data.get('feature_cols')
            PER_TICKER_MODE = False
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


@app.get("/stream/{ticker}")
async def stream_ticker(
    ticker: str,
    interval: int = 10,
    include_options: bool = False,
    min_delta: float = 0.10,
    max_delta: float = 0.60,
    options_interval: int = 30,
):
    """Stream live quote + optional options data via Server-Sent Events.

    Sends two event types:
      {"type": "price",   "symbol": ..., "price": ..., ...}
      {"type": "options", "symbol": ..., "options": [...]}  (if include_options=true)

    Price is polled every `interval` seconds (default 10, min 5).
    Options are polled every `options_interval` seconds (default 30).
    """
    ticker = ticker.upper()
    price_interval  = max(interval, 5)
    opt_ticks_every = max(1, options_interval // price_interval)

    async def event_generator():
        last_price = None
        tick = 0
        while True:
            try:
                if not SCHWAB_AVAILABLE:
                    yield f"data: {json.dumps({'error': 'Schwab API not available'})}\n\n"
                    return

                # --- Price tick (every interval) ---
                quote = get_stock_price(ticker)
                price_payload = {
                    "type": "price",
                    "symbol": ticker,
                    "price": quote["price"],
                    "change": quote["change"],
                    "change_percent": quote["change_percent"],
                    "bid": quote["bid"],
                    "ask": quote["ask"],
                    "high": quote["high"],
                    "low": quote["low"],
                    "volume": quote["volume"],
                    "timestamp": datetime.now().isoformat(),
                }
                if quote["price"] != last_price:
                    last_price = quote["price"]
                    yield f"data: {json.dumps(price_payload)}\n\n"
                else:
                    yield f": heartbeat\n\n"

                # --- Options tick (every options_interval) ---
                tick += 1
                if include_options and tick % opt_ticks_every == 0:
                    try:
                        loop = asyncio.get_event_loop()
                        opts = await loop.run_in_executor(
                            None,
                            lambda: get_csp_options_schwab(ticker, min_delta=min_delta, max_delta=max_delta)
                        )
                        if opts:
                            yield f"data: {json.dumps({'type': 'options', 'symbol': ticker, 'options': opts, 'timestamp': datetime.now().isoformat()})}\n\n"
                    except Exception as opt_err:
                        print(f"[Stream] Options fetch error for {ticker}: {opt_err}")

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(price_interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chart/{ticker}")
async def get_chart_data(ticker: str, range: int = 22):
    """Return OHLC candles for charting.

    range: 5=1W (30-min bars via Schwab), 22=1M, 66=3M, 132=6M (daily bars).
    Falls back to collector daily data when Schwab is unavailable.
    """
    ticker = ticker.upper()
    try:
        if SCHWAB_AVAILABLE:
            if range <= 5:
                raw = get_price_history(
                    ticker, period_type="day", period=5,
                    frequency_type="minute", frequency=30,
                )
                fmt = "%Y-%m-%d %H:%M"
            elif range <= 22:
                # 30-min bars (10 trading days max for minute data); client
                # will aggregate 12 bars → one 6-hr candle
                raw = get_price_history(
                    ticker, period_type="day", period=10,
                    frequency_type="minute", frequency=30,
                )
                fmt = "%Y-%m-%d %H:%M"
            elif range <= 66:
                raw = get_price_history(
                    ticker, period_type="month", period=3,
                    frequency_type="daily", frequency=1,
                )
                fmt = "%Y-%m-%d"
            else:
                raw = get_price_history(
                    ticker, period_type="month", period=6,
                    frequency_type="daily", frequency=1,
                )
                fmt = "%Y-%m-%d"

            candles = []
            for c in raw.get("candles", []):
                ts = c.get("datetime", 0)
                dt = datetime.fromtimestamp(ts / 1000)
                candles.append({
                    "date": dt.strftime(fmt),
                    "open": round(float(c.get("open", 0)), 2),
                    "high": round(float(c.get("high", 0)), 2),
                    "low": round(float(c.get("low", 0)), 2),
                    "close": round(float(c.get("close", 0)), 2),
                    "volume": int(c.get("volume", 0)),
                })
            return {
                "ticker": ticker,
                "candles": candles,
                "frequency": "intraday" if range <= 22 else "daily",
                "range": range,
            }

        # Schwab unavailable — serve collector daily data
        collector = CSPDataCollector(ticker)
        collector.fetch_data(period="6mo")
        hist_df = collector.data.tail(range)
        candles = []
        for idx, row in hist_df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            candles.append({
                "date": date_str,
                "open": round(float(row.get("Open", row["Close"])), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
            })
        return {
            "ticker": ticker,
            "candles": candles,
            "frequency": "daily",
            "range": range,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
    if MODEL is None and not PER_TICKER_MODE and not ENSEMBLE_MODE and not CONTRACT_MODE:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        ticker = request.ticker.upper()

        # Collect data and features — 2y is sufficient for inference (longest
        # rolling window is 252 trading days); 10y is only needed for training.
        # Note: Schwab only accepts periodType=year with values [1,2,3,5,10].
        collector = CSPDataCollector(ticker, period='2y')
        collector.fetch_data()
        collector.calculate_technical_indicators()
        feature_forward_days = CONTRACT_FORWARD_DAYS if CONTRACT_MODE else 35
        collector.create_target_variable(forward_days=feature_forward_days, strike_otm_pct=0.05)

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

        features_df, feature_parity = prepare_inference_features(collector.data, FEATURE_NAMES)
        if feature_parity["missing_count"] > 0:
            print(f"[FeatureParity] {ticker}: {feature_parity['missing_count']} missing features filled with 0")

        if len(features_df) == 0:
            raise HTTPException(status_code=400, detail=f"No data available for {ticker}")

        # Get most recent features
        X_current = features_df.iloc[-1:].values

        # Determine VIX regime from current data
        current_vix = float(features_df.iloc[-1].get('VIX', 15.0))
        vix_boundary = CONTRACT_VIX_BOUNDARY if CONTRACT_MODE else ENSEMBLE_VIX_BOUNDARY
        vix_regime = 'low_vix' if current_vix < vix_boundary else 'high_vix'

        # ---- Contract model dispatch (highest priority) ----
        if CONTRACT_MODE:
            from scipy.stats import norm as _norm
            group = CONTRACT_GROUP_MAPPING.get(ticker, 'tech_growth')
            model_key = f'{group}_{vix_regime}'
            if model_key not in CONTRACT_SCALERS:
                alt_key = f'{group}_low_vix' if vix_regime == 'high_vix' else f'{group}_high_vix'
                model_key = alt_key if alt_key in CONTRACT_SCALERS else group

            scaler = CONTRACT_SCALERS[model_key]
            base = CONTRACT_BASE_MODELS[model_key]
            meta = CONTRACT_META_LEARNERS[model_key]

            # Get market state values for contract feature construction
            fc_market = CONTRACT_MARKET_FEATURE_COLS
            def _mfeat(name, default=0.0):
                return float(X_current[0, fc_market.index(name)] if name in fc_market else default)

            rv_val = max(_mfeat('Volatility_20D', 20.0) / 100.0, 0.01)
            ivrr = max(min(_mfeat('IV_RV_Ratio', 1.2), 2.5), 1.0)
            vol_est = rv_val * ivrr
            T_val = float(CONTRACT_FORWARD_DAYS) / 365.0
            vix_val = _mfeat('VIX', 15.0)
            iv_rank_val = _mfeat('IV_Rank', 50.0)
            rsi_val = _mfeat('RSI', 50.0)
            atr_pct_val = _mfeat('ATR_Pct', 1.0)
            regime_val = _mfeat('Regime_Trend', 1.0)

            # Evaluate P(breach) for each contract; used later for EV-based selection
            contract_breach_probs = {}  # {delta: p_breach}
            delta_buckets = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

            for target_delta in delta_buckets:
                # Delta-to-OTM conversion
                try:
                    strike_otm = vol_est * np.sqrt(T_val) * _norm.ppf(1 - target_delta)
                    strike_otm = max(0.01, min(0.30, strike_otm))
                except Exception:
                    strike_otm = target_delta * 0.30

                # Build 7 contract features
                contract_feats = {
                    'target_delta': target_delta,
                    'delta_x_vol': target_delta * vol_est,
                    'delta_x_vix': target_delta * vix_val / 100.0,
                    'delta_x_iv_rank': target_delta * iv_rank_val / 100.0,
                    'delta_squared': target_delta ** 2,
                    'delta_x_rsi': target_delta * rsi_val / 100.0,
                    'strike_otm_x_atr': strike_otm * atr_pct_val,
                }

                # Concatenate market features + contract features → 63-dim vector
                full_vec = np.zeros((1, len(CONTRACT_FEATURE_COLS)))
                for j, col in enumerate(CONTRACT_FEATURE_COLS):
                    if col in contract_feats:
                        full_vec[0, j] = contract_feats[col]
                    elif col in fc_market:
                        full_vec[0, j] = X_current[0, fc_market.index(col)]
                    # else: 0.0

                # Route through model
                X_sc = scaler.transform(full_vec)
                xgb_p = base['xgb'].predict_proba(X_sc)[0, 1]
                lgbm_p = base['lgbm'].predict_proba(X_sc)[0, 1] if base.get('lgbm') else xgb_p

                fc_full = CONTRACT_FEATURE_COLS
                def _cfeat(name, default=0.0):
                    return float(full_vec[0, fc_full.index(name)] if name in fc_full else default)

                meta_X = np.array([[
                    xgb_p, lgbm_p,
                    _cfeat('VIX', 15.0),
                    _cfeat('VIX_Rank', 50.0),
                    _cfeat('Regime_Trend', 1.0),
                ]])

                if CONTRACT_META_SCALERS and model_key in CONTRACT_META_SCALERS:
                    meta_X = CONTRACT_META_SCALERS[model_key].transform(meta_X)

                if CONTRACT_CALIBRATORS and model_key in CONTRACT_CALIBRATORS:
                    p_breach = float(CONTRACT_CALIBRATORS[model_key].predict_proba(meta_X)[0, 1])
                else:
                    p_breach = float(meta.predict_proba(meta_X)[0, 1])

                contract_breach_probs[target_delta] = p_breach

            # Derive p_safe from delta~0.30 for backward-compatible API response
            p_breach_ref = contract_breach_probs.get(0.30, 0.20)
            p_safe = 1.0 - p_breach_ref
            p_downside = p_breach_ref
            prediction = int(p_safe >= 0.5)
            model_type = f"Contract V1 ({group}/{vix_regime})"

            # Store breach probs on collector for later use in options EV
            collector._contract_breach_probs = contract_breach_probs

            print(f"[Contract] {ticker} -> {group}/{vix_regime}  "
                  f"P(breach|d=0.30)={p_breach_ref:.3f}  "
                  f"P(breach) range: {min(contract_breach_probs.values()):.3f}-{max(contract_breach_probs.values()):.3f}")

        # ---- Ensemble V3 dispatch ----
        elif ENSEMBLE_MODE:
            group = ENSEMBLE_GROUP_MAPPING.get(ticker, 'tech_growth')
            model_key = f'{group}_{vix_regime}'
            if model_key not in ENSEMBLE_SCALERS:
                alt_key = f'{group}_low_vix' if vix_regime == 'high_vix' else f'{group}_high_vix'
                model_key = alt_key if alt_key in ENSEMBLE_SCALERS else group

            scaler = ENSEMBLE_SCALERS[model_key]
            base = ENSEMBLE_BASE_MODELS[model_key]
            meta = ENSEMBLE_META_LEARNERS[model_key]
            threshold = ENSEMBLE_THRESHOLDS.get(model_key, -5.0)

            X_scaled = scaler.transform(X_current)
            xgb_prob = base['xgb'].predict_proba(X_scaled)[0, 1]
            lgbm_prob = base['lgbm'].predict_proba(X_scaled)[0, 1] if base.get('lgbm') else xgb_prob

            fc = ENSEMBLE_FEATURE_COLS
            def _get_feat(name, default=0.0):
                return float(X_current[0, fc.index(name)] if name in fc else default)

            meta_X = np.array([[
                xgb_prob, lgbm_prob,
                _get_feat('VIX', 15.0),
                _get_feat('VIX_Rank', 50.0),
                _get_feat('Regime_Trend', 1.0),
            ]])

            if ENSEMBLE_META_SCALERS and model_key in ENSEMBLE_META_SCALERS:
                meta_X = ENSEMBLE_META_SCALERS[model_key].transform(meta_X)

            if ENSEMBLE_CALIBRATORS and model_key in ENSEMBLE_CALIBRATORS:
                p_safe = float(ENSEMBLE_CALIBRATORS[model_key].predict_proba(meta_X)[0, 1])
            else:
                p_safe = float(meta.predict_proba(meta_X)[0, 1])
            p_downside = 1.0 - p_safe
            prediction = int(p_safe >= 0.5)
            model_type = f"Ensemble V3 ({group}/{vix_regime})"
            print(f"[Ensemble] {ticker} -> {group}/{vix_regime}  "
                  f"xgb={xgb_prob:.3f} lgbm={lgbm_prob:.3f} meta={p_safe:.3f}")

        # ---- Per-ticker V3 regime model ----
        elif PER_TICKER_MODE:
            group = TIMING_GROUP_MAPPING.get(ticker, 'tech_growth')
            # Try regime-specific key first, fall back to plain group
            regime_key = f'{group}_{vix_regime}'
            if TIMING_MODELS and regime_key in TIMING_MODELS:
                model_key = regime_key
            else:
                model_key = group
            model = TIMING_MODELS[model_key]
            scaler = TIMING_SCALERS[model_key]
            threshold = TIMING_THRESHOLDS.get(model_key, TIMING_THRESHOLDS.get(group, -5.0))
            print(f"[Per-Ticker] {ticker} → {model_key} (threshold: {threshold:+.1f}%)")

            X_scaled = scaler.transform(X_current)
            prediction = int(model.predict(X_scaled)[0])
            probabilities = model.predict_proba(X_scaled)[0]
            p_safe = float(probabilities[1])
            p_downside = float(probabilities[0])
            model_type = f"{type(model).__name__} (Per-Ticker: {model_key})"

        else:
            # Use global model (legacy fallback)
            model = MODEL
            scaler = SCALER
            group = 'global'
            print(f"[Global] Using global model for {ticker}")
            X_scaled = scaler.transform(X_current)
            prediction = int(model.predict(X_scaled)[0])
            probabilities = model.predict_proba(X_scaled)[0]
            p_safe = float(probabilities[1])
            p_downside = float(probabilities[0])
            model_type = type(MODEL).__name__

        # Apply optional temporal probability calibration from settled logs.
        p_safe_raw = float(p_safe)
        p_safe = apply_probability_calibration(p_safe_raw)
        p_downside = 1.0 - p_safe
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
                    all_options = get_csp_options_schwab(ticker, min_delta=request.min_delta, max_delta=request.max_delta, max_dte=100)
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

            # Calculate edge: contract model (preferred) or strike model (fallback)
            use_contract_edge = CONTRACT_MODE and hasattr(collector, '_contract_breach_probs')
            use_strike_model = (not use_contract_edge and
                                STRIKE_PROB_MODEL is not None and STRIKE_PROB_MODEL.model_loaded)

            if use_contract_edge:
                print("[Edge] Using contract model P(breach)")
            elif use_strike_model:
                print("[Edge] Using strike-specific probability model")
            else:
                print("[Edge] No edge model available - edge estimation disabled")

            for option in all_options:
                market_delta = abs(option['delta'])
                strike = option['strike']
                premium = float(option.get('premium', 0.0) or 0.0)
                bid = float(option.get('bid', 0.0) or 0.0)
                ask = float(option.get('ask', 0.0) or 0.0)

                otm_pct = ((current_price - strike) / current_price) * 100

                model_prob = None

                if use_contract_edge:
                    # Find closest delta bucket from contract model predictions
                    breach_probs = collector._contract_breach_probs
                    closest_delta = min(breach_probs.keys(), key=lambda d: abs(d - market_delta))
                    model_prob = breach_probs[closest_delta]

                elif use_strike_model:
                    if STRIKE_PROB_VERSION in (2, 3):
                        model_prob = STRIKE_PROB_MODEL.predict_for_delta(collector.data, market_delta, ticker)
                    else:
                        model_prob = STRIKE_PROB_MODEL.predict_strike_probability(collector.data, otm_pct)

                if model_prob is None:
                    option['market_delta'] = round(market_delta, 3)
                    option['model_prob_assignment'] = None
                    option['edge'] = None
                    option['edge_prob'] = None
                    option['edge_pp'] = None
                    option['edge_signal'] = 'NO_EDGE_ESTIMATE' if not use_contract_edge else 'NO_EDGE_MODEL'
                    option['spread_pct'] = None
                    option['ev_per_share'] = None
                    option['ev_pct_of_strike'] = None
                    option['risk_adjusted_ev'] = None
                    option['is_suggested'] = False
                    continue

                # Edge in both unit systems:
                #   edge_prob: probability units (0.01 = 1 percentage point)
                #   edge_pp: percentage points
                edge_prob = market_delta - model_prob
                edge_pp = edge_prob * 100.0

                # EV model
                spread = max(0.0, ask - bid)
                spread_pct = (spread / premium) if premium > 0 else 1.0
                slippage_cost = spread * 0.25
                assignment_loss_pct = 0.05
                expected_assignment_loss = model_prob * strike * assignment_loss_pct
                expected_value = premium - expected_assignment_loss - slippage_cost

                # Contract model: timing is embedded, no timing_multiplier needed
                if use_contract_edge:
                    risk_adjusted_ev = expected_value
                else:
                    timing_multiplier = 0.75 + (0.5 * p_safe)
                    risk_adjusted_ev = expected_value * timing_multiplier

                option['market_delta'] = round(market_delta, 3)
                option['model_prob_assignment'] = round(model_prob, 3)
                option['edge_prob'] = round(edge_prob, 4)
                option['edge_pp'] = round(edge_pp, 2)
                option['edge'] = option['edge_pp']  # legacy alias
                option['edge_signal'] = 'GOOD EDGE' if edge_prob > 0 else 'NEGATIVE EDGE'
                option['spread_pct'] = round(spread_pct, 3)
                option['ev_per_share'] = round(expected_value, 4)
                option['ev_pct_of_strike'] = round((expected_value / strike) * 100, 3) if strike > 0 else None
                option['risk_adjusted_ev'] = round(risk_adjusted_ev, 4)
                option['is_suggested'] = False

            # Select best option via EV optimization with uncertainty no-trade band.
            has_edge_model = use_contract_edge or use_strike_model
            if all_options:
                if not has_edge_model:
                    options_data = None
                    print("No calibrated edge model - EV ranking unavailable, no suggestion")
                else:
                    # Contract mode embeds timing — skip the timing uncertainty gate
                    timing_uncertain = (not use_contract_edge) and abs(csp_score) < 0.15
                    near_earnings = bool(current_data.get('Near_Earnings', False))
                    days_to_earnings = float(current_data.get('Days_To_Earnings', 999))

                    if timing_uncertain:
                        options_data = None
                        print(f"No-trade band: timing uncertainty (csp_score={csp_score:.3f})")
                        print("Skipping suggestions due to low timing decisiveness")
                    elif near_earnings and days_to_earnings <= 7:
                        options_data = None
                        print(f"No-trade band: earnings risk (days_to_earnings={days_to_earnings:.0f})")
                        print("Skipping suggestions due to near-term earnings event risk")
                    else:
                        # Filter for executable contracts and positive risk-adjusted EV.
                        ev_candidates = [
                            opt for opt in all_options
                            if opt.get('risk_adjusted_ev') is not None
                            and opt['risk_adjusted_ev'] > 0
                            and opt.get('spread_pct') is not None
                            and opt['spread_pct'] <= 0.20
                            and opt.get('edge_prob') is not None
                            and abs(opt['edge_prob']) >= MIN_EDGE_PROB  # no-trade band for weak edge
                            and abs(opt.get('delta', 0)) <= 0.4  # cap delta at 0.40
                            and 20 <= opt.get('dte', 0) <= 50  # 20-50 DTE window
                        ]
                        if ev_candidates:
                            best_option = max(ev_candidates, key=lambda x: x.get('risk_adjusted_ev', -1e9))
                            best_option['is_suggested'] = True
                            options_data = best_option
                            print(
                                f"Best EV: ${best_option['strike']} strike, "
                                f"EV/share={best_option['ev_per_share']:.3f}, "
                                f"RiskAdjEV={best_option['risk_adjusted_ev']:.3f}, "
                                f"Edge={best_option['edge']:.1f}"
                            )
                        else:
                            options_data = None
                            print("No options passed positive EV + liquidity filters - no suggestion")
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
            },
            'vix_regime': {
                'current_vix': round(current_vix, 2),
                'regime': vix_regime,
                'boundary': ENSEMBLE_VIX_BOUNDARY,
            },
            'feature_parity': {
                'missing_count': feature_parity['missing_count'],
                'missing_features_sample': feature_parity['missing_features'][:10],
                'extra_count': feature_parity['extra_count'],
            },
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
            log_kwargs = dict(
                ticker=ticker,
                price=current_price,
                p_safe=p_safe,
                p_downside=p_downside,
                csp_score=csp_score,
                suggested_strike=options_data['strike'] if options_data else None,
                suggested_expiration=options_data['expiration'] if options_data else None,
            )
            if CONTRACT_MODE:
                log_kwargs['model_version'] = 'contract_v1'
                log_kwargs['label_forward_days'] = int(CONTRACT_FORWARD_DAYS)
                if options_data:
                    log_kwargs['model_p_breach'] = options_data.get('model_prob_assignment')
                    log_kwargs['suggested_delta'] = abs(options_data.get('delta', 0))
                    log_kwargs['suggested_otm_pct'] = ((current_price - options_data.get('strike', current_price)) / current_price) if current_price > 0 else None
            _log_prediction(**log_kwargs)
        except Exception as log_err:
            print(f"Failed to log prediction: {log_err}")

        # model_type is set above in the dispatch block

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
            model_type=model_type,
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

# ---------------------------------------------------------------------------
# Lightweight prediction logger (replaces old ModelValidator dependency)
# ---------------------------------------------------------------------------
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_log.json")
_log_lock = threading.Lock()


def _load_predictions_log():
    if os.path.exists(_LOG_PATH):
        try:
            with open(_LOG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"predictions": []}


def _save_predictions_log(data):
    with open(_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _log_prediction(ticker, price, p_safe, p_downside, csp_score,
                    suggested_strike=None, suggested_expiration=None,
                    model_p_breach=None, suggested_delta=None,
                    suggested_otm_pct=None, model_version=None,
                    label_forward_days=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "price": float(price),
        "p_safe": float(p_safe),
        "p_downside": float(p_downside),
        "csp_score": float(csp_score),
        "prediction": "SAFE" if p_safe > p_downside else "RISKY",
        "suggested_strike": suggested_strike,
        "suggested_expiration": suggested_expiration,
        "outcome": None,
        "outcome_date": None,
    }
    # Contract model extended fields
    if model_p_breach is not None:
        entry["model_p_breach"] = float(model_p_breach)
    if suggested_delta is not None:
        entry["suggested_delta"] = float(suggested_delta)
    if suggested_otm_pct is not None:
        entry["suggested_otm_pct"] = float(suggested_otm_pct)
    if model_version is not None:
        entry["model_version"] = model_version
    if label_forward_days is not None:
        entry["label_forward_days"] = int(label_forward_days)
    with _log_lock:
        data = _load_predictions_log()
        data["predictions"].append(entry)
        _save_predictions_log(data)


@app.get("/validation/stats")
def get_validation_stats():
    """Get prediction accuracy stats from logged predictions"""
    data = _load_predictions_log()
    preds = data.get("predictions", [])
    settled = [p for p in preds if p.get("outcome") is not None]

    if not settled:
        return {
            "status": "success",
            "accuracy_stats": {"message": "No predictions with outcomes yet",
                               "total_predictions": len(preds),
                               "settled": 0},
            "recent_predictions": preds[-10:],
            "timestamp": datetime.now().isoformat(),
        }

    correct = sum(
        1 for p in settled
        if (p["p_safe"] > 0.5) == (p["outcome"] == 1)
    )
    return {
        "status": "success",
        "accuracy_stats": {
            "total_predictions": len(preds),
            "settled": len(settled),
            "accuracy": round(correct / len(settled), 4),
        },
        "recent_predictions": preds[-10:],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/validation/record_outcome")
def record_outcome(ticker: str, prediction_date: str, outcome: int, actual_return: float = None):
    """Record the actual outcome for a past prediction"""
    with _log_lock:
        data = _load_predictions_log()
        found = False
        for pred in data["predictions"]:
            if pred["ticker"] == ticker and pred["timestamp"].startswith(prediction_date):
                pred["outcome"] = int(outcome)
                pred["outcome_date"] = datetime.now().isoformat()
                if actual_return is not None:
                    pred["actual_return"] = float(actual_return)
                found = True
                break
        if found:
            _save_predictions_log(data)
    return {
        "status": "success" if found else "not_found",
        "message": f"Outcome recorded for {ticker}" if found else f"No prediction found for {ticker} on {prediction_date}",
    }


@app.get("/validation/recent")
def get_recent_predictions(n: int = 20):
    """Get recent logged predictions"""
    data = _load_predictions_log()
    return {
        "status": "success",
        "predictions": data.get("predictions", [])[-n:],
        "timestamp": datetime.now().isoformat(),
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
