# CSP Timing Model V2: Stock-Specific Mean Reversion

**Date:** 2026-02-09
**Status:** ✅ Implementation Complete, Training In Progress
**Version:** Per-Ticker V1 with Mean Reversion Features

---

## Overview

Implemented a new CSP timing model that addresses the key limitation of the previous global model: **stock-specific bounce patterns**. The new model learns that NVDA typically bounces after hard drops, while AAPL has different characteristics.

### Key Improvements

1. **Per-Ticker Group Models** (4 groups instead of 1 global model)
2. **12 New Mean Reversion Features** (54 total features, up from 42)
3. **Adaptive Thresholds** (each group gets optimal threshold based on volatility)
4. **Enhanced Mean Reversion Capture** (explicitly models "drop → bounce" patterns)

---

## What Changed

### 1. Mean Reversion Features (12 New Features)

Added to `data_collector.py` in `calculate_technical_indicators()`:

| Feature | Description | Purpose |
|---------|-------------|---------|
| `Recent_Drop_1D` | Magnitude of 1-day drop (0 if up) | Capture immediate drop size |
| `Recent_Drop_3D` | Magnitude of 3-day drop | Short-term momentum reversal |
| `Recent_Drop_5D` | Magnitude of 5-day drop | Medium-term oversold signal |
| `RSI_Oversold` | Binary: RSI < 30 | Classic oversold indicator |
| `RSI_Extreme` | Binary: RSI < 25 | Extreme oversold (capitulation) |
| `Pullback_From_5D_High` | Distance from 5-day high (%) | Recent weakness |
| `Pullback_From_20D_High` | Distance from 20-day high (%) | Pullback depth |
| `Return_Acceleration` | Change in daily return | Drop decelerating = bottom forming |
| `Volume_Spike_Down` | High volume + down day | Capitulation signal |
| `Return_Mean_20D` | 20-day average return | Mean for z-score |
| `Return_Std_20D` | 20-day return std dev | Volatility for z-score |
| `Return_ZScore` | Return z-score | Statistical oversold (z < -2) |

**Total Features:** 42 → 54 (+12)

### 2. Per-Ticker Training Architecture

**New File:** `train_timing_model_per_ticker.py`

#### Ticker Groups

```python
TICKER_GROUPS = {
    'high_vol':    ['TSLA', 'NVDA', 'AMD'],      # High volatility, strong mean reversion
    'tech_growth': ['META', 'GOOGL', 'AMZN'],    # Tech with moderate volatility
    'tech_stable': ['AAPL', 'MSFT'],             # Lower volatility, steady growth
    'etf':         ['SPY', 'QQQ'],               # Market proxies
}
```

#### Why Groups Instead of Individual Models?

- **More Training Data:** 3 tickers per group = 3x more data than individual models
- **Shared Patterns:** NVDA and AMD have similar volatility characteristics
- **Better Generalization:** Prevents overfitting to single ticker quirks
- **Scalability:** Easy to add new tickers to existing groups

#### Adaptive Thresholds

Each group gets its own threshold based on historical drawdown distribution:

```python
# Example output (will vary):
high_vol:    -7.5%  (wider threshold for volatile stocks)
tech_growth: -5.2%  (moderate)
tech_stable: -3.1%  (tighter for stable stocks)
etf:         -4.0%  (market baseline)
```

**Why Adaptive?**
- Fixed -5% threshold doesn't work for all stocks
- NVDA routinely drops 8% and bounces → should still signal "good time"
- AAPL rarely drops 5% → -3% is more appropriate threshold

### 3. API Server Updates

**File:** `simple_api_server.py`

#### Model Loading (lines ~89-145)

```python
# Try per-ticker model first (preferred)
per_ticker_path = 'csp_timing_model_per_ticker.pkl'
if os.path.exists(per_ticker_path):
    TIMING_MODELS = timing_data['models']        # Dict: {group: model}
    TIMING_SCALERS = timing_data['scalers']      # Dict: {group: scaler}
    TIMING_THRESHOLDS = timing_data['thresholds'] # Dict: {group: threshold}
    TIMING_GROUP_MAPPING = timing_data['group_mapping']  # Dict: {ticker: group}
    PER_TICKER_MODE = True
else:
    # Fallback to global model
    MODEL = global_model
```

#### Prediction Logic (lines ~320-380)

```python
if PER_TICKER_MODE:
    group = TIMING_GROUP_MAPPING.get(ticker, 'tech_growth')  # Default group
    model = TIMING_MODELS[group]
    scaler = TIMING_SCALERS[group]
    print(f"[Per-Ticker] Using {group} model for {ticker}")
else:
    model = MODEL  # Global fallback
```

**Backward Compatible:** If per-ticker model doesn't exist, automatically uses global model.

### 4. Validation Script

**New File:** `validate_timing_model.py`

Compares global vs per-ticker models on:

1. **Overall Accuracy & ROC-AUC**
2. **Mean Reversion Capture Rate:** P(Good_Signal | Return_1D < -3%)
3. **Bounce Prediction:** After signal, how often does stock bounce?
4. **Per-Ticker Breakdown:** Which stocks benefit most?

**Usage:**
```bash
python3 validate_timing_model.py
# Or specific tickers:
python3 validate_timing_model.py --tickers NVDA AMD TSLA
```

---

## Training & Deployment

### Training Command

```bash
# Train on 5 years of data (recommended)
python3 train_timing_model_per_ticker.py --period 5y

# Train on 10 years (more data but slower)
python3 train_timing_model_per_ticker.py --period 10y --output csp_timing_model_10y.pkl

# Disable GPU if needed
python3 train_timing_model_per_ticker.py --period 5y --no-gpu
```

### Current Training Status

**Started:** Background training with 5y data
**Monitor:** `tail -f training_per_ticker.log`
**Output:** `csp_timing_model_per_ticker.pkl`

### Deployment Steps

1. **Wait for Training to Complete**
   ```bash
   tail -f training_per_ticker.log
   # Look for "✅ TRAINING COMPLETE!"
   ```

2. **Validate Models**
   ```bash
   python3 validate_timing_model.py
   ```

3. **Restart API Server**
   ```bash
   pkill -f simple_api_server
   python3 simple_api_server.py --port 8000
   ```

   API will automatically detect and load per-ticker model.

4. **Verify in API**
   ```bash
   curl http://localhost:8000/health
   # Should show: "model_type": "per_ticker"
   ```

---

## Expected Performance Improvements

### Hypothesis

| Metric | Global Model | Per-Ticker Model | Improvement |
|--------|--------------|------------------|-------------|
| Overall ROC-AUC | ~0.65 | ~0.70 | +7.7% |
| Mean Reversion Capture (after -5% drop) | 40% | 70% | +75% |
| NVDA-specific accuracy | 60% | 75% | +25% |
| AAPL-specific accuracy | 65% | 70% | +7.7% |

### Key Metric: Mean Reversion Capture Rate

**Definition:** After a stock drops >5%, what % of time does model signal "good time to sell CSP"?

**Target:**
- High-vol stocks (NVDA): 60-70% signal rate (matches historical bounce rate)
- Stable stocks (AAPL): 40-50% signal rate (less frequent bounces)

**Why This Matters:**
- User's key insight: "If NVDA drops hard, you'll make money the next day"
- Old model missed this because it didn't explicitly track drops
- New model has `Recent_Drop_5D`, `RSI_Oversold`, `Return_ZScore` features

---

## Rollback Plan

If per-ticker model underperforms:

1. **Keep Global Model as Fallback**
   - API automatically falls back if `csp_timing_model_per_ticker.pkl` missing
   - Just delete/rename the per-ticker file

2. **Mean Reversion Features Still Help**
   - Even global model benefits from new features
   - Can retrain global model with 54 features:
   ```bash
   python3 train_gpu.py --period 10y --trees 500
   ```

3. **Hybrid Approach**
   - Use per-ticker for high-vol (NVDA, TSLA, AMD)
   - Use global for others

---

## Files Modified/Created

### Modified
- ✅ `data_collector.py` - Added 12 mean reversion features (lines 367-399, 516-521)
- ✅ `simple_api_server.py` - Per-ticker model loading and prediction (lines 89-145, 320-380)

### Created
- ✅ `train_timing_model_per_ticker.py` - Per-ticker training script (320 lines)
- ✅ `validate_timing_model.py` - Model comparison and validation (290 lines)
- ✅ `TIMING_MODEL_V2_SUMMARY.md` - This document

### Training Outputs
- ⏳ `csp_timing_model_per_ticker.pkl` - Per-ticker models (in progress)
- ⏳ `training_per_ticker.log` - Training progress log

---

## Next Steps

1. ✅ **Implementation Complete**
   - Mean reversion features added
   - Per-ticker training script created
   - API server updated
   - Validation script ready

2. ⏳ **Training In Progress** (background)
   - Monitor: `tail -f training_per_ticker.log`
   - ETA: ~10-15 minutes for 5y data

3. ⬜ **Validation** (after training)
   ```bash
   python3 validate_timing_model.py
   ```

4. ⬜ **Deploy** (if validation successful)
   ```bash
   pkill -f simple_api_server
   python3 simple_api_server.py --port 8000
   ```

5. ⬜ **Test Live** (after deployment)
   ```bash
   curl http://localhost:8000/predict -X POST -H "Content-Type: application/json" -d '{"ticker": "NVDA"}'
   ```

6. ⬜ **Monitor Real Performance**
   - Track predictions over next 35 days
   - Compare to validation metrics
   - Fine-tune thresholds if needed

---

## Technical Details

### Feature Engineering

**Challenge:** 252-day features (like `Volatility_252D`, `IV_Rank`) require 1+ years of data.

**Solution:** Training requires 2+ years of data. API handles recent data gracefully by forward-filling.

### GPU Acceleration

**Status:** ✅ GPU detected and used for training
**Speedup:** ~3-5x faster than CPU for XGBoost
**Fallback:** Automatically uses CPU if GPU unavailable

### Model Size

- Global model: ~5 MB (1 XGBoost model)
- Per-ticker model: ~20 MB (4 XGBoost models + metadata)
- Still small enough for instant loading

---

## Questions & Answers

### Q: Why not train individual models for each ticker?

**A:** Not enough data. Even with 5-10 years, each ticker only has ~1,000-2,500 samples. Grouping similar tickers provides 3x more training data while preserving stock-specific characteristics.

### Q: What if I want to predict a ticker not in the groups?

**A:** API defaults to `tech_growth` group. You can also:
1. Add ticker to appropriate group in `train_timing_model_per_ticker.py`
2. Retrain models
3. Or use global model as fallback

### Q: How do I know which model is being used?

**A:** Check the prediction response:
```json
{
  "model_type": "XGBClassifier (Per-Ticker: high_vol)",
  ...
}
```

### Q: Can I use both models simultaneously?

**A:** Yes! The validation script does exactly this - runs both models and compares results.

---

## Conclusion

This implementation directly addresses your observation: **"If NVDA drops hard, you'll make money the next day."**

The new model:
1. ✅ Tracks recent drops explicitly (`Recent_Drop_1D/3D/5D`)
2. ✅ Learns stock-specific patterns (per-ticker groups)
3. ✅ Adjusts thresholds for volatility (adaptive)
4. ✅ Captures mean reversion (12 new features)

**Next:** Wait for training to complete, then validate and deploy!
