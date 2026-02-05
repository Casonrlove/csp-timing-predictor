# CSP Timing Model Architecture

## Overview

The CSP Timing Predictor uses a hybrid deep learning + ensemble approach to predict optimal times to sell Cash Secured Puts.

## Architecture Diagram

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                    INPUT DATA                           │
                        │  (10 years historical prices from Schwab API)           │
                        └─────────────────────────────────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
           ┌────────────────────────┐                      ┌────────────────────────┐
           │   LSTM Time Series     │                      │  Technical Indicators  │
           │   (PyTorch, GPU)       │                      │  (27 features)         │
           │                        │                      │                        │
           │  - 20-day sequences    │                      │  - RSI, MACD, ADX      │
           │  - 2 LSTM layers       │                      │  - Bollinger Bands     │
           │  - 128 hidden units    │                      │  - Moving Averages     │
           └────────────────────────┘                      │  - ATR, Volatility     │
                        │                                  │  - Support/Resistance  │
                        ▼                                  │  - VIX, IV Rank        │
           ┌────────────────────────┐                      └────────────────────────┘
           │   LSTM Predictions     │                                 │
           │                        │                                 │
           │  - 5-day return        │                                 │
           │  - 10-day return       │                                 │
           │  - Direction prob      │                                 │
           └────────────────────────┘                                 │
                        │                                             │
                        └──────────────────┬──────────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │              COMBINED FEATURE SET (30 features)         │
                        └─────────────────────────────────────────────────────────┘
                                                 │
                 ┌───────────────────────────────┼───────────────────────────────┐
                 ▼                               ▼                               ▼
    ┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
    │       XGBoost          │    │       LightGBM         │    │    Random Forest       │
    │                        │    │                        │    │                        │
    │  - GPU accelerated     │    │  - GPU accelerated     │    │  - 928 trees           │
    │  - Optuna tuned        │    │  - Optuna tuned        │    │  - Optuna tuned        │
    │  - 100 trials          │    │  - 100 trials          │    │  - 50 trials           │
    └────────────────────────┘    └────────────────────────┘    └────────────────────────┘
                 │                               │                               │
                 └───────────────────────────────┼───────────────────────────────┘
                                                 │
                                                 ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │            STACKING CLASSIFIER                          │
                        │            (Logistic Regression Meta-Learner)           │
                        │                                                         │
                        │  Learns optimal weights for each base model             │
                        └─────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │                 FINAL PREDICTION                        │
                        │                                                         │
                        │  "GOOD TIME TO SELL CSP" or "WAIT - NOT OPTIMAL"        │
                        │  + Confidence score                                     │
                        └─────────────────────────────────────────────────────────┘
```

## Components

### 1. LSTM Time Series Model

The LSTM captures sequential patterns in price movements that tree-based models might miss.

**Input:** 20-day sequences of:
- Close price, Volume
- RSI, MACD, Bollinger Band position
- ATR%, Returns (1D, 5D)
- Volatility, Price vs SMAs

**Output:** 3 predictions used as features:
- Predicted 5-day return
- Predicted 10-day return
- Direction probability (up/down)

### 2. Technical Indicators (27 features)

Calculated from Schwab historical data:

| Category | Features |
|----------|----------|
| Price Position | Price vs SMA20/50/200, Distance from Support/Resistance |
| Momentum | RSI, Stochastic K/D, MACD, MACD Signal/Diff |
| Volatility | Bollinger Position/Width, ATR%, 20D/252D Volatility, IV Rank |
| Trend | ADX |
| Volume | Volume Ratio, OBV |
| Returns | 1D, 5D, 20D returns |
| Market | VIX |
| Earnings | Days to earnings, Near earnings flag |

### 3. Ensemble Models

Three gradient boosting / tree models trained in parallel:

**XGBoost:**
- GPU accelerated (CUDA)
- 100 Optuna hyperparameter trials
- Handles class imbalance with scale_pos_weight

**LightGBM:**
- GPU accelerated
- 100 Optuna trials
- Fast training on large datasets

**Random Forest:**
- 928 estimators
- 50 Optuna trials
- Provides diversity in ensemble

### 4. Stacking Classifier

Logistic Regression meta-learner that:
- Takes predictions from all 3 base models
- Learns optimal weights for different market conditions
- Outputs final probability

## Training Pipeline

```bash
./start --train
```

1. **Data Collection** - Fetches 10 years of data for 30+ tickers from Schwab API
2. **LSTM Training** - 50 epochs on GPU (~5 minutes)
3. **Feature Generation** - LSTM predictions added to feature set
4. **Hyperparameter Tuning** - Optuna optimizes each model (100 trials each)
5. **Ensemble Training** - StackingClassifier with 3-fold CV
6. **Model Saving** - Saves both LSTM (.pt) and ensemble (.pkl)

## Data Sources

| Data | Source |
|------|--------|
| Historical OHLCV | Schwab API |
| VIX | Schwab API |
| Current Price | Schwab API |
| Options/Greeks | Schwab API |
| Earnings Dates | Yahoo Finance |

## Files

- `train_ultimate.py` - Main training script
- `lstm_features.py` - LSTM model and feature generator
- `data_collector.py` - Data collection and technical indicators
- `csp_model_multi.pkl` - Trained ensemble model
- `csp_model_multi_lstm.pt` - Trained LSTM model

## Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (RTX 5070Ti recommended)
- **RAM:** 16GB+ recommended
- **Training Time:** ~30-60 minutes with GPU
