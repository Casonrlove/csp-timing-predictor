# CSP Timing Predictor

AI-powered predictions for optimal Cash Secured Put (CSP) timing using an LSTM + XGBoost ensemble model.

## Model Performance Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **ROC-AUC** | 0.9410 | Excellent |
| **Accuracy** | 79.58% | Good |
| **Precision** | 93.98% | Excellent |
| **Recall** | 56.66% | Moderate |
| **Walk-Forward Accuracy** | 54.44% | Concerning |
| **Brier Score** | 0.1592 | Fair |
| **Simulated Win Rate** | 94.0% | Excellent |
| **Profit Factor** | 11.71 | Excellent |
| **EV per Trade** | $1.40 | Positive |

### What Each Metric Means

#### ROC-AUC (0.9410)
**Receiver Operating Characteristic - Area Under Curve**

Measures how well the model ranks predictions. A score of 1.0 is perfect, 0.5 is random guessing.
- **0.94 = Excellent** - The model is very good at distinguishing between "safe" and "risky" CSP opportunities
- This means if you pick a random safe trade and a random risky trade, 94% of the time the model correctly ranks which is safer

#### Accuracy (79.58%)
**Percentage of correct predictions**

How often the model's binary prediction (safe vs risky) is correct.
- **79.58% = Good** - Model is right ~4 out of 5 times on historical data
- Note: Walk-forward accuracy (54%) is more realistic for live trading

#### Precision (93.98%)
**When model says "safe", how often is it actually safe?**

`Precision = True Positives / (True Positives + False Positives)`
- **93.98% = Excellent** - When the model recommends a CSP, it's almost always correct
- This is the most important metric for CSP sellers - you want high confidence in "safe" calls

#### Recall (56.66%)
**Of all actual safe opportunities, how many did the model catch?**

`Recall = True Positives / (True Positives + False Negatives)`
- **56.66% = Moderate** - Model misses ~43% of safe opportunities
- Trade-off: High precision but moderate recall means the model is conservative (better to miss opportunities than lose money)

#### Walk-Forward Accuracy (54.44%)
**Out-of-sample accuracy using rolling train/test windows**

More realistic measure of live performance - trains on past data, tests on future data, rolls forward.
- **54.44% = Concerning** - Barely better than coin flip in true out-of-sample testing
- This is common in financial models and highlights the difference between backtesting and live trading

#### Brier Score (0.1592)
**Probability calibration - are predicted probabilities accurate?**

Measures how close predicted probabilities are to actual outcomes. Lower is better.
- **0.0** = Perfect calibration
- **0.25** = Random guessing (always predicting 50%)
- **0.1592 = Fair** - Probabilities are somewhat calibrated but not perfectly reliable

#### Simulated Win Rate (94.0%)
**Percentage of simulated trades that were profitable**

Based on backtested CSP trades following model recommendations.
- **94.0% = Excellent** - 94 out of 100 trades would have been winners
- Note: CSPs have inherently high win rates; the key is avoiding catastrophic losses

#### Profit Factor (11.71)
**Total profits divided by total losses**

`Profit Factor = Gross Profits / Gross Losses`
- **> 1.0** = Profitable system
- **> 2.0** = Good system
- **11.71 = Excellent** - Profits are 11.7x larger than losses

#### Expected Value per Trade ($1.40)
**Average profit per trade**

`EV = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)`
- **$1.40 = Positive** - On average, each trade makes $1.40 (based on simulation assumptions)

---

## Key Insights

1. **High ROC-AUC but low walk-forward accuracy**: Model is good at *ranking* (which trades are better) but probabilities aren't perfectly calibrated for out-of-sample prediction.

2. **High precision, moderate recall**: Conservative model - prefers to miss opportunities rather than make bad recommendations.

3. **Profitable in simulation**: Despite accuracy concerns, the simulated P/L is strongly positive because:
   - High precision means recommended trades usually work
   - CSPs have natural edge (most stocks don't crash)
   - Model helps avoid the worst scenarios

---

## Architecture

```
Input Data (10yr Schwab API)
         │
    ┌────┴────┐
    ▼         ▼
  LSTM     Technical
(PyTorch)  Indicators
    │         │
    └────┬────┘
         ▼
   Combined Features (31)
         │
    ┌────┼────┐
    ▼    ▼    ▼
 XGBoost LightGBM RandomForest
    │    │    │
    └────┼────┘
         ▼
   Stacking Classifier
   (Logistic Regression)
         │
         ▼
   CSP Score & Prediction
```

## Usage

### Start Server
```bash
./start              # API server only
./start --all        # API server + ngrok tunnel
./start --train      # Train model, then start server
```

### Validate Model
```bash
python validate_model.py              # Full validation report
python validate_model.py --stats-only # Just prediction log stats
```

### API Endpoints
- `POST /predict` - Get CSP prediction for a ticker
- `GET /validation/stats` - View logged prediction accuracy
- `GET /validation/recent` - Recent predictions

## Requirements
- Python 3.10+
- PyTorch 2.6+ (CUDA 12.8 for RTX 5070 Ti)
- XGBoost, LightGBM, scikit-learn
- Schwab API credentials

## Data Sources
| Data | Source |
|------|--------|
| Historical OHLCV | Schwab API |
| Options/Greeks | Schwab API |
| VIX | Schwab API |
| Earnings Dates | Yahoo Finance |
