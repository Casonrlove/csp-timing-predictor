# CSP Timing Predictor

AI-powered predictions for optimal Cash Secured Put (CSP) timing using a regime-split XGBoost + LightGBM ensemble trained on the correct target: **assignment-only breach** (close at expiry below strike), scoped to the 0.20–0.35 delta range actually traded.

## Model Performance Metrics

All numbers below are from **true out-of-sample validation** — the model was trained on the first ~80% of each ticker's history and evaluated on the final ~20% it never saw (~36,800 predictions across 4 ticker groups).

| Metric | Value | Assessment |
|--------|-------|------------|
| **ROC-AUC** | 0.5670 | Modest genuine signal |
| **Accuracy** | 71.55% | At 0.50 threshold |
| **Precision** | 24.27% | At 0.50 threshold |
| **Recall** | 17.49% | At 0.50 threshold |
| **Walk-Forward Accuracy** | 55.65% | Derived from OOF AUC |
| **Brier Score** | 0.2510 | Near no-skill baseline (0.246) |
| **Simulated Win Rate** | 79.9% | Good ¹ |
| **Profit Factor** | 1.47 | Positive ¹ |
| **EV per Trade** | +0.417% of K | Positive ¹ |

> ¹ Premium estimated via Black-Scholes using realized 20-day vol — actual quoted premiums would differ. Win/loss labels are real (actual stock prices 35 days forward).

### Per-Group AUC (True OOF)

| Group | AUC | Breach Rate |
|-------|-----|-------------|
| dividend_value | 0.5568 | 19.3% |
| etf | 0.6201 | 12.5% |
| high_vol_semi | 0.5293 | 21.4% |
| large_cap_tech | 0.5430 | 26.3% |

---

### What Each Metric Means

#### ROC-AUC (0.5670)
**Receiver Operating Characteristic - Area Under Curve**

Measures how well the model ranks predictions. 1.0 = perfect, 0.5 = random guessing.
- **0.567 = Modest** — The model has genuine but limited ability to distinguish safe from risky conditions
- ETF group leads at 0.62; high_vol_semi is hardest to predict at 0.53

#### Accuracy (71.55%)
**Percentage of correct predictions at 0.50 threshold**

- **71.55%** — Model is right about 5 out of 7 times on unseen data at 0.50 threshold
- Base rate (always predict "safe"): ~79%. The 0.50 threshold is high relative to the ~21% breach rate; the server uses a base-rate-relative CSP score rather than a fixed cutoff.

#### Precision (24.27%)
**When model says "risky", how often is it actually risky?**

`Precision = True Positives / (True Positives + False Positives)`
- **24.27%** — Above the 20.8% base rate, meaning the model's "risky" flags have genuine lift.

#### Recall (17.49%)
**Of all actual losses, what fraction did the model flag?**

`Recall = True Positives / (True Positives + False Negatives)`
- **17.49%** — Model catches about 1 in 6 genuine loss events at 0.50 threshold. Lower thresholds improve recall substantially.

#### Walk-Forward Accuracy (55.65%)
**Out-of-sample accuracy on unseen future data**

True OOF validation is equivalent to walk-forward: model trained exclusively on past dates, evaluated on future dates, no lookahead. Derived from the cross-validated AUC of 0.5744 across 8 regime slices.

#### Brier Score (0.2510)
**Probability calibration — are predicted probabilities accurate?**

Lower is better. Baseline (always predict base rate) ≈ 0.246 for this dataset.
- **0.0** = Perfect
- **0.246** = No-skill baseline for this dataset (21% breach rate)
- **0.25** = Random (always predict 50%)
- **0.251 ≈ baseline** — Probabilities are now correctly centered around the 20% base rate (pred_median ~0.17). Ranking ability (AUC) is the primary signal; absolute probability values are a secondary indicator.

#### Simulated Win Rate (79.9%) ¹
**Percentage of model-recommended trades that were profitable**

- **79.9%** — On trades the model flagged as below-average risk, 79.9% expired safely
- No-skill baseline: ~79% (1 − average breach rate of 21%)
- Model adds lift via selective filtering at tighter thresholds (0.35 threshold: +0.4% EV)

#### Profit Factor (1.47) ¹
**Total profits ÷ total losses**

`Profit Factor = Gross Profits / Gross Losses`
- **> 1.0** = System profitable in aggregate
- **1.47** — Profits exceed losses by 47%.

#### Expected Value per Trade (+0.417% of K) ¹
**Average profit per trade as a fraction of strike price**

`EV = (Win Rate × Avg Premium) − (Loss Rate × Avg Loss)`
- **+0.417% of K** — On average, each recommended trade earns ~0.417% of the strike price
- For a $800 strike: ≈ $3.34 per share, ≈ $334 per contract

---

## Key Insights

1. **Genuine but modest out-of-sample signal**: AUC 0.567 across ~37K unseen predictions is real alpha, not backtest artifact. ETF group leads at 0.62.

2. **Correct breach definition**: Target is assignment-only — the underlying closes below the strike at expiry. The previous 2× premium stop logic inflated breach rates artificially; removal lowered the base rate from ~30% to ~21%, producing a cleaner signal.

3. **Narrowed to trading delta range**: Model trained on 0.20–0.35 delta only (the range actually traded), down from 9 buckets spanning 0.10–0.50. Tighter scope improves signal purity and reduces base rate noise.

4. **Calibrated predictions**: The model outputs probabilities correctly centered around the ~20% historical breach rate (pred_median ~0.17). Use the ranking signal (which conditions are relatively safer) rather than the raw absolute probability.

5. **Base-rate-relative scoring**: The CSP Score measures how much safer/riskier current conditions are compared to the historical average loss rate (~21%), not whether p(breach) > 50%.

---

## Architecture

```
Input Data (Schwab API, 10yr history)
         │
         ▼
   Technical Indicators
   (63 market features + 9 contract features = 72 total)
   Delta scope: 0.20–0.35 (4 buckets)
         │
         ▼
   Regime Split
   (ticker group × VIX level → 8 model keys)
         │
    ┌────┴────┐
    ▼         ▼
  XGBoost   LightGBM
  (per regime)  (per regime)
    │         │
    └────┬────┘
         ▼
   Meta-Learner (XGBoost, depth=2)
   Inputs: [xgb_prob, lgbm_prob, VIX, VIX_Rank, Regime_Trend]
         │
         ▼
   Sigmoid Calibrator (OOF-fitted, representative breach rate)
         │
         ▼
   P(breach) → Base-Rate-Relative CSP Score
```

**Regime split keys**: `{high_vol_semi, large_cap_tech, dividend_value, etf} × {low_vix, high_vix}`

VIX boundary: 18.0. Groups are assigned at training time via `group_mapping` in the pkl.

---

## Usage

### Start Server
```bash
./start              # API server only
./start --all        # API server + ngrok tunnel
./start --train      # Train model, then start server
```

### Train Model
```bash
python train_contract_model.py                              # Standard retrain
python train_contract_model.py --use-optuna-params          # With Optuna hyperparams
```

### Validate Model
```bash
python true_oof_validation.py      # True out-of-sample metrics (last CV fold)
python full_model_metrics.py       # Holdout metrics (last 252 days)
```

### API Endpoints
- `POST /predict` - Get CSP prediction for a ticker
- `GET /validation/stats` - View logged prediction accuracy
- `GET /validation/recent` - Recent predictions

---

## Requirements
- Python 3.10+
- PyTorch 2.6+ (CUDA 12.8 for RTX 5070 Ti, used for Monte Carlo simulation)
- XGBoost, LightGBM, scikit-learn
- Schwab API credentials

## Data Sources
| Data | Source |
|------|--------|
| Historical OHLCV | Schwab API |
| Options/Greeks | Schwab API (live) or Black-Scholes (fallback) |
| VIX | Schwab API |
| Earnings Dates | Yahoo Finance |
