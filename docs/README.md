# CSP Timing Predictor

AI-powered Cash Secured Put (CSP) analysis — live options data, contract-level breach probability, and expected value modeling.

![Model](https://img.shields.io/badge/Model-XGB%2BLGBM%20Ensemble-purple)
![Tickers](https://img.shields.io/badge/Trained%20Tickers-89-blue)
![Features](https://img.shields.io/badge/Features-63-green)
![Data](https://img.shields.io/badge/Training%20Data-10%20years-orange)

---

## What It Does

For any ticker you enter, the system:

1. **Fetches live market data** from Schwab API (price, options chain, VIX)
2. **Calculates 88 technical indicators** (RSI, MACD, Bollinger Bands, ATR, IV Rank, VIX regime, Hurst exponent, etc.)
3. **Predicts breach probability** for each delta bucket (0.10–0.50) using a trained XGB+LGBM stacking ensemble
4. **Scores live options** from the Schwab chain with Edge, EV, ROR, and Theta Efficiency
5. **Recommends the best strike** based on maximum positive EV after slippage

---

## Architecture

### Contract Model (`csp_contract_model.pkl`)

The primary model — predicts `P(strike_breach)` for a specific delta bucket given current market conditions.

| Component | Detail |
|-----------|--------|
| Base learners | XGBoost (GPU, `tree_method=hist`) + LightGBM DART |
| Meta-learner | Logistic Regression on `[xgb_prob, lgbm_prob, VIX, VIX_Rank, Regime_Trend]` |
| Calibration | Isotonic regression on held-out 20% OOF data |
| CV strategy | 5-fold contract-aware time split (date-level, no future leakage) |
| Recency weighting | 2-year half-life exponential decay |
| Regime split | VIX boundary at 18.0 → `low_vix` / `high_vix` models |
| Features | 56 market indicators + 7 contract features = **63 total** |
| Target | `strike_breached` — did price touch the strike within 35 days? |

### Ticker Groups (4 × 2 VIX regimes = 8 models)

| Group | Tickers | Notes |
|-------|---------|-------|
| `high_vol_semi` | NVDA, AMD, TSLA, PLTR, SMCI, ARM, MU, MRVL, LRCX, AMAT, KLAC, ON, APP, RDDT, ACLS, ENTG, MPWR, COHR, CRDO, AVGO | Primary trading universe — IV typically >40% |
| `large_cap_tech` | MSFT, AAPL, GOOGL, META, AMZN, NFLX, CRM, ADBE, NOW, SNOW, SHOP, UBER, PANW, CRWD, ZS, DDOG, NET, DASH, ABNB, ORCL | Moderate IV, high liquidity |
| `dividend_value` | TXN, QCOM, CSCO, IBM, INTC, VZ, T, JNJ, ABBV, MRK, PFE, KO, PEP, WMT, JPM, BAC, V, MA, CVX, XOM | Lower IV, high win rate |
| `etf` | QQQ, TQQQ, SQQQ, XLK, SMH, SOXX, SOXL, ARKK, SPY, VOO, IVV, VTI, IWM, MDY, DIA, GLD, SLV, TLT, HYG, XLF, XLE, XLV, SCHB + more | Reference — not actively traded |

Unknown tickers fall back to the `large_cap_tech` model.

### Key Metrics Explained

| Metric | What it means |
|--------|--------------|
| **CSP Score** | Model confidence this is a good time to sell (ticker timing signal, 0–1) |
| **Edge** | `market_delta − model_P(breach)` — how much the market is overpaying for risk |
| **EV** | `premium − P(breach) × loss − slippage` — expected dollars per share |
| **ROR** | `premium / strike` — raw yield, ignores breach probability |
| **Theta Eff** | `theta / premium` — how fast you collect per dollar at risk |

**What to prioritize:** High Edge + positive EV + CSP Score > 0.55

---

## Web Interface

The UI is a single HTML file (`index.html`) served by GitHub Pages. It connects to your local server via an ngrok tunnel.

### Sections

- **Single Ticker Prediction** — full analysis with options table, technical indicators, candlestick chart
- **Compare Multiple Tickers** — side-by-side EV/Edge/Score comparison
- **Top CSP Opportunities** — scans all 60 tradeable tickers, ranks by edge
- **Smart Search** — autocomplete across all 100 tickers in 5 buckets; reference-only tickers (ETFs) searchable but not scanned

---

## Local Setup

### Prerequisites
- Python 3.10+
- Schwab Developer account (for live data)
- ngrok account (free tier works)
- NVIDIA GPU recommended (XGBoost + LightGBM GPU training)

### Installation

```bash
git clone https://github.com/Casonrlove/csp-timing-predictor.git
cd csp-timing-predictor
pip install -r requirements.txt
```

### Configure Schwab API

Create `.env` with your Schwab credentials:
```
SCHWAB_CLIENT_ID=your_app_key
SCHWAB_CLIENT_SECRET=your_secret
SCHWAB_REDIRECT_URI=https://127.0.0.1
```

Authenticate once to generate `schwab_tokens.json`:
```bash
python schwab_auth.py
```

### Train the Model

```bash
./start --train
```

Fetches 10 years of data for all 89 tickers from Schwab, trains 8 regime-specific ensembles. Takes **1–2 hours**. Saves to `csp_contract_model.pkl`.

### Run the Server

```bash
./start --all        # API server + ngrok tunnel
./start              # API server only (if ngrok already running)
```

Server runs on `http://localhost:8000`. Open `index.html` in a browser and enter your ngrok URL.

---

## Monte Carlo Simulator (Standalone)

GPU-accelerated GARCH(1,1) price path simulation — independent of the API server.

```bash
python monte_carlo_sim.py NVDA
python monte_carlo_sim.py NVDA --paths 200000000   # 200M paths (default)
python monte_carlo_sim.py NVDA --forward-days 45
```

- Fits GARCH(1,1) to recent returns (arch library or scipy fallback)
- Runs 200M paths × 35 days in ~2s on RTX 5070 Ti (float16, raw PyTorch)
- Fetches live strikes + premiums from Schwab
- Outputs: breach probability per delta, EV curve, path-minimum distribution chart

---

## Data Sources

| Source | Used for |
|--------|---------|
| Schwab API | Live price, options chain, historical OHLCV |
| CBOE via Schwab | VIX, VIX9D |
| Calculated | All 88 technical indicators |

Yahoo Finance was removed — all data comes from Schwab.

---

## Project Structure

```
├── simple_api_server.py       # FastAPI server — prediction endpoint
├── data_collector.py          # Schwab data fetch + 88 technical indicators
├── schwab_client.py           # Schwab API wrapper (options chain, prices)
├── train_contract_model.py    # Model trainer — XGB+LGBM stacking ensemble
├── feature_utils.py           # Feature engineering, CV splits
├── monte_carlo_sim.py         # Standalone GPU GARCH Monte Carlo
├── index.html                 # Web UI (GitHub Pages)
├── start                      # Launch script (./start --all / --train / --kill)
├── csp_contract_model.pkl     # Trained model (git-tracked)
└── docs/
    └── README.md              # This file
```

---

## ⚠️ Disclaimer

For educational and informational purposes only. Not financial advice. Options trading involves substantial risk of loss. Always do your own research.

---

**Last trained:** February 2026 — 89 tickers, 8 regime models, 63 features, 10 years of data
