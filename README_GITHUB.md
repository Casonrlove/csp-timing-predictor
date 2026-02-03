# CSP Timing Predictor

AI-powered predictions for optimal Cash Secured Put (CSP) timing using machine learning.

![Model Performance](https://img.shields.io/badge/ROC--AUC-0.60-green)
![Features](https://img.shields.io/badge/Features-28-blue)
![Data](https://img.shields.io/badge/Training%20Data-10%20years-orange)

## 🎯 What It Does

Predicts the optimal timing to sell Cash Secured Puts (CSPs) on stocks by analyzing:
- **28 technical indicators** (RSI, MACD, Bollinger Bands, Support/Resistance, etc.)
- **Volatility metrics** (IV Rank, Volatility Ratio)
- **Earnings proximity** (won't recommend before earnings)
- **10 years of historical data**

**Strategy optimized for:**
- Delta: 0.25-0.35 (balanced)
- DTE: 30-45 days
- Max drawdown: -5%

## 📊 Performance

- **ROC-AUC**: 0.60 (solid predictive power)
- **Precision**: 64% (when it says GOOD, it's right 64% of the time)
- **Training**: 2,208 samples from 10 years of data
- **Model**: Random Forest (best performer)

## 🚀 Web Interface

**Live Demo**: [Your GitHub Pages URL]

Enter your API URL (from ngrok) and get instant predictions on any ticker!

## 🖥️ Local Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/csp-timing-predictor.git
cd csp-timing-predictor

# Install dependencies
pip install -r requirements.txt

# Train the model (2-3 minutes)
python model_trainer.py

# Get predictions
python predictor.py NVDA
```

## 📱 Usage

### Command Line

```bash
# Single ticker
python predictor.py NVDA

# Multiple tickers
python predictor.py NVDA,AMD,TSLA,AAPL,MSFT
```

### Web Interface

1. **Start backend server:**
   ```bash
   python api_server.py
   ```

2. **Expose with ngrok:**
   ```bash
   ngrok http 8000
   ```

3. **Open web interface:**
   - Go to your GitHub Pages URL
   - Enter ngrok URL
   - Get predictions!

### Automated Daily Alerts

```bash
# Setup daily alerts (runs at 4:30 PM after market close)
./setup_daily_alerts.sh

# Or run manually
python daily_alerts.py
```

## 🎓 Example Output

```
CSP TIMING PREDICTION FOR NVDA
======================================================================

Current Price: $185.61
Date: 2026-02-02

RECOMMENDATION:      GOOD TIME TO SELL CSP
Confidence:          68.1%
Probability Good:    68.1%
Probability Bad:     31.9%

[Support/Resistance]
  Distance from Support:      4.50%
  Distance from Resistance:   4.78%

[Moving Averages]
  Price vs SMA20:   -0.51%
  Price vs SMA50:    0.93%
  Price vs SMA200:  10.41%

[Momentum Indicators]
  RSI:              48.81
```

## 📈 Features

- **28 Technical Indicators**:
  - Support/Resistance levels
  - Moving averages (SMA20, 50, 200)
  - Momentum (RSI, MACD, Stochastic)
  - Volatility (Bollinger Bands, ATR, IV Rank)
  - Volume indicators
  - Earnings proximity

- **Multi-Ticker Support**: Train on multiple stocks for better generalization

- **Web Interface**: Beautiful, mobile-responsive interface for checking signals anywhere

- **Automated Alerts**: Daily scans after market close

## 🔧 Advanced Usage

### Train Multi-Ticker Model

```bash
python train_multi_ticker.py
```

Trains on: NVDA, AMD, TSLA, AAPL, MSFT, GOOGL, META, AMZN

### Customize Strategy

Edit `data_collector.py`:

```python
def create_target_variable(self, forward_days=35, threshold_pct=-5):
    # Adjust forward_days for different DTE targets
    # Adjust threshold_pct for risk tolerance
```

## 📚 Documentation

- [SYSTEM_READY.md](SYSTEM_READY.md) - Quick start guide
- [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md) - Full deployment instructions
- [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) - Comprehensive documentation

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Always do your own research and consult with a financial advisor before making investment decisions. Past performance does not guarantee future results.

Trading options involves substantial risk of loss and is not suitable for all investors.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with PyTorch, scikit-learn, XGBoost
- Data from Yahoo Finance
- Powered by Claude Sonnet 4.5

---

**Built on**: Feb 2, 2026
**Model**: Random Forest, 0.60 ROC-AUC
**Data**: 10 years, 2,208 samples, 28 features
