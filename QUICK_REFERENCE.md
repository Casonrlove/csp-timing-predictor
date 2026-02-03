# CSP Timing System - Quick Reference

## 🚀 Daily Commands

```bash
# Get prediction for single ticker
python predictor.py NVDA

# Scan multiple tickers
python predictor.py NVDA,AMD,TSLA,AAPL,MSFT

# Run manual alert check
python daily_alerts.py

# Start web interface
./start_web_system.sh
# Then in new terminal: ngrok http 8000
```

## 📊 Current Signals (Feb 2, 2026)

| Ticker | Signal | Confidence | Price |
|--------|--------|------------|-------|
| **NVDA** | ✅ **GOOD** | **68.1%** | $185.61 |
| **AAPL** | ✅ **GOOD** | **53.5%** | $270.01 |
| TSLA | ⏸️ WAIT | 45.3% | $421.81 |
| AMD | ⏸️ WAIT | 33.1% | $246.27 |
| MSFT | ⏸️ WAIT | 26.5% | $423.37 |

## 🎯 Performance

- **ROC-AUC**: 0.60
- **Precision**: 64%
- **Features**: 28
- **Training**: 10 years, 2,208 samples

## 📂 Key Files

| File | Purpose |
|------|---------|
| `predictor.py` | Get predictions |
| `model_trainer.py` | Retrain model |
| `daily_alerts.py` | Automated alerts |
| `start_web_system.sh` | Start web interface |
| `train_multi_ticker.py` | Train on multiple stocks |

## ⚙️ Monthly Maintenance

```bash
# Retrain with latest data (takes 2-3 min)
python model_trainer.py

# Or train multi-ticker for better generalization
python train_multi_ticker.py
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | `python model_trainer.py` |
| Web not working | `./start_web_system.sh` |
| Alerts not running | `crontab -l` to check |
| Need help | Read `DEPLOYMENT_COMPLETE.md` |

## 💡 Confidence Guide

- **>65%**: Strong signal - Consider trade
- **55-65%**: Good signal - Review carefully
- **<55%**: Weak signal - Wait

## 📈 Strategy

- **Delta**: 0.25-0.35
- **DTE**: 30-45 days
- **Max Loss**: -5%
- **Focus**: Support levels

## 🌐 Web Access

1. Start: `./start_web_system.sh`
2. Expose: `ngrok http 8000`
3. Access: Your GitHub Pages URL
4. Enter: Your ngrok URL

## 📧 Setup Alerts

```bash
# Install automated daily alerts
./setup_daily_alerts.sh

# Runs Mon-Fri at 4:30 PM after market close
```

## 📚 Documentation

- `SYSTEM_READY.md` - Overview
- `DEPLOYMENT_COMPLETE.md` - Full deployment
- `COMPLETE_GUIDE.md` - Comprehensive docs
- `QUICKSTART.md` - Quick start
