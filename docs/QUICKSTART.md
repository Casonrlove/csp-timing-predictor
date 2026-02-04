
# Quick Start Guide

Get your CSP timing predictor up and running in minutes!

## Current Status

✅ All code is ready
✅ Dependencies installed
✅ Data fetching works
⚠️ **GPU note**: Your RTX 5070 Ti requires PyTorch nightly build (see GPU_SETUP.md)
   - For now, training will use CPU (works fine, just slower)

## Step 1: Train Your First Model (15-30 minutes on CPU)

Choose one:

### Option A: Train Hybrid Model (Recommended)
```bash
python train_quick.py
```

Best balance of performance and training time.

### Option B: Train TabNet (Fastest - 5-10 min)
```bash
python tabnet_trainer.py
```

Optimized for CPU, includes feature importance.

### Option C: Train Basic Models (Fastest - 2-3 min)
```bash
python model_trainer.py
```

Good baseline, uses sklearn.

## Step 2: Test Predictions

After training completes:

### Using Deep Learning Model:
```bash
python deep_predictor.py csp_hybrid_model.pkl NVDA
```

### Using TabNet:
```bash
python deep_predictor.py csp_tabnet_nvda.pkl NVDA
```

### Using Basic Model:
```bash
python predictor.py NVDA
```

## Step 3: Start the Web Server (Optional)

To access from work:

### Terminal 1 - Start API Server:
```bash
./start_server.sh
```

### Terminal 2 - Expose with ngrok:
```bash
# Install ngrok first: https://ngrok.com/download
ngrok http 8000
```

Copy the `https://xxxxx.ngrok.io` URL

### Deploy Frontend to GitHub:
1. Create a new repo on GitHub
2. Push the `web/` folder
3. Enable GitHub Pages
4. Open your GitHub Pages URL
5. Enter your ngrok URL
6. Get predictions from anywhere!

## What Each Model Does

| Model | Best For | Training Time (CPU) | Accuracy |
|-------|----------|---------------------|----------|
| **Hybrid (LSTM+Transformer)** | Best overall performance | 15-30 min | Best |
| **TabNet** | Interpretability, feature importance | 5-10 min | Very Good |
| **LSTM** | Temporal patterns | 15-25 min | Good |
| **Transformer** | Complex relationships | 15-25 min | Good |
| **XGBoost** | Fast baseline | 2-3 min | Baseline |

## Expected Performance

With 5 years of NVDA data:
- **Samples**: ~1021 (after removing NaN)
- **Features**: 23 technical indicators
- **Target**: "Good CSP timing" (stock doesn't drop >3% in next 35 days)
- **Positive class**: ~30% (good timing is rare)
- **Current performance**: 0.53-0.65 ROC-AUC

**Note**: This is a hard problem! Predicting 35-day returns is challenging.

## Improving Results

### 1. Relax the Criteria

Edit `data_collector.py` line 143:
```python
def create_target_variable(self, forward_days=21, threshold_pct=-5):
    # Changed from 35 days/-3% to 21 days/-5%
```

This gives more "good" examples to learn from.

### 2. Add More Data

```python
collector = CSPDataCollector('NVDA', period='10y')
```

### 3. Train on Multiple Tickers

```bash
# Edit tabnet_trainer.py to include more tickers
python tabnet_trainer.py
```

## Troubleshooting

### "No samples after feature engineering"
- VIX data issue (fixed automatically)
- Check internet connection

### "CUDA not compatible"
- Expected with RTX 5070 Ti
- See GPU_SETUP.md for PyTorch nightly installation
- Or just use CPU (works fine)

### Training is slow
- CPU training: 15-30 min is normal
- Use TabNet for faster training (5-10 min)
- Or install PyTorch nightly for GPU support

### Model predicts "WAIT" too often
- Model is being conservative
- Try relaxing threshold_pct to -5%
- Or train with more data

## File Overview

**Training:**
- `train_quick.py` - Quick training script (Hybrid only)
- `deep_learning_model.py` - Train all deep learning models
- `tabnet_trainer.py` - Train TabNet
- `model_trainer.py` - Train basic models

**Prediction:**
- `deep_predictor.py` - Predictions with deep learning models
- `predictor.py` - Predictions with basic models

**Web App:**
- `api_server.py` - Backend API server
- `web/index.html` - Frontend interface
- `start_server.sh` - Quick start script

**Documentation:**
- `QUICKSTART.md` - This file
- `COMPLETE_GUIDE.md` - Comprehensive documentation
- `GPU_SETUP.md` - GPU setup for RTX 5070 Ti
- `DEPLOYMENT_GUIDE.md` - Web deployment instructions

**Data:**
- `data_collector.py` - Data fetching and feature engineering

## Next Steps

1. ✅ **Train a model**: `python train_quick.py`
2. ✅ **Test predictions**: `python deep_predictor.py csp_hybrid_model.pkl NVDA`
3. ✅ **Review results**: Check if predictions make sense
4. ⏭️ **Paper trade**: Track predictions for a week
5. ⏭️ **Deploy web app**: Follow DEPLOYMENT_GUIDE.md
6. ⏭️ **Live trade**: Start with small positions

## Tips for Success

1. **Start simple**: Use the Hybrid or TabNet model first
2. **Test predictions**: Compare against your own analysis
3. **Paper trade first**: Track accuracy before risking capital
4. **Retrain monthly**: Markets change, keep models fresh
5. **Use as a tool**: Not a crystal ball - combine with your analysis
6. **Start small**: Even if confident, start with small CSP positions

## Getting Help

**Model not training?**
- Check data_collector.py runs: `python data_collector.py`
- Check dependencies: `pip list | grep torch`

**Predictions seem wrong?**
- Check technical indicators manually
- Compare with your own analysis
- Model might be too conservative (adjust threshold)

**Want to customize?**
- Edit `data_collector.py` for different indicators
- Edit target variable definition for different strategy
- Train on your preferred tickers

## Ready to Start?

```bash
# Train your first model
python train_quick.py

# Wait 15-30 minutes...

# Test it out
python deep_predictor.py csp_hybrid_model.pkl NVDA

# Start trading! (paper trade first!)
```

Good luck! 🚀💰
