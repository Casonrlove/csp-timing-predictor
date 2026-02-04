# GPU Setup for RTX 5070 Ti

Your RTX 5070 Ti (CUDA sm_120) is too new for the stable PyTorch release. Here's how to fix it:

## Problem

Current PyTorch (2.5.1 stable) only supports up to CUDA sm_90.
Your RTX 5070 Ti uses CUDA sm_120 (Blackwell architecture).

## Solution Options

### Option 1: Install PyTorch Nightly (Recommended)

PyTorch nightly builds have support for newer GPUs:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install nightly with CUDA 12.4 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

**Test it:**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Option 2: Wait for PyTorch 2.6 Stable

PyTorch 2.6 stable (expected Q1 2025) should support RTX 50-series GPUs.

### Option 3: Use CPU (Current Setup)

The models will train on CPU - slower but works fine:
- LSTM/Transformer: ~15-30 min training
- TabNet: ~5-10 min training
- Predictions: <1 second (fast enough)

For the CSP use case, CPU is perfectly acceptable since you're not training frequently.

## Performance Comparison

| Hardware | Training Time | Prediction | Cost |
|----------|---------------|------------|------|
| RTX 5070 Ti (GPU) | 3-5 min | <100ms | $0 |
| CPU (16 cores) | 15-30 min | <1s | $0 |
| CPU (8 cores) | 30-60 min | <1s | $0 |

**Recommendation**: Stick with CPU for now unless you plan to:
- Train models daily
- Train on 10+ tickers
- Experiment with hyperparameters frequently

## Current Status

Your system will automatically use CPU when GPU isn't compatible. You'll see:

```
Using device: cpu
Note: Training on CPU. For RTX 5070 Ti, install PyTorch 2.6+ for GPU support.
```

This is normal and expected.

## After Installing Nightly Build

1. Test GPU access:
```bash
python -c "import torch; torch.zeros(1).cuda()"
```

2. Retrain your model:
```bash
python train_quick.py
```

You should see:
```
Using device: cuda
GPU: NVIDIA GeForce RTX 5070 Ti
```

3. Training should be 5-10x faster

## Troubleshooting

**"RuntimeError: CUDA error: no kernel image is available"**
- Your PyTorch version doesn't support your GPU
- Install nightly build as shown above

**"CUDA out of memory"**
- Reduce batch_size in training
- Your 17GB VRAM should be more than enough though

**"ImportError: cannot import name 'packaging'"**
- Install packaging: `pip install packaging`

## Alternative: Use TabNet Instead

TabNet is optimized for CPU and trains faster:

```bash
python tabnet_trainer.py
```

TabNet benefits:
- CPU-optimized (less GPU dependency)
- Faster training (5-10 min on CPU)
- Feature importance visualization
- Can train on multiple tickers easily

## Questions?

The system works perfectly on CPU for your use case. GPU acceleration is nice-to-have but not required for:
- Daily CSP timing checks
- Single ticker predictions
- Occasional retraining

Save the GPU hassle until PyTorch 2.6 stable is released!
