# GPU-Accelerated CSP Timing Models

Advanced deep learning models for predicting optimal Cash Secured Put (CSP) timing using your 5070Ti GPU.

## Models

### 1. LSTM (Long Short-Term Memory)
- **Best for**: Capturing temporal patterns in price movements
- **Architecture**: 2-layer LSTM with attention mechanism
- **Strengths**: Remembers long-term price patterns, identifies momentum shifts
- **Parameters**: 128 hidden units, 20-day sequence length

### 2. Transformer
- **Best for**: Parallel processing of time series with self-attention
- **Architecture**: 3-layer transformer encoder with positional encoding
- **Strengths**: Captures complex relationships between features across time
- **Parameters**: 128 d_model, 8 attention heads

### 3. Hybrid (LSTM + Transformer)
- **Best for**: Combining strengths of both architectures
- **Architecture**: LSTM + Transformer branches with fusion layer
- **Strengths**: Best overall performance, captures both sequential and attention patterns
- **Recommended**: Start with this model

### 4. TabNet
- **Best for**: Tabular financial data with feature importance
- **Architecture**: Attention-based sequential decision steps
- **Strengths**: Interpretable, shows which features matter most, handles sparse data
- **Special**: Can train on multiple tickers simultaneously for better generalization

## Installation

```bash
pip install -r requirements_gpu.txt
```

This installs PyTorch with CUDA support for your 5070Ti.

## Quick Start

### Train All Models

```bash
# Train deep learning models (LSTM, Transformer, Hybrid)
python deep_learning_model.py

# Train TabNet (single ticker)
python tabnet_trainer.py
```

### Make Predictions

```bash
# Using Hybrid model (recommended)
python deep_predictor.py csp_hybrid_model.pkl NVDA

# Using TabNet
python deep_predictor.py csp_tabnet_nvda.pkl NVDA
```

## Advanced Usage

### Multi-Ticker Training (TabNet Only)

Train on multiple tickers for better generalization:

```python
from tabnet_trainer import TabNetTrainer

trainer = TabNetTrainer()
X, y = trainer.load_multi_ticker_data(['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT'], period='5y')
X_train, y_train, X_test, y_test = trainer.prepare_data(X, y)
trainer.train(X_train, y_train, max_epochs=200)
trainer.save_model('csp_tabnet_multi.pkl')
```

This creates a model that works well across different stocks.

### Custom Model Training

```python
from deep_learning_model import DeepLearningTrainer

# Train with custom parameters
trainer = DeepLearningTrainer(model_type='hybrid', seq_length=30)
X, y, df = trainer.load_data('NVDA', period='5y')
(X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.prepare_data(X, y)

trainer.train(
    X_train, y_train, X_val, y_val,
    epochs=150,
    batch_size=64,
    lr=0.001
)

trainer.evaluate(X_test, y_test)
trainer.save_model('my_custom_model.pkl')
```

### Hyperparameter Tuning

For best results, tune:
- **seq_length**: 10-30 days (how far back to look)
- **hidden_size**: 64-256 (model capacity)
- **learning_rate**: 0.0001-0.01 (training speed)
- **batch_size**: 32-128 (GPU utilization)
- **dropout**: 0.2-0.5 (regularization)

## Performance Expectations

With GPU acceleration:
- **Training Time**: 5-15 minutes per model
- **Inference**: <100ms per prediction
- **Memory Usage**: ~2-4GB VRAM

Expected improvements over sklearn models:
- **ROC-AUC**: 0.53 → 0.65-0.75 (target)
- **Recall**: Better at catching good opportunities
- **Precision**: Fewer false signals

## Model Comparison

Compare multiple models on same ticker:

```python
from deep_predictor import DeepLearningPredictor

predictor = DeepLearningPredictor('csp_hybrid_model.pkl')
results = predictor.compare_models_prediction('NVDA', [
    'csp_lstm_model.pkl',
    'csp_transformer_model.pkl',
    'csp_hybrid_model.pkl',
    'csp_tabnet_nvda.pkl'
])
```

## Improving Performance Further

### 1. More Data
```python
# Train on 10 years instead of 5
trainer.load_data('NVDA', period='10y')
```

### 2. Data Augmentation
- Add noise to features
- Bootstrap sampling
- Synthetic minority oversampling (SMOTE)

### 3. Feature Engineering
- Add more technical indicators
- Include options data (IV rank, skew)
- Market regime indicators

### 4. Ensemble Methods
- Train multiple models with different seeds
- Average their predictions
- Voting mechanism

### 5. Adjust Target Variable

Make the task easier (more training samples):

```python
# In data_collector.py
def create_target_variable(self, forward_days=30, threshold_pct=-5):
    # Changed from 35 days/-3% to 30 days/-5%
    # This is more forgiving and gives more positive examples
```

## GPU Optimization Tips

### Maximize GPU Usage
- Increase batch_size until you hit memory limit
- Use larger models (hidden_size=256)
- Train multiple models in parallel

### Monitor GPU
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check if PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Mixed Precision Training (Faster)
```python
# In deep_learning_model.py, add to training loop:
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size
- Reduce seq_length
- Reduce hidden_size
- Clear cache: `torch.cuda.empty_cache()`

### Poor Performance
- Train longer (more epochs)
- Adjust threshold_pct to get more positive samples
- Train on multiple tickers
- Add more features
- Use ensemble predictions

### Overfitting
- Increase dropout
- Add L2 regularization (weight_decay)
- Use early stopping (already implemented)
- Train on more data

## Files Generated

- `csp_lstm_model.pkl` - LSTM model
- `csp_transformer_model.pkl` - Transformer model
- `csp_hybrid_model.pkl` - Hybrid model (recommended)
- `csp_tabnet_nvda.pkl` - TabNet single ticker
- `csp_tabnet_multi.pkl` - TabNet multi-ticker
- `*_training_history.png` - Training curves
- `tabnet_feature_importance.png` - Feature importance plot
- `best_*_model.pth` - PyTorch model weights

## Next Steps

1. **Train models**: Run `python deep_learning_model.py`
2. **Compare results**: Check training history plots and test metrics
3. **Make predictions**: Use best model with `deep_predictor.py`
4. **Iterate**: Adjust hyperparameters and retrain
5. **Deploy**: Integrate into your trading workflow

## Recommended Workflow

For best results:
1. Train TabNet with multi-ticker data (best generalization)
2. Train Hybrid model on NVDA specifically (best for single stock)
3. Use ensemble of both for final predictions
4. Monitor performance over time and retrain monthly

## Support

For issues or questions:
- Check GPU availability: `nvidia-smi`
- Verify CUDA version: `torch.version.cuda`
- Reduce complexity if training fails
- Use smaller seq_length or batch_size
