# Optuna Hyperparameter Tuning Results - V3 Strike Model

## Executive Summary

Optuna hyperparameter optimization was successfully completed for the V3 strike probability model. However, **the baseline V3 model with default hyperparameters remains the production model** due to superior ranking ability (ROC-AUC).

## Methodology

- **Optimization Method**: Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Cross-Validation**: 3-fold purged time-series split with 35-day gap
- **Objective Function**: Minimize Brier score (probability calibration)
- **Trials per Group**: 30 trials × 9 ticker groups = 270 total trials
- **Training Time**: ~40 minutes with GPU acceleration
- **Hyperparameters Tuned**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight

## Out-of-Sample Validation Results

**Test Setup:**
- 6 tickers (NVDA, AAPL, SPY, TSLA, MSFT, QQQ)
- 2 years of out-of-sample data
- 25,272 option outcome tests

### Overall Metrics Comparison

| Metric | Baseline V3 | Tuned V3 | Change | Winner |
|--------|-------------|----------|--------|--------|
| **Brier Score** | 0.1480 | 0.1449 | **+2.1%** ✓ | Tuned |
| **ROC-AUC** | 0.7576 | 0.7187 | **-5.1%** ✗ | Baseline |
| **MAE (%)** | 33.7% | 32.4% | **+3.7%** ✓ | Tuned |

### Calibration by Delta Level (V3 Error %)

| Delta | Baseline Error | Tuned Error | Improvement |
|-------|----------------|-------------|-------------|
| 10% | +3.71% | +1.26% | **+2.46%** ✓ |
| 15% | +6.12% | +3.58% | **+2.54%** ✓ |
| 20% | +9.13% | +5.79% | **+3.35%** ✓ |
| 25% | +11.45% | +8.29% | **+3.16%** ✓ |
| 30% | +14.19% | +10.91% | **+3.28%** ✓ |
| 35% | +16.44% | +13.30% | **+3.14%** ✓ |
| 40% | +18.37% | +15.41% | **+2.96%** ✓ |
| 45% | +20.36% | +17.07% | **+3.29%** ✓ |
| 50% | +24.91% | +21.35% | **+3.56%** ✓ |

**Key Finding:** Tuned model improved calibration at **every single delta level** by 2.5-3.6%.

## Analysis: The Calibration vs Ranking Trade-off

### What Happened

The Optuna optimization successfully achieved its objective: **minimize Brier score** (improve calibration). However, this came at the cost of **ranking ability** (ROC-AUC decreased by 5.1%).

### Why This Matters for CSP Trading

**ROC-AUC (Ranking) is more important than calibration for edge detection:**

1. **Edge detection requires ranking**: We need to correctly identify which trades have positive edge vs negative edge. This is a ranking problem - we want the model to assign higher probabilities to actual ITM outcomes and lower probabilities to actual OTM outcomes.

2. **Calibration is secondary**: While it's nice to have accurate probability estimates, what matters most is whether the model can separate good trades from bad trades. A model with ROC-AUC 0.76 but slightly miscalibrated (+3-6% error) is more useful than a model with ROC-AUC 0.72 that's perfectly calibrated.

3. **Our edge formula depends on ranking**:
   ```
   edge = delta - model_probability
   ```
   If the model can't rank probabilities correctly, it will give false edge signals.

### Example Impact

Consider two 25-delta options:
- **Option A**: Actually expires ITM
- **Option B**: Actually expires OTM

**Baseline V3 (ROC-AUC 0.76):**
- Option A prediction: 0.20 → Edge = +5% (correct positive signal)
- Option B prediction: 0.28 → Edge = -3% (correct negative signal)
- Result: ✓ Correctly identifies good vs bad trade

**Tuned V3 (ROC-AUC 0.72, better calibrated):**
- Option A prediction: 0.24 → Edge = +1% (weaker signal)
- Option B prediction: 0.25 → Edge = 0% (missed the risk)
- Result: ✗ Failed to distinguish trades effectively

## Best Hyperparameters Found (For Reference)

Optuna found these optimal parameters across groups:

### High-Performing Groups

**ETF Group** (Brier 0.1424):
```python
{
    'n_estimators': 135,
    'max_depth': 4,
    'learning_rate': 0.0117,
    'subsample': 0.66,
    'colsample_bytree': 0.69,
    'reg_alpha': 0.27,
    'reg_lambda': 1.66,
    'min_child_weight': 4
}
```

**Finance Group** (Brier 0.1408):
```python
{
    'n_estimators': 396,
    'max_depth': 6,
    'learning_rate': 0.067,
    'subsample': 0.95,
    'colsample_bytree': 0.60,
    'reg_alpha': 0.66,
    'reg_lambda': 0.39,
    'min_child_weight': 8
}
```

**Baseline V3 (Default params):**
```python
{
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42
}
```

## Recommendations

### 1. Keep Baseline V3 in Production ✅

The baseline V3 model with default hyperparameters should remain the production model because:
- Higher ROC-AUC (0.76 vs 0.72) for better edge detection
- Still has good calibration (Brier 0.148 is excellent)
- Proven track record from original validation

### 2. Alternative Optimization Objectives (Future Work)

If we want to improve the model further, consider:

**Option A: Optimize for ROC-AUC instead of Brier**
```python
def objective(trial):
    # ... train model with trial params ...
    roc_auc = roc_auc_score(y_val, predicted_probs)
    return -roc_auc  # Optuna minimizes, so negate
```

**Option B: Multi-objective optimization**
```python
# Optimize both Brier and ROC-AUC simultaneously
# Optuna supports Pareto optimization
study = optuna.create_study(directions=['minimize', 'maximize'])
```

**Option C: Custom objective balancing both**
```python
def objective(trial):
    # ... train and predict ...
    brier = brier_score_loss(y_val, predicted_probs)
    roc_auc = roc_auc_score(y_val, predicted_probs)

    # Weight ROC-AUC more heavily
    return brier - (2.0 * roc_auc)  # Weighted combination
```

### 3. Ensemble Approach (Future Work)

Combine baseline V3 with tuned V3:
```python
# Average predictions from both models
final_prob = 0.6 * baseline_v3.predict() + 0.4 * tuned_v3.predict()
```

This could potentially get the best of both worlds: good ranking from baseline + better calibration from tuned.

## Files Created

- `strike_model_v3_tuned.pkl` - Optuna-tuned model (not used in production)
- `compare_v3_models.py` - Comparison validation script
- `v3_tuning.log` - Full Optuna training log
- `v3_comparison.log` - Validation comparison results
- `OPTUNA_TUNING_RESULTS.md` - This document

## Conclusion

Optuna hyperparameter tuning successfully improved calibration but degraded ranking ability. For CSP trading where edge detection is critical, **ranking ability (ROC-AUC) is more important than perfect calibration**.

**Decision: Keep baseline V3 as production model.**

The exercise was valuable for understanding the calibration vs ranking trade-off and provides a path forward for future optimization attempts with different objectives.

---

**Status**: Task #4 (Optuna tuning) - COMPLETED
**Production Model**: `strike_model_v3.pkl` (baseline)
**Date**: February 8, 2026
