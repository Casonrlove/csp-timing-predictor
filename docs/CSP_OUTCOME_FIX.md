# CSP Outcome Fix - Handling Inverse ETFs Correctly

**Date:** 2026-02-09
**Issue:** Model gave same signals for TQQQ (bull) and SQQQ (bear), which is logically wrong
**Fix:** Train on actual CSP trade outcomes instead of simple drawdowns

---

## The Problem

### What We Had Wrong:
```python
# Old target: Just check if stock drops
Good_CSP_Time = (Max_Drawdown_35D > -5%)
```

**Why this fails for inverse ETFs:**
- CSP = Bullish bet (profit if stock goes UP)
- TQQQ = Bull ETF (goes up when market goes up)
- SQQQ = Bear ETF (goes up when market goes DOWN)

**Example of the bug:**
1. Market goes UP
2. SQQQ drops (it's inverse) → RSI oversold, recent drops
3. Model sees "oversold" → signals "GOOD TIME"
4. **But CSP on SQQQ = betting market goes DOWN = WRONG!**

The old model treated each ticker independently using technical indicators, without understanding:
- CSP is a directional (bullish) bet
- Some ETFs move inverse to the market

---

## The Solution

### Train on Actual CSP Trade Outcomes:

```python
# New target: Simulate actual CSP trade
strike_price = entry_price * 0.90  # 10% OTM (typical 0.30 delta)
min_price_35d = future_prices.min()
CSP_Profitable = (min_price_35d >= strike_price)
```

**Why this works for ALL tickers:**

**Normal Stock (NVDA):**
- Entry: $100 → Strike: $90
- Stock stays at $95 → CSP profitable ✓
- Stock drops to $85 → CSP loses ✓

**Bull ETF (TQQQ):**
- Entry: $50 → Strike: $45
- Market up → TQQQ up to $55 → CSP profitable ✓
- Market down → TQQQ down to $40 → CSP loses ✓

**Bear ETF (SQQQ):**
- Entry: $100 → Strike: $90
- Market UP → SQQQ DOWN to $80 → CSP loses ✓ (Correct!)
- Market DOWN → SQQQ UP to $110 → CSP profitable ✓ (Correct!)

**The key:** We're not asking "did the stock drop?" We're asking "would the CSP trade be profitable?" This automatically captures the directional nature of the bet!

---

## Implementation

### Code Changes:

**File:** `data_collector.py`

**New method signature:**
```python
def create_target_variable(
    self,
    forward_days=35,
    strike_otm_pct=0.10,  # 10% OTM = ~0.30 delta
    use_csp_outcome=True  # NEW: Use actual CSP simulation
):
```

**What it does:**
1. For each day, calculates what the strike price would be (10% OTM)
2. Looks forward 35 days (typical holding period)
3. Checks if stock ever breached the strike
4. Labels as "profitable" only if strike was never breached

**New columns created:**
- `CSP_Profitable` - 1 if trade would be profitable, 0 if not
- `Min_Price_35D` - Lowest price during holding period
- `Strike_Breach_Pct` - How far below strike it went (if breached)
- `Good_CSP_Time` - Same as CSP_Profitable (for compatibility)

---

## Expected Results After Retraining

### Before (Wrong):
```
TQQQ: GOOD TIME (CSP Score: +0.45)
SQQQ: GOOD TIME (CSP Score: +0.42)  ← WRONG! Can't both be good
```

### After (Correct):
```
When market is bullish:
TQQQ: GOOD TIME (CSP Score: +0.60)   ← TQQQ goes up
SQQQ: WAIT (CSP Score: -0.35)        ← SQQQ goes down

When market is bearish:
TQQQ: WAIT (CSP Score: -0.40)        ← TQQQ goes down
SQQQ: GOOD TIME (CSP Score: +0.55)   ← SQQQ goes up
```

---

## Testing Plan

### 1. Quick Validation Test:
```python
from data_collector import CSPDataCollector

# Test on SQQQ
collector = CSPDataCollector('SQQQ', period='2y')
collector.fetch_data()
collector.calculate_technical_indicators()
collector.create_target_variable(use_csp_outcome=True)

# Check target distribution
print(collector.data['Good_CSP_Time'].value_counts())
print(f"Profitable CSP rate: {collector.data['Good_CSP_Time'].mean():.1%}")
```

### 2. Retrain Models:
```bash
cd ~/Github/csp-timing-predictor
python3 train_timing_model_per_ticker.py --period 5y
```

### 3. Test Predictions:
```bash
# Test both TQQQ and SQQQ
curl http://localhost:8000/predict -X POST -d '{"ticker": "TQQQ"}' | jq
curl http://localhost:8000/predict -X POST -d '{"ticker": "SQQQ"}' | jq

# They should have OPPOSITE or at least different signals
```

---

## Technical Notes

### Why 10% OTM?
- Typical 0.30 delta CSP is ~10% out-of-the-money
- This is a standard risk level for CSP traders
- Can be adjusted via `strike_otm_pct` parameter

### Why 35 days?
- Most CSPs are sold 30-45 DTE (days to expiration)
- 35 days is the midpoint
- This is the typical holding period

### Backward Compatibility:
- Can still use old method with `use_csp_outcome=False`
- Default is now `True` (new method)
- Old models trained with drawdown method still work

---

## Next Steps

1. ✅ Code updated (data_collector.py)
2. ⬜ Test on TQQQ/SQQQ to verify logic
3. ⬜ Retrain per-ticker models with 5y data
4. ⬜ Deploy and test predictions
5. ⬜ Monitor real-world performance

---

## Files Modified

- `data_collector.py` - Updated `create_target_variable()` method
- `CSP_OUTCOME_FIX.md` - This documentation

## Files to Update Next

- Retrain: `train_timing_model_per_ticker.py`
- May need to update: `simple_api_server.py` (check compatibility)
- May need to update: `validate_timing_model.py` (check compatibility)

---

**This fix ensures the model understands CSP is a bullish bet and handles inverse ETFs correctly!**
