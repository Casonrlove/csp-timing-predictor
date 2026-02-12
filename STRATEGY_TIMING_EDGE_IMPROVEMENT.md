# Strategy: Improving Timing and Edge Detection

Biggest gains now will come from problem framing and validation discipline, not extra model complexity.

## Recommended Direction (Ordered by Impact)

1. Reframe to contract-level prediction as the core task
- Predict `P(strike breach within holding window | ticker, date, strike, DTE)` directly.
- Derive edge versus market-implied risk from this probability.
- Treat timing as a prior, not the final decision.

2. Keep one target definition everywhere
- Use the same outcome definition for timing, strike, validation, and settlement.
- Avoid mixing terminal-return targets with path-breach targets.

3. Use timing as a regime prior, not a hard gate
- Let timing adjust aggressiveness (position size / required EV).
- Let edge model rank contracts.

4. Optimize policy utility, not just classifier metrics
- Primary objective: policy-level walk-forward metrics (`EV/trade`, drawdown, hit rate, tail-loss behavior).
- Use AUC and Brier as secondary diagnostics.

5. Make calibration a first-class pipeline
- Run rolling recalibration on settled outcomes.
- Version calibrators and monitor drift over time.
- Edge quality depends on calibrated probabilities.

6. Add explicit uncertainty abstention
- No-trade zone when timing confidence is low.
- No-trade zone near earnings risk.
- No-trade zone when estimated edge is weak/marginal.

7. Run a proper experiment grid
- Compare a small set of clear variants:
  - Baseline delta-only
  - Residual edge model
  - Residual + timing prior
  - Residual + timing + uncertainty band
- Select by out-of-sample utility, not in-sample lift.

8. Prioritize data loop quality next
- Current blocker is settled outcomes coverage.
- Until this loop is healthy, calibration and EV ranking remain unstable.
