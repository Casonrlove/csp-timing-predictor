#!/bin/bash
set -e
cd /home/cason/Github/csp-timing-predictor

echo "=== ROUND 3 PIPELINE ==="
echo "Start: $(date)"

echo ""
echo "=== Step 1: Optuna Tuning ==="
python3 tune_ensemble_v3.py 2>&1 | tee tune_v3_round3.log

echo ""
echo "=== Step 2: Train Ensemble ==="
python3 train_ensemble_v3.py --use-optuna-params --recency-halflife 2.0 2>&1 | tee train_v3_round3.log

echo ""
echo "=== Step 3: Walk-Forward Validation ==="
python3 walkforward_validation.py 2>&1 | tee wf_v3_round3.log

echo ""
echo "=== DONE ==="
echo "End: $(date)"
