#!/usr/bin/env bash
set -e
cd /home/cason/Github/csp-timing-predictor

echo "=== Round 2 Pipeline: $(date) ==="

echo ""
echo "=== Step 1: Optuna Tuning ==="
python tune_ensemble_v3.py 2>&1 | tee tune_v3_round2.log

echo ""
echo "=== Step 2: Train Ensemble ==="
python train_ensemble_v3.py --use-optuna-params --recency-halflife 2.0 2>&1 | tee train_v3_round2.log

echo ""
echo "=== Step 3: Walk-Forward Validation ==="
python walkforward_validation.py 2>&1 | tee wf_v3_round2.log

echo ""
echo "=== Pipeline complete: $(date) ==="
