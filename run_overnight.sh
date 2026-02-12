#!/bin/bash
set -e
cd /home/cason/Github/csp-timing-predictor

echo "============================================================"
echo "OVERNIGHT TRAINING PIPELINE — $(date)"
echo "============================================================"

# Phase 1: Train regime-split V3 model
echo ""
echo "=== PHASE 1: Regime-Split V3 Model ==="
echo "Started: $(date)"
START=$SECONDS
python train_timing_model_v3.py --no-gpu --period 10y
echo "Phase 1 done in $((SECONDS - START))s at $(date)"

# Phase 3: Optuna tuning (overnight)
echo ""
echo "=== PHASE 3: Optuna Hyperparameter Tuning ==="
echo "Started: $(date)"
START=$SECONDS
python tune_ensemble_v3.py --period 10y --trials 200 --timeout 3600
echo "Phase 3 done in $((SECONDS - START))s at $(date)"

# Phase 2: Train ensemble with Optuna params
echo ""
echo "=== PHASE 2: Train Ensemble with Optuna Params ==="
echo "Started: $(date)"
START=$SECONDS
python train_ensemble_v3.py --no-gpu --period 10y --use-optuna-params
echo "Phase 2 done in $((SECONDS - START))s at $(date)"

# Validation
echo ""
echo "=== VALIDATION: Walk-Forward ==="
echo "Started: $(date)"
START=$SECONDS
python walkforward_validation.py --group etf 2>&1 | tee /tmp/wf_etf.txt
python walkforward_validation.py 2>&1 | tee /tmp/wf_full.txt
echo "Validation done in $((SECONDS - START))s at $(date)"

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "Results in: /tmp/wf_full.txt"
echo "============================================================"
