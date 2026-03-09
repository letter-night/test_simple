#!/bin/bash
# Tune all baseline models on cancer_sim dataset (coeff=10.0)
# Usage: bash runnables/tune_all_baselines.sh [gpu_id]
#
# Models: CRN, TE-CDE, CT, G-Net, RMSN, SCIP
# (GT already has cancer_sim tuned configs)

set -e

GPU_ID="${1:-0}"
PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SEED=101
COEFF=10.0
RESULTS_DIR="outputs/baseline_tuning"

cd /data1/letternight/test_simple-1
export PYTHONPATH=/data1/letternight/test_simple-1:$PYTHONPATH

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "Baseline Hyperparameter Tuning on cancer_sim"
echo "GPU: $GPU_ID | coeff: $COEFF | seed: $SEED"
echo "=============================================="

# --- 1. CRN ---
echo ""
echo "[1/6] Tuning CRN..."
$PYTHON runnables/train_enc_dec.py \
    +backbone=crn \
    +backbone/crn_hparams=hparams_grid \
    +dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    2>&1 | tee "$RESULTS_DIR/crn_tuning.log"

# --- 2. TE-CDE ---
echo ""
echo "[2/6] Tuning TE-CDE..."
$PYTHON runnables/train_enc_dec.py \
    +backbone=tecde \
    +backbone/tecde_hparams=hparams_grid \
    +dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    dataset.fill_missing=False \
    2>&1 | tee "$RESULTS_DIR/tecde_tuning.log"

# --- 3. CT ---
echo ""
echo "[3/6] Tuning CT..."
$PYTHON runnables/train_multi.py \
    +backbone=ct \
    +backbone/ct_hparams=hparams_grid \
    +dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    2>&1 | tee "$RESULTS_DIR/ct_tuning.log"

# --- 4. G-Net ---
echo ""
echo "[4/6] Tuning G-Net..."
$PYTHON runnables/train_gnet.py \
    +backbone=gnet \
    +backbone/gnet_hparams=hparams_grid \
    +dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    2>&1 | tee "$RESULTS_DIR/gnet_tuning.log"

# --- 5. RMSN ---
echo ""
echo "[5/6] Tuning RMSN..."
$PYTHON runnables/train_rmsn.py \
    +backbone=rmsn \
    +backbone/rmsn_hparams=hparams_grid \
    +dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    dataset.treatment_mode=multilabel \
    2>&1 | tee "$RESULTS_DIR/rmsn_tuning.log"

# --- 6. SCIP ---
echo ""
echo "[6/6] Tuning SCIP..."
$PYTHON runnables/train_scip.py \
    +backbone=scip \
    +backbone/scip_hparams=hparams_grid \
    +dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    dataset.treatment_mode=multilabel \
    dataset.fill_missing=False \
    2>&1 | tee "$RESULTS_DIR/scip_tuning.log"

echo ""
echo "=============================================="
echo "All baseline tuning complete."
echo "Results in: $RESULTS_DIR/"
echo "=============================================="

# --- Summary ---
echo ""
echo "=== TUNING RESULTS SUMMARY ==="
for logfile in "$RESULTS_DIR"/*_tuning.log; do
    name=$(basename "$logfile" _tuning.log)
    echo "--- $name ---"
    grep -E "best values|Best trial|Current best" "$logfile" | tail -3 || echo "  (no best trial found)"
    grep -E "Val normalised RMSE" "$logfile" | tail -1 || echo "  (no val RMSE found)"
    grep -E "Test normalised RMSE" "$logfile" | head -1 || echo "  (no test RMSE found)"
done
