#!/bin/bash
# Sweep script for transfer learning experiments
# Usage: bash runnables/sweep_transfer.sh <pretrained_ckpt_path> [gpu_id]
#
# Runs zero-shot, few-shot (varying sample sizes), and fine-tune (varying LR)
# experiments on the downstream tumour growth dataset.

set -e

CKPT_PATH="${1:?Usage: bash runnables/sweep_transfer.sh <pretrained_ckpt_path> [gpu_id]}"
GPU_ID="${2:-0}"
PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEED=101
COEFF=10.0
RESULTS_DIR="outputs/transfer_sweep"

cd /data1/letternight/test_simple-1
export PYTHONPATH=/data1/letternight/test_simple-1:$PYTHONPATH

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "Transfer Learning Sweep"
echo "Checkpoint: $CKPT_PATH"
echo "GPU: $GPU_ID"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="

# --- Zero-shot ---
echo ""
echo "[1/N] Zero-shot evaluation..."
$PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_tuned \
    dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    'model.dynamic_causal_pfn.pretrained_ckpt='"$CKPT_PATH" \
    model.dynamic_causal_pfn.transfer_mode=zero_shot \
    hydra.run.dir="$RESULTS_DIR/zero_shot" \
    2>&1 | tee "$RESULTS_DIR/zero_shot.log"

# --- Few-shot with varying sample sizes ---
for N_SAMPLES in 10 25 50 100 200; do
    echo ""
    echo "[Few-shot] n=$N_SAMPLES samples..."
    $PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
        +backbone/dynamic_causal_pfn_hparams=cancer_sim_tuned \
        dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
        exp.gpus="[$GPU_ID]" exp.logging=False exp.max_epochs=25 \
        'model.dynamic_causal_pfn.pretrained_ckpt='"$CKPT_PATH" \
        model.dynamic_causal_pfn.transfer_mode=few_shot \
        model.dynamic_causal_pfn.few_shot_samples=$N_SAMPLES \
        hydra.run.dir="$RESULTS_DIR/few_shot_n${N_SAMPLES}" \
        2>&1 | tee "$RESULTS_DIR/few_shot_n${N_SAMPLES}.log"
done

# --- Fine-tune with varying learning rates ---
for LR in 0.01 0.001 0.0001 0.00001; do
    echo ""
    echo "[Fine-tune] lr=$LR..."
    $PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
        +backbone/dynamic_causal_pfn_hparams=cancer_sim_tuned \
        dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
        exp.gpus="[$GPU_ID]" exp.logging=False exp.max_epochs=25 \
        'model.dynamic_causal_pfn.pretrained_ckpt='"$CKPT_PATH" \
        model.dynamic_causal_pfn.transfer_mode=fine_tune \
        model.dynamic_causal_pfn.optimizer.learning_rate=$LR \
        hydra.run.dir="$RESULTS_DIR/fine_tune_lr${LR}" \
        2>&1 | tee "$RESULTS_DIR/fine_tune_lr${LR}.log"
done

# --- Fine-tune with varying epochs (at best LR from above, default 0.001) ---
for EPOCHS in 10 25 50 100; do
    echo ""
    echo "[Fine-tune] epochs=$EPOCHS (lr=0.001)..."
    $PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
        +backbone/dynamic_causal_pfn_hparams=cancer_sim_tuned \
        dataset=cancer_sim dataset.coeff=$COEFF exp.seed=$SEED \
        exp.gpus="[$GPU_ID]" exp.logging=False exp.max_epochs=$EPOCHS \
        'model.dynamic_causal_pfn.pretrained_ckpt='"$CKPT_PATH" \
        model.dynamic_causal_pfn.transfer_mode=fine_tune \
        model.dynamic_causal_pfn.optimizer.learning_rate=0.001 \
        hydra.run.dir="$RESULTS_DIR/fine_tune_ep${EPOCHS}" \
        2>&1 | tee "$RESULTS_DIR/fine_tune_ep${EPOCHS}.log"
done

echo ""
echo "=============================================="
echo "Sweep complete. Results in: $RESULTS_DIR/"
echo "=============================================="

# --- Summary ---
echo ""
echo "=== RESULTS SUMMARY ==="
for logfile in "$RESULTS_DIR"/*.log; do
    name=$(basename "$logfile" .log)
    echo "--- $name ---"
    grep -E "(Val normalised RMSE|Test normalised RMSE)" "$logfile" || echo "  (no RMSE found)"
done
