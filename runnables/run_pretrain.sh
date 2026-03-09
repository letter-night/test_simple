#!/bin/bash
# Run full pretraining with tuned hyperparameters
# Usage: bash runnables/run_pretrain.sh [gpu_id] [num_train_datasets] [max_epochs]
#
# After tuning, use this script to run a full pretraining run with
# the best hyperparameters. The checkpoint will be saved for transfer.

set -e

GPU_ID="${1:-0}"
N_TRAIN="${2:-10}"
MAX_EPOCHS="${3:-100}"
PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEED=101

cd /data1/letternight/test_simple-1
export PYTHONPATH=/data1/letternight/test_simple-1:$PYTHONPATH

echo "=============================================="
echo "Full Pretraining Run"
echo "GPU: $GPU_ID | Datasets: $N_TRAIN | Epochs: $MAX_EPOCHS"
echo "=============================================="

$PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_tuned \
    +dataset=cancer_sim_pretrain dataset.coeff=10.0 exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    exp.max_epochs=$MAX_EPOCHS \
    dataset.num_pretrain_datasets_train=$N_TRAIN \
    dataset.num_pretrain_dataset_val=2 \
    2>&1 | tee outputs/pretrain_full.log

echo ""
echo "Pretraining complete. Checkpoint saved in outputs/ directory."
echo "Use the checkpoint path with sweep_transfer.sh for transfer experiments."
