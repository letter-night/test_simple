#!/bin/bash
# Run hyperparameter tuning for pretraining on synthetic data
# Usage: bash runnables/run_pretrain_tuning.sh [gpu_id]
#
# Uses Ray Tune to search over the pretraining HP grid defined in
# cancer_sim_pretrain_grid.yaml. Best HPs should be copied to
# cancer_sim_pretrain_tuned.yaml after the run.

set -e

GPU_ID="${1:-0}"
PYTHON="/home/letter_night/anaconda3/envs/scip_stable/bin/python"
SCRIPT="runnables/train_dynamic_causal_pfn.py"
SEED=101

cd /data1/letternight/test_simple-1
export PYTHONPATH=/data1/letternight/test_simple-1:$PYTHONPATH

echo "=============================================="
echo "Pretraining Hyperparameter Tuning"
echo "GPU: $GPU_ID"
echo "=============================================="

$PYTHON $SCRIPT +backbone=dynamic_causal_pfn \
    +backbone/dynamic_causal_pfn_hparams=cancer_sim_pretrain_grid \
    +dataset=cancer_sim_pretrain dataset.coeff=10.0 exp.seed=$SEED \
    exp.gpus="[$GPU_ID]" exp.logging=False \
    dataset.num_pretrain_datasets_train=5 \
    dataset.num_pretrain_dataset_val=1 \
    2>&1 | tee outputs/pretrain_tuning.log

echo ""
echo "=============================================="
echo "Tuning complete. Check log for best hyperparameters."
echo "Update cancer_sim_pretrain_tuned.yaml with the best values."
echo "=============================================="
