# DynamicCausalPFN - Project Guide

## Project Overview
DynamicCausalPFN is a foundation-model approach for time-varying treatment effect estimation.
The core idea: pretrain on large-scale synthetic data, then transfer to the tumour growth dataset
via zero-shot, few-shot, and fine-tuning — comparing against baselines trained directly on tumour growth.

## Current Status
- **Completed:** Phase 1 (checkpoint save/reload), Phase 2 (transfer evaluation), Phase 3 setup (tuning scripts/configs)
- **In progress:** Phase 3 execution (pretraining HP tuning → full pretrain → transfer sweep)

## Architecture
- **Framework:** PyTorch Lightning 1.4.5, Hydra configs, MLflow logging
- **Base class:** `src/models/time_varying_model.py` → `TimeVaryingCausalModel`
- **Main model:** `src/models/dynamic_causal_pfn.py` → `DynamicCausalPFN` (transformer-based, G-computation heads)
- **Training script:** `runnables/train_dynamic_causal_pfn.py`

## Key Data Flow
1. **Pretraining data:** `cancer_pretrain.py` generates synthetic datasets with random DAG mechanisms
   - `PretrainCancerDataset` / `PretrainCancerDatasetCollection` in `src/data/cancer_sim/dataset.py`
   - Config: `config/dataset/cancer_sim.yaml` (currently points to `PretrainCancerDatasetCollection`)
2. **Downstream data:** `cancer_simple.py` generates tumour growth simulation
   - `SyntheticCancerDataset` / `SyntheticCancerDatasetCollection` in `src/data/cancer_sim/dataset.py`
3. **Processing:** `dataset_collection.py` has `process_data_pretrain()` for pretraining, `process_data_multi()` for downstream

## Data Format (Unified)
All datasets produce: `prev_treatments`, `current_treatments`, `prev_outputs`, `static_features`,
`outputs`, `active_entries`, `sequence_lengths`

## Config Structure
- `config/config.yaml` — global defaults (seed, gpus, epochs, mlflow)
- `config/backbone/dynamic_causal_pfn.yaml` — model architecture config
- `config/backbone/dynamic_causal_pfn_hparams/` — hyperparameter presets
- `config/dataset/cancer_sim.yaml` — downstream tumour growth dataset config
- `config/dataset/cancer_sim_pretrain.yaml` — pretraining synthetic dataset config

## Running Experiments
```bash
PYTHONPATH=. python3 runnables/train_dynamic_causal_pfn.py \
  +dataset=cancer_sim +backbone=dynamic_causal_pfn \
  +backbone/dynamic_causal_pfn_hparams='cancer_sim_tuned' \
  dataset.coeff=15.0 exp.seed=101
```

## Important Implementation Details
- `projection_horizon=0`: 1-step-ahead factual prediction (encoder training)
- `projection_horizon>0`: multi-step G-computation with pseudo-outcomes
- Treatment is binary (radiotherapy on/off), encoded as one-hot `[1,0]`/`[0,1]` in multiclass mode
- Model uses `torch.double` precision by default
- Pretraining uses `PretrainCancerDatasetCollection`; downstream uses `SyntheticCancerDatasetCollection`

## Remaining Work (Phases)
### Phase 1: Checkpoint Save & Reload (DONE)
- ModelCheckpoint callback saves best/last checkpoints during pretraining
- Checkpoint loading via `torch.load(path, weights_only=False)` + `load_state_dict`

### Phase 2: Transfer Evaluation on Tumour Growth (DONE)
- Zero-shot, few-shot, fine-tune modes implemented via `transfer_mode` config
- Pretrained checkpoint: `outputs/2026-03-07/23-09-47/checkpoints/last.ckpt`

### Phase 3: Hyperparameter Tuning (IN PROGRESS)
**Scripts:**
- `runnables/run_pretrain_tuning.sh` — Ray Tune search over pretraining HPs
- `runnables/run_pretrain.sh` — Full pretrain with tuned HPs
- `runnables/sweep_transfer.sh` — Sweep zero-shot/few-shot/fine-tune settings

**Configs:**
- `cancer_sim_pretrain_grid.yaml` — Pretraining HP search space
- `cancer_sim_pretrain_tuned.yaml` — Best pretraining HPs (fill after tuning)
- `cancer_sim_transfer_grid.yaml` — Transfer sweep settings reference

**Workflow:**
1. Run pretraining HP tuning → update `cancer_sim_pretrain_tuned.yaml`
2. Run full pretrain with best HPs → get checkpoint
3. Run transfer sweep with checkpoint → find best transfer settings

## Conventions
- Use Hydra for all config management (no hardcoded hyperparameters)
- Log all metrics to MLflow
- Use `torch.double` dtype throughout
- Seed everything via `exp.seed` for reproducibility
