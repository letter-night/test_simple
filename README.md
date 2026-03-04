DynamicCausalPFN
==============================

DynamicCausalPFN for conditional average potential outcome estimation over time.

### Setup
Please set up a virtual environment and install the libraries as given in the requirements file.
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow
To start an experiments server, run: 

`mlflow server --port=3335`

Then, one can go to the local browser <http://localhost:3335>.

## Experiments

The main training script `config/config.yaml` is run automatically for all models and datasets.
___
The training `<script>` for each different models specified by:

**CRN**: `runnables/train_enc_dec.py`

**TE-CDE**: `runnables/train_enc_dec.py`

**CT**: `runnables/train_multi.py`

**RMSNs**: `runnables/train_rmsn.py`

**G-Net**: `runnables/train_gnet.py`

**GT**: `runnables/train_gtransformer.py`

**SCIP-Net**: `runnables/train_scip.py`

**DynamicCausalPFN**: `runnables/train_dynamic_causal_pfn.py`
___

The `<backbone>` is specified by:

**CRN**: `crn`

**TE-CDE**: `tecde`

**CT**: `ct`

**RMSNs**: `rmsn`

**G-Net**: `gnet`

**GT**: `gt`

**SCIP-Net**: `scip`

**DynamicCausalPFN**: `dynamic_causal_pfn`
___

The `<hyperparameter>` configuration for each model is specified by:

**CRN**: `backbone/crn_hparams='HPARAMS'`

**TE-CDE**: `backbone/tecde_hparams='HPARAMS'`

**CT**: `backbone/ct_hparams='HPARAMS'`

**RMSNs**: `backbone/rmsn_hparams='HPARAMS'`

**G-Net**: `backbone/gnet_hparams='HPARAMS'`

**GT**: `backbone/gt_hparams='HPARAMS'`

**SCIP-Net**: `backbone/scip_hparams='HPARAMS'`

**DynamicCausalPFN**: `backbone/dynamic_causal_pfn_hparams='HPARAMS'`


`HPARAMS` is either one of:
`cancer_sim_hparams_grid.yaml` / `cancer_sim_tuned.yaml`,

for tuning the hyperparameters / reproducing our results on tuned hyperparameters for synthetic data.

___

The `<dataset>` is specified by:

**Synthetic**: `cancer_sim`

___

Please use the following commands to run the experiments. 
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 <script> +dataset=<dataset> +backbone=<backbone> +<hyperparameter> exp.seed=<seed> exp.logging=True 
```

## Example usage
To run our DynamicCausalPFN with optimized hyperparameters on synthetic data with random seeds 101--105 and confounding level 15.0, use the command:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_dynamic_causal_pfn.py --multirun +dataset=cancer_sim +backbone=dynamic_causal_pfn +backbone/dynamic_causal_pfn_hparams='cancer_sim_tuned' dataset.coeff=15.0 exp.seed=101,102,103,104,105
```

___

