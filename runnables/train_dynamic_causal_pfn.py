import logging
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.models.utils import FilteringMlFlowLogger
from src.models.dynamic_causal_pfn import DynamicCausalPFN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for DynamicCausalPFN (GT backbone)
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False) # turn of strict mode
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True) # custom resolver: add interpolation expression for within config files
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True)) # convert config object to yaml-formatted string. Resolve interpolation expressions. Log yaml

    # ============================== Data init ===========================================
    seed_everything(args.exp.seed) # global seed
    dataset_collection = instantiate(args.dataset, _recursive_=True) # Instantiate dataset dynamically from dataset config

    # Choose processing method based on dataset type
    from src.data.cancer_sim.dataset import PretrainCancerDatasetCollection
    if isinstance(dataset_collection, PretrainCancerDatasetCollection):
        dataset_collection.process_data_pretrain()
    else:
        dataset_collection.process_data_multi()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # ============================== Callbacks & Logger =================================
    pfn_callbacks = []

    # Checkpoint callback — saves best model by validation loss
    ckpt_dir = os.path.join(os.getcwd(), 'checkpoints')
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='pretrained-{epoch:02d}-{dynamic_causal_pfn_train_loss:.4f}',
        monitor='dynamic_causal_pfn_train_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    pfn_callbacks.append(ckpt_callback)

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri, run_name='0') # exclude submodels from logging
        pfn_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None

    # ============================== 1-step ahead prediction ===========================================
    pretrained_ckpt = args.model.dynamic_causal_pfn.get('pretrained_ckpt', None)
    transfer_mode = args.model.dynamic_causal_pfn.get('transfer_mode', None)

    args.model.dynamic_causal_pfn.projection_horizon = 0

    # --- Model initialisation ---
    if pretrained_ckpt is not None:
        # Load pretrained checkpoint for transfer
        pretrained_ckpt = hydra.utils.to_absolute_path(pretrained_ckpt)
        logger.info(f'Loading pretrained checkpoint from: {pretrained_ckpt}')

        # PL 1.4.5 checkpoints contain non-tensor objects; allow them with weights_only=False
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)

        # Instantiate model with current args/dataset, then load pretrained weights
        pfn_model = instantiate(args.model.dynamic_causal_pfn, args, dataset_collection, _recursive_=False)
        pfn_model.load_state_dict(checkpoint['state_dict'])
        logger.info('Pretrained checkpoint loaded successfully.')
    else:
        pfn_model = instantiate(args.model.dynamic_causal_pfn, args, dataset_collection, _recursive_=False)
        if args.model.dynamic_causal_pfn.tune_hparams:
            pfn_model.finetune(resources_per_trial=args.model.dynamic_causal_pfn.resources_per_trial)

    # --- Transfer mode handling ---
    if transfer_mode == 'zero_shot':
        logger.info('Transfer mode: ZERO-SHOT — skipping training, evaluating pretrained model directly.')
        # Still need a trainer for predict() calls in evaluation
        pfn_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=0,
                              callbacks=pfn_callbacks, terminate_on_nan=True)
        # Attach trainer to model without training
        pfn_trainer.fit(pfn_model)

    elif transfer_mode == 'few_shot':
        few_shot_n = int(args.model.dynamic_causal_pfn.get('few_shot_samples', 50))
        n_total = len(dataset_collection.train_f)
        few_shot_n = min(few_shot_n, n_total)
        logger.info(f'Transfer mode: FEW-SHOT — fine-tuning on {few_shot_n}/{n_total} training samples.')

        # Create a subset of the training dataset
        import numpy as np
        indices = np.random.RandomState(args.exp.seed).choice(n_total, few_shot_n, replace=False).tolist()
        few_shot_dataset = Subset(dataset_collection.train_f, indices)

        # Override train_dataloader to use the subset
        few_shot_batch_size = min(args.model.dynamic_causal_pfn.batch_size, few_shot_n)
        pfn_model.train_dataloader = lambda: DataLoader(
            few_shot_dataset, shuffle=True, batch_size=few_shot_batch_size, drop_last=True
        )

        pfn_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                              callbacks=pfn_callbacks, terminate_on_nan=True,
                              gradient_clip_val=args.model.dynamic_causal_pfn.max_grad_norm)
        pfn_trainer.fit(pfn_model)

    else:
        # fine_tune or train from scratch — full training on all data
        if transfer_mode == 'fine_tune':
            logger.info('Transfer mode: FINE-TUNE — training pretrained model on full downstream data.')
        else:
            logger.info('Training from scratch on full data.')

        pfn_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                              callbacks=pfn_callbacks, terminate_on_nan=True,
                              gradient_clip_val=args.model.dynamic_causal_pfn.max_grad_norm)
        pfn_trainer.fit(pfn_model)

    # Log checkpoint path
    if ckpt_callback.best_model_path:
        logger.info(f'Best checkpoint: {ckpt_callback.best_model_path}')
        logger.info(f'Last checkpoint: {ckpt_callback.last_model_path}')

    # Validation factual rmse
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    pfn_trainer.test(pfn_model, test_dataloaders=val_dataloader)
  
    val_rmse_orig, val_rmse_all = pfn_model.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step') and dataset_collection.test_cf_one_step is not None:
        test_rmse_orig, test_rmse_all, test_rmse_last = pfn_model.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step,
            one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f') and dataset_collection.test_f is not None:
        test_rmse_orig, test_rmse_all = pfn_model.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }
    else:
        logger.info('No downstream test set available — skipping test evaluation.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
        }

    if args.exp.logging:
        mlf_logger.log_metrics(encoder_results)
    results.update(encoder_results)

    # ============================== multi-step ahead prediction ===========================================
    has_downstream_test = (
        (hasattr(dataset_collection, 'test_cf_treatment_seq') and dataset_collection.test_cf_treatment_seq is not None)
        or (hasattr(dataset_collection, 'test_f_multi') and dataset_collection.test_f_multi is not None)
    )

    for t in range(1, args.dataset.projection_horizon+1) if has_downstream_test else []:
        seed_everything(args.exp.seed)  # global seed -> if training breaks for some reason, start again with same seed
        test_rmses = {}
        decoder_results = {
            'decoder_val_rmse_all': val_rmse_all,
            'decoder_val_rmse_orig': val_rmse_orig
        }
        if args.exp.logging:
            mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
                                               tracking_uri=args.exp.mlflow_uri, run_name=str(t))
        # ============================== Train ===========================================
        args.model.dynamic_causal_pfn.projection_horizon = t
        pfn_model = instantiate(args.model.dynamic_causal_pfn, args, dataset_collection, _recursive_=False)

        # Load pretrained checkpoint for decoder if available
        if pretrained_ckpt is not None:
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
            pfn_model.load_state_dict(checkpoint['state_dict'], strict=False)

        if transfer_mode == 'zero_shot':
            pfn_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=0,
                                      callbacks=pfn_callbacks, terminate_on_nan=True)
            pfn_trainer.fit(pfn_model)
        elif transfer_mode == 'few_shot':
            few_shot_n = int(args.model.dynamic_causal_pfn.get('few_shot_samples', 50))
            n_total = len(dataset_collection.train_f)
            few_shot_n = min(few_shot_n, n_total)
            import numpy as np
            indices = np.random.RandomState(args.exp.seed).choice(n_total, few_shot_n, replace=False).tolist()
            few_shot_dataset = Subset(dataset_collection.train_f, indices)
            few_shot_batch_size = min(args.model.dynamic_causal_pfn.batch_size, few_shot_n)
            pfn_model.train_dataloader = lambda: DataLoader(
                few_shot_dataset, shuffle=True, batch_size=few_shot_batch_size, drop_last=True
            )
            pfn_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                      callbacks=pfn_callbacks, terminate_on_nan=True,
                                      gradient_clip_val=args.model.dynamic_causal_pfn.max_grad_norm)
            pfn_trainer.fit(pfn_model)
        else:
            pfn_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                      callbacks=pfn_callbacks, terminate_on_nan=True,
                                      gradient_clip_val=args.model.dynamic_causal_pfn.max_grad_norm)
            pfn_trainer.fit(pfn_model)

        # ============================== Test ===========================================
        if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
            test_rmse = pfn_model.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
        elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
            test_rmse = pfn_model.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
        test_rmses = {f'{t+1}-step': test_rmse}
        logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')

        decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

        mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
        results.update(decoder_results)

    if args.exp.logging:
        mlf_logger.experiment.set_terminated(mlf_logger.run_id)

    return results


if __name__ == "__main__":
    main()