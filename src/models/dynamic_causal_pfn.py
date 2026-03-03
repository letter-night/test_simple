import logging
from typing import Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue

from src.data import SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel
from src.models.utils import OutcomeHead
from src.models.utils_pfn import (
    DataEmbedding_FeaturePatching,
    Encoder,
    EncoderLayer,
    AttentionLayer,
    FullAttention,
)
from src.models.utils_transformer import AbsolutePositionalEncoding

logger = logging.getLogger(__name__)

class FixedBoolMask:
    def __init__(self, mask: torch.Tensor):
        self._mask = mask 
    
    @property
    def mask(self):
        return self._mask 
    



class DynamicCausalPFN(TimeVaryingCausalModel):
    """
    Baselines-repo compatible DynamicCausalPFN:
    - GT-style single model (model_type is a single key)
    - GT-style G-computation training logic (pseudo outcomes) when projection_horizon > 0
    - Standard masked MSE loss (like other baselines)

    IMPORTANT:
      For cancer_sim, processed sequences have length (dataset.max_seq_length - 1) due to offset=1 in process_data().
      We therefore infer seq_len directly from the processed dataset tensors to avoid time-dimension mismatches.
    """

    model_type = "dynamic_causal_pfn"
    possible_model_types = {"dynamic_causal_pfn"}
    tuning_criterion = "rmse"

    def __init__(
        self,
        args: DictConfig,
        dataset_collection: Union[SyntheticDatasetCollection] = None,
        autoregressive: bool = None,
        has_vitals: bool = None,
        projection_horizon: int = None,
        bce_weights: np.array = None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        sub_args = args.model[self.model_type]

        # --- projection horizon logic (same as GT) ---
        if projection_horizon is not None:
            self.projection_horizon = projection_horizon
        elif sub_args.projection_horizon is not None:
            self.projection_horizon = sub_args.projection_horizon
        elif self.dataset_collection is not None:
            self.projection_horizon = args.dataset.projection_horizon
        else:
            raise MissingMandatoryValue()

        # Used by GT g-computation under fixed treatment sequence intervention
        self.treatment_sequence = torch.tensor(args.dataset.treatment_sequence)[: self.projection_horizon + 1, :]
        self.max_projection = (
            args.dataset.projection_horizon if self.dataset_collection is not None else self.projection_horizon
        )
        assert self.projection_horizon <= self.max_projection

        self.seq_len = self._infer_seq_len(args)
        logger.info(f"[{self.model_type}] Using seq_len={self.seq_len}")

        # --- repo tuning convention (same as GT) ---
        self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, self.dim_outcome)
        logger.info(f"Max input size of {self.model_type}: {self.input_size}")
        logger.info(f"[{self.model_type}] Using seq_len={self.seq_len} (processed time length).")

        self._init_specific(args)
        self.save_hyperparameters(args)

    
    def _infer_seq_len(self, args: DictConfig) -> int:
        lengths = []

        if self.dataset_collection is not None:
            for _, obj in vars(self.dataset_collection).items():
                if obj is None:
                    continue 

                data = getattr(obj, "data", None)
                if isinstance(data, dict) and "current_treatments" in data:
                    lengths.append(int(data['current_treatments'].shape[1]))
                
                data_processed_seq = getattr(obj, "data_processed_seq", None)
                if isinstance(data_processed_seq, dict) and "current_treatments" in data_processed_seq:
                    lengths.append(int(data_processed_seq["current_treatments"].shape[1]))
        
        if lengths:
            return max(lengths)
        
        return int(args.dataset.max_seq_length) - 1     
    
    
    def prepare_data(self) -> None:
        # Match GT: multi-input (prev_treatments, prev_outputs, vitals, static)
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def _init_specific(self, args: DictConfig) -> None:
        """
        PFN backbone + g-computation heads init.
        Must output per-time hidden representation hr: [B,T,hr_size] with T matching batch time dim.
        """
        try:
            sub_args = args.model[self.model_type]

            # Keep for logging/config, but PFN itself uses self.seq_len
            self.max_seq_length = sub_args.max_seq_length

            self.hr_size = sub_args.hr_size
            self.seq_hidden_units = sub_args.seq_hidden_units  # embed_dim
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate

            self.patch_size = sub_args.patch_size
            self.n_heads = sub_args.n_heads
            self.d_ff = sub_args.d_ff
            self.e_layers = sub_args.e_layers
            self.activation = sub_args.activation

            if (
                self.hr_size is None
                or self.seq_hidden_units is None
                or self.fc_hidden_units is None
                or self.dropout_rate is None
                or self.patch_size is None
                or self.n_heads is None
                or self.d_ff is None
                or self.e_layers is None
            ):
                raise MissingMandatoryValue()

            L = self.seq_len 
            self.patch_stride = self.patch_size // 2
            assert self.patch_stride > 0, "patch_size must be >= 2"
            assert L >= self.patch_size, "patch_size cannot exceed sequence length"

            # Dynamic channels only; static will be injected separately.
            self.dynamic_channels = self.dim_treatments + self.dim_outcome
            if self.has_vitals:
                self.dynamic_channels += self.dim_vitals 

            # 1) PFN embedding (expects x: [B,L,C])
            self.enc_embedding = DataEmbedding_FeaturePatching(
                seq_len=L,
                patch_size=self.patch_size,
                embed_dim=self.seq_hidden_units,
                dropout=self.dropout_rate,
            )

            # patch endpoints on the original time grid
            patch_ends = torch.arange(
                0, L - self.patch_size + 1, self.patch_stride
            ) + (self.patch_size - 1)
            self.register_buffer("patch_end_idx", patch_ends.long(), persistent=False)
            self._n_patches = int(patch_ends.numel())

            # Learn feature importance within each patch
            self.feature_gate = nn.Linear(self.seq_hidden_units, 1)

            n_patch_layers = max(1, self.e_layers // 2)
            n_time_layers = max(0, self.e_layers - n_patch_layers)

            # causal patch-time encoder
            self.patch_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(True, attention_dropout=self.dropout_rate, output_attention=False),
                            self.seq_hidden_units,
                            self.n_heads,
                        ),
                        self.seq_hidden_units,
                        self.d_ff,
                        dropout=self.dropout_rate,
                        activation=self.activation,
                    )
                    for _ in range(n_patch_layers)
                ],
                norm_layer=nn.LayerNorm(self.seq_hidden_units),
            )

            # per-time skip from raw dymanic inputs
            self.input_skip_proj = nn.Linear(self.dynamic_channels, self.seq_hidden_units)

            # static conditioning like GT
            self.static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)

            self.time_positional_encoding = AbsolutePositionalEncoding(
                L, self.seq_hidden_units, trainable=True
            )

            # causal time encoder after patch  -> time lifting
            self.time_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(True, attention_dropout=self.dropout_rate, output_attention=False),
                            self.seq_hidden_units,
                            self.n_heads,
                        ),
                        self.seq_hidden_units,
                        self.d_ff,
                        dropout=self.dropout_rate,
                        activation=self.activation,
                    )
                    for _ in range(n_time_layers)
                ],
                norm_layer=nn.LayerNorm(self.seq_hidden_units),
            )

            self.pre_hr_norm = nn.LayerNorm(self.seq_hidden_units)
            self.hr_output_transformation = nn.Linear(self.seq_hidden_units, self.hr_size)
            self.hr_dropout = nn.Dropout(self.dropout_rate)

            # 4) G-computation heads (exactly like GT)
            self.G_comp_heads = nn.ModuleList(
                [
                    OutcomeHead(
                        self.seq_hidden_units,  # kept consistent with other models
                        self.hr_size,
                        self.fc_hidden_units,
                        self.dim_treatments,
                        self.dim_outcome,
                    )
                    for _ in range(self.projection_horizon + 1)
                ]
            )

        except MissingMandatoryValue:
            logger.warning(
                f"{self.model_type} not fully initialised - some mandatory args are missing "
                f"(ok if doing hyperparameter search)."
            )

    def _build_series_input(self, prev_treatments, vitals, prev_outputs) -> torch.Tensor:
        """
        Build x: [B,T,C] on a discrete grid (same spirit as GT; no torchcde coefficients).
        """
        parts = [prev_treatments, prev_outputs]
        if self.has_vitals:
            parts.append(vitals)
        x = torch.cat(parts, dim=-1)  # [B,T,C_dyn]
        return x
    
    def _align_to_len(self, x: torch.Tensor, L: int) -> torch.Tensor:
        B, T, C = x.shape
        if T < L:
            pad = torch.zeros(B, L - T, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif T > L:
            x = x[:, :L, :]
        return x 
    
    def _make_causal_mask(self, valid_vec: torch.Tensor) -> FixedBoolMask:
        """
        valid_vec: [B, L_bool]
        Masks:
            - future keys
            - invalid keys (inactive positions)
        We do NOT mask invalid queries, to avoid all -inf rows.
        """
        B, L = valid_vec.shape
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=valid_vec.device),
            diagonal=1
        ) # [L, L]
        invalid_keys = ~valid_vec[:, None, None, :] # [B, 1, 1, L]
        mask = causal[None, None, :, :] | invalid_keys
        return FixedBoolMask(mask) 
    
    def _make_safe_valid(self, valid_vec: torch.Tensor) -> torch.Tensor:
        """
        valid_vec: [B, L_bool]
        Ensures at least one valid key per batch row to avoid all -inf attention rows.
        """
        safe_valid = valid_vec.clone()
        no_valid = ~safe_valid.any(dim=1)
        if no_valid.any():
            safe_valid[no_valid, 0] = True
        return safe_valid 
    

    def build_hr(self, prev_treatments, vitals, prev_outputs, static_features, active_entries) -> torch.Tensor:
        """
        Causal PFN:
            1) masked dynamic input
            2) feature-patch extraction
            3) feature pooling within each patch
            4) causal patch-time encoder
            5) causal patch->time lift
            6) causal time encoder + static conditioning
        """
        x_dyn = self._build_series_input(prev_treatments, vitals, prev_outputs)  # [B,T,C_dyn]
        B, T_batch, C_dyn = x_dyn.shape
        L = self.seq_len

        x_dyn = self._align_to_len(x_dyn, L)
        active = self._align_to_len(active_entries, L) # [B, L, 1]
        x_dyn = x_dyn * active # crucial 

        # [B, C_dyn, P, D]
        tokens = self.enc_embedding(x_dyn, flatten=False)

        #Learned feature pooling for each patch
        feat_logits = self.feature_gate(tokens).squeeze(-1) # [B, C_dyn, P]
        feat_weights = torch.softmax(feat_logits, dim=1).unsqueeze(-1)
        patch_tokens = (tokens * feat_weights).sum(dim=1) # [B, P, D]

        # a patch is valid iff its endpoint is active 
        patch_valid = active[:, self.patch_end_idx, 0] > 0 #[B, P]
        safe_patch_valid = self._make_safe_valid(patch_valid)
        patch_mask = self._make_causal_mask(safe_patch_valid)

        patch_tokens, _ = self.patch_encoder(patch_tokens, attn_mask=patch_mask)
        patch_tokens = patch_tokens * patch_valid.unsqueeze(-1)

        # causal patch -> time lift:
        # at time t, only use the latest patch whose endpoint <= t
        t_idx = torch.arange(L, device=x_dyn.device)
        latest_patch = torch.bucketize(t_idx, self.patch_end_idx, right=True) - 1 # [L]

        # raw per-time skip path
        time_hidden = self.input_skip_proj(x_dyn) # [B, L, D]

        usable = latest_patch >= 0
        if usable.any():
            time_hidden[:, usable, :] = (
                time_hidden[:, usable, :]
                + patch_tokens[:, latest_patch[usable], :]
            )
        
        # static conditioning
        static_ctx = self.static_input_transformation(static_features).unsqueeze(1) # [B, 1, D]
        time_hidden = time_hidden + static_ctx

        # explicit temporal position
        time_hidden = time_hidden + self.time_positional_encoding(time_hidden)
        time_hidden = time_hidden * active 

        # causal time encoder
        time_valid = active.squeeze(-1) > 0 # [B, L]
        safe_time_valid = self._make_safe_valid(time_valid)
        time_mask = self._make_causal_mask(safe_time_valid)
        time_hidden, _ = self.time_encoder(time_hidden, attn_mask=time_mask)

        time_hidden = self.pre_hr_norm(time_hidden)
        time_hidden = time_hidden * active 

        hr = F.elu(self.hr_output_transformation(self.hr_dropout(time_hidden)))
        hr = hr * active 

        return hr[:, :T_batch, :]



    def forward(self, batch):
        """
        GT-compatible forward:
        - training:
            - if projection_horizon==0: return (pred_factuals, None, None, active_entries)
            - else: return (None, pred_pseudos_all_steps, pseudo_outcomes_all_steps, active_entries_all_steps)
        - eval:
            return (pred_outcomes, hr)
        """
        prev_treatments = batch["prev_treatments"]
        vitals = batch["vitals"] if self.has_vitals else None
        prev_outputs = batch["prev_outputs"]
        static_features = batch["static_features"]
        curr_treatments = batch["current_treatments"]
        active_entries = batch["active_entries"].clone()

        batch_size = prev_treatments.size(0)
        time_dim = prev_treatments.size(1)

        if self.training:
            # ---- factual one-step training ----
            if self.projection_horizon == 0:
                hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)
                # Guard: ensure time dims match
                if hr.size(1) != curr_treatments.size(1):
                    T = curr_treatments.size(1)
                    hr = hr[:, :T, :]
                pred_factuals = self.G_comp_heads[0].build_outcome(hr, curr_treatments)  # [B,T,Y]
                return pred_factuals, None, None, active_entries

            # ---- GT-style g-computation training ----
            pseudo_outcomes_all_steps = torch.zeros(
                (batch_size, time_dim - self.projection_horizon - 1, self.projection_horizon + 1, self.dim_outcome),
                device=self.device,
            )
            pred_pseudos_all_steps = torch.zeros_like(pseudo_outcomes_all_steps)
            active_entries_all_steps = torch.zeros(
                (batch_size, time_dim - self.projection_horizon - 1, 1), device=self.device
            )

            for t in range(1, time_dim - self.projection_horizon):
                current_active_entries = batch["active_entries"].clone()
                current_active_entries[:, int(t + self.projection_horizon) :] = 0.0
                active_entries_all_steps[:, t - 1, :] = current_active_entries[:, t + self.projection_horizon - 1, :]

                # 1) pseudo outcomes under counterfactual treatment sequence (no grad)
                with torch.no_grad():
                    indexes_cf = (torch.arange(0, time_dim, device=self.device) >= (t - 1)) & (
                        torch.arange(0, time_dim, device=self.device) < (t + self.projection_horizon)
                    )

                    curr_treatments_cf = curr_treatments.clone()
                    curr_treatments_cf[:, indexes_cf, :] = self.treatment_sequence.to(
                        device=curr_treatments.device,
                        dtype=curr_treatments.dtype 
                    )

                    prev_treatments_cf = torch.cat((prev_treatments[:, :1, :], curr_treatments_cf[:, :-1, :]), dim=1)

                    hr_cf = self.build_hr(
                        prev_treatments_cf, vitals, prev_outputs, static_features, current_active_entries
                    )

                    pseudo_outcomes = torch.zeros(
                        (batch_size, self.projection_horizon + 1, self.dim_outcome), device=self.device
                    )

                    for i in range(self.projection_horizon, 0, -1):
                        pseudo_outcome = self.G_comp_heads[i].build_outcome(hr_cf, curr_treatments_cf)[:, t + i - 1, :]
                        pseudo_outcomes[:, i - 1, :] = pseudo_outcome

                    pseudo_outcomes[:, -1, :] = batch["outputs"][:, t + self.projection_horizon - 1, :]
                    pseudo_outcomes_all_steps[:, t - 1, :, :] = pseudo_outcomes

                # 2) predict pseudo outcomes from factual hr (grad)
                hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, current_active_entries)

                pred_pseudos = torch.zeros(
                    (batch_size, self.projection_horizon + 1, self.dim_outcome), device=self.device
                )
                for i in range(self.projection_horizon, -1, -1):
                    pred_pseudo = self.G_comp_heads[i].build_outcome(hr, curr_treatments)[:, t + i - 1, :]
                    pred_pseudos[:, i, :] = pred_pseudo

                pred_pseudos_all_steps[:, t - 1, :, :] = pred_pseudos

            return None, pred_pseudos_all_steps, pseudo_outcomes_all_steps, active_entries_all_steps

        # ---- evaluation / prediction ----
        fixed_split = batch["sequence_lengths"] - self.max_projection if self.projection_horizon > 0 else batch["sequence_lengths"]
        for i in range(len(active_entries)):
            active_entries[i, int(fixed_split[i] + self.projection_horizon) :] = 0.0

        hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)

        if self.projection_horizon > 0:
            pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)
            index_pred = (torch.arange(0, time_dim, device=self.device) == fixed_split[..., None] - 1)
            pred_outcomes = pred_outcomes[index_pred]  # [B,Y] (GT-style)
        else:
            pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)  # [B,T,Y]

        return pred_outcomes, hr

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        pred_factuals, pred_pseudos, pseudo_outcomes, active_entries_all_steps = self(batch)

        if self.projection_horizon > 0:
            active_entries_all_steps = active_entries_all_steps.unsqueeze(-2)  # [B,steps,1,1]
            mse = F.mse_loss(pred_pseudos, pseudo_outcomes, reduction="none")  # [B,steps,H+1,Y]
            mse = (mse * active_entries_all_steps).sum(dim=(0, 1)) / (
                active_entries_all_steps.sum(dim=(0, 1)).clamp_min(1.0) * self.dim_outcome
            )

            for i in range(mse.shape[0]):
                self.log(
                    f"{self.model_type}_mse_{i}",
                    mse[i].mean(),
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    prog_bar=True,
                )

            loss = mse.mean()
        else:
            mse = F.mse_loss(pred_factuals, batch["outputs"], reduction="none")  # [B,T,Y]
            loss = (mse * batch["active_entries"]).sum() / (batch["active_entries"].sum().clamp_min(1.0) * self.dim_outcome)

        self.log(f"{self.model_type}_train_loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        pred_outcomes, hr = self(batch)
        return pred_outcomes.cpu(), hr.cpu()

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f"Predictions for {dataset.subset_name}.")
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        outcome_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy()
    

    def get_normalised_n_step_rmses(self, dataset: Dataset):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')

        outputs_scaled = self.get_predictions(dataset)
        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        # Only evaluate RMSE on final outcome (same as GT)
        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means
            mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs'][:, (self.projection_horizon - 1)]) ** 2)
        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs'][:, (self.projection_horizon - 1)]) ** 2)

        nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
        not_nan = np.array([i for i in range(outputs_scaled.shape[0]) if i not in nan_idx])
        mse = mse[not_nan]

        mse = mse.mean()  # mean across batch
        rmse_normalised = np.sqrt(mse) / dataset.norm_const

        if percentage:
            rmse_normalised *= 100.0

        return rmse_normalised



    def configure_optimizers(self):
        optimizer = self._get_optimizer(list(self.named_parameters()))
        if self.hparams.model[self.model_type]["optimizer"]["lr_scheduler"]:
            return self._get_lr_schedulers(optimizer)
        return optimizer

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args["learning_rate"]
        sub_args.batch_size = new_args["batch_size"]
        sub_args.projection_horizon = 0 

        if "n_heads" in new_args:
            sub_args.n_heads = new_args["n_heads"]

        if "e_layers" in new_args:
            sub_args.e_layers = new_args["e_layers"]
        
        if "patch_size" in new_args:
            sub_args.patch_size = new_args["patch_size"]

        if "d_ff" in new_args:
            sub_args.d_ff = new_args["d_ff"]

        if "seq_hidden_units" in new_args:
            sub_args.seq_hidden_units = int(input_size * new_args["seq_hidden_units"])
            common_multiplier = np.lcm.reduce([sub_args.n_heads, 2]).item()
            if sub_args.seq_hidden_units % common_multiplier != 0:
                sub_args.seq_hidden_units += (
                    common_multiplier - sub_args.seq_hidden_units % common_multiplier
                )

        if "hr_size" in new_args:
            sub_args.hr_size = int(input_size * new_args["hr_size"])

        sub_args.fc_hidden_units = int(sub_args.hr_size * new_args["fc_hidden_units"])
        sub_args.dropout_rate = new_args["dropout_rate"]
