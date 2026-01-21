# batch_topk_sae.py
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, model_validator
from jaxtyping import Float

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.utils import (
    init_decoder_orthogonal_cuda,
    update_dead_feature_stats,
    maybe_compute_auxk_features,
    compute_aux_loss_with_logging,
)
from utils.enums import SAEType


class BatchTopKSAEConfig(SAEConfig):
    """
    Config for Batch Top-K SAE.

    Notes:
    - Applies Top-K selection across the entire batch (flattened) rather than per-sample
    - Supports auxiliary loss for dead feature mitigation
    - Optionally applies input unit normalization
    - Learns a running threshold for inference (when batch top-k isn't applicable)
    """
    sae_type: SAEType = Field(default=SAEType.BATCH_TOPK, description="Type of SAE (automatically set to batch_topk)")
    k: int = Field(..., description="Number of active features to keep per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    
    # Optional: dead-feature mitigation via auxiliary Top-K
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead (number of tokens)")
    
    # Input preprocessing
    input_unit_norm: bool = Field(False, description="Apply unit normalization to inputs")
    
    # Running threshold for inference
    threshold_momentum: float = Field(0.99, description="Momentum for running threshold EMA update")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.BATCH_TOPK
        return values


class BatchTopKSAEOutput(SAEOutput):
    """
    BatchTopK SAE output extending SAEOutput with useful intermediates.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (after centering)
    mask: Float[torch.Tensor, "... c"]     # binary mask of selected Top-K indices
    x_mean: torch.Tensor | None = None     # mean for unit norm (if applied)
    x_std: torch.Tensor | None = None      # std for unit norm (if applied)
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents


class BatchTopKSAE(BaseSAE):
    """
    Batch Top-K Sparse Autoencoder:
      - Linear encoder/decoder (no bias on the linear layers)
      - Single learned decoder_bias used to center input and add back after decode
      - Top-K selection applied across the entire batch (flattened activations)
      - MSE reconstruction loss with optional auxiliary loss for dead features
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        k: int,
        sparsity_coeff: float | None = None,  # unused; kept for API parity
        mse_coeff: float | None = None,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        dead_toks_threshold: int | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        input_unit_norm: bool = False,
        threshold_momentum: float = 0.99,
    ):
        """
        Args:
            input_size: Dimensionality of inputs (e.g., residual stream width).
            n_dict_components: Number of dictionary features (latent size).
            k: Number of active features to keep per sample (Top-K).
            sparsity_coeff: Unused for Top-K (present for interface compatibility).
            mse_coeff: Coefficient on MSE reconstruction loss (default 1.0).
            aux_k: If provided (>0), number of auxiliary features for dead latents.
            aux_coeff: Coefficient on the auxiliary reconstruction loss.
            dead_toks_threshold: Threshold for considering a feature as dead.
            init_decoder_orthogonal: Initialize decoder weight columns to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
            input_unit_norm: Apply unit normalization to inputs.
            threshold_momentum: Momentum for running threshold EMA update.
        """
        super().__init__()
        assert k >= 0, "k must be non-negative"
        assert n_dict_components > 0 and input_size > 0

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.k = int(k)
        assert self.k <= n_dict_components, "k must be less than or equal to n_dict_components"

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0  # not used
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0
        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None

        # Input preprocessing flag
        self.input_unit_norm = input_unit_norm
        
        # Running threshold for inference (use double precision for numerical stability, as in SAELens)
        self.threshold_momentum = threshold_momentum
        self.register_buffer("running_threshold", torch.tensor(0.0, dtype=torch.double))

        # Bias used for input centering and added back on decode
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))

        # Linear maps (no bias)
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)

        # Initialize decoder, then (optionally) tie encoder init to decoder^T
        if init_decoder_orthogonal:
            self.decoder.weight.data = init_decoder_orthogonal_cuda(self.decoder.weight)
        else:
            # Random unit-norm columns
            dec_w = torch.randn_like(self.decoder.weight)
            dec_w = F.normalize(dec_w, dim=0)
            self.decoder.weight.data.copy_(dec_w)

        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)

        # Dead latent tracking - counts tokens since last activation
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dict_components, dtype=torch.long))

    def _preprocess_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Apply optional unit normalization to inputs."""
        if self.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def _postprocess_output(self, x_reconstruct: torch.Tensor, x_mean: torch.Tensor | None, x_std: torch.Tensor | None) -> torch.Tensor:
        """Undo unit normalization if it was applied."""
        if self.input_unit_norm and x_mean is not None and x_std is not None:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    def _apply_batch_topk(self, acts: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Top-K selection across the entire batch (flattened).
        Returns:
            acts_topk: sparse activations after masking to Top-K
            mask: binary mask (same shape as acts) with ones at Top-K indices
        """
        # Flatten all activations across the batch
        acts_flat = acts.flatten()
        
        # Select top-k across the entire batch
        total_k = self.k * batch_size
        topk_values, topk_indices = torch.topk(acts_flat, k=total_k, dim=-1)
        
        # Create sparse tensor with selected values
        acts_topk_flat = torch.zeros_like(acts_flat)
        acts_topk_flat.scatter_(-1, topk_indices, topk_values)
        
        # Reshape back to original shape
        acts_topk = acts_topk_flat.reshape(acts.shape)
        mask = (acts_topk > 0).to(acts.dtype)
        
        return acts_topk, mask

    @torch.no_grad()
    def _update_running_threshold(self, acts: torch.Tensor) -> None:
        """Update running threshold based on minimum positive activation after batch top-k.
        
        This threshold is used during inference when batch top-k isn't applicable.
        Following SAELens implementation for numerical stability.
        
        Args:
            acts: Activations after batch top-k (sparse tensor with selected values)
        """
        positive_mask = acts > 0
        
        # Disable autocast to prevent numerical issues (following SAELens)
        with torch.autocast(self.running_threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = acts[positive_mask].min().to(self.running_threshold.dtype)
                lr = 1 - self.threshold_momentum  # Convert momentum to learning rate
                self.running_threshold = (1 - lr) * self.running_threshold + lr * min_positive

    def _apply_threshold(self, acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply learned threshold during inference (JumpReLU-style).
        
        Args:
            acts: Pre-activations (after ReLU, before sparsification)
            
        Returns:
            acts_sparse: Activations with values below threshold zeroed out
            mask: Binary mask of active features
        """
        # Convert threshold to acts dtype for comparison
        threshold = self.running_threshold.to(acts.dtype)
        mask = (acts > threshold).to(acts.dtype)
        acts_sparse = acts * mask
        return acts_sparse, mask

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> BatchTopKSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        
        During training: Uses batch-level top-k selection and updates running threshold.
        During inference: Uses learned threshold for sparsification.
        """
        # Store original shape and flatten batch dimensions
        original_shape = x.shape
        x = x.reshape(-1, self.input_size)
        batch_size = x.shape[0]
        
        # Preprocess input
        x, x_mean, x_std = self._preprocess_input(x)
        
        # Center input
        x_centered = x - self.decoder_bias
        
        # Encoder preactivations
        preacts = F.relu(self.encoder(x_centered))
        
        # Apply sparsification: batch top-k during training, threshold during inference
        if self.training:
            # Batch-level Top-K sparsification
            acts_topk, mask = self._apply_batch_topk(preacts, batch_size)
            
            # Update running threshold for inference
            self._update_running_threshold(acts_topk)
        else:
            # Use learned threshold during inference
            acts_topk, mask = self._apply_threshold(preacts)
        
        # Update dead feature statistics
        update_dead_feature_stats(
            activations=acts_topk,
            stats_last_nonzero=self.stats_last_nonzero,
            training=self.training,
            dead_toks_threshold=self.dead_toks_threshold,
        )
        
        # Compute auxiliary indices for dead features if needed
        auxk_values, auxk_indices = maybe_compute_auxk_features(
            preacts=preacts,
            stats_last_nonzero=self.stats_last_nonzero,
            aux_k=self.aux_k,
            aux_coeff=self.aux_coeff,
            dead_toks_threshold=self.dead_toks_threshold,
        )
        
        # Decode using normalized dictionary elements + add bias back
        x_hat = F.linear(acts_topk, self.dict_elements, bias=self.decoder_bias)
        
        # Postprocess if unit norm was applied
        x_hat = self._postprocess_output(x_hat, x_mean, x_std)
        
        # Reshape outputs back to original shape
        x_hat = x_hat.reshape(original_shape)
        acts_topk = acts_topk.reshape(*original_shape[:-1], self.n_dict_components)
        preacts = preacts.reshape(*original_shape[:-1], self.n_dict_components)
        mask = mask.reshape(*original_shape[:-1], self.n_dict_components)
        
        return BatchTopKSAEOutput(
            input=x.reshape(original_shape),
            c=acts_topk,
            output=x_hat,
            logits=None,
            preacts=preacts,
            mask=mask,
            x_mean=x_mean.reshape(original_shape[:-1] + (1,)) if x_mean is not None else None,
            x_std=x_std.reshape(original_shape[:-1] + (1,)) if x_std is not None else None,
            auxk_indices=auxk_indices,
            auxk_values=auxk_values
        )

    def compute_loss(self, output: BatchTopKSAEOutput) -> SAELoss:
        """
        Loss = mse_coeff * MSE + aux_coeff * AuxK (optional)

        - No explicit L1 sparsity term (sparsity enforced by batch Top-K).
        - AuxK: Reconstruct the residual error using dead features.
        """
        # Need to flatten for loss computation if unit norm was applied
        input_for_loss = output.input
        output_for_loss = output.output
        
        if self.input_unit_norm and output.x_mean is not None and output.x_std is not None:
            # Re-normalize the input for fair MSE comparison
            input_normalized = (input_for_loss - output.x_mean) / (output.x_std + 1e-5)
            output_normalized = (output_for_loss - output.x_mean) / (output.x_std + 1e-5)
            mse_loss = F.mse_loss(output_normalized, input_normalized)
        else:
            mse_loss = F.mse_loss(output_for_loss, input_for_loss)
            
        total_loss = self.mse_coeff * mse_loss
        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": mse_loss.detach().clone(),
            "topk_threshold": self.running_threshold.detach().clone().float(),
        }

        # Optional auxiliary dead-feature loss using residual reconstruction
        weighted_aux_loss, aux_loss_for_logging = compute_aux_loss_with_logging(
            auxk_indices=output.auxk_indices,
            auxk_values=output.auxk_values,
            input_tensor=output.input,
            output_tensor=output.output,
            decoder_bias=self.decoder_bias,
            dict_elements=self.dict_elements,
            n_dict_components=self.n_dict_components,
            input_size=self.input_size,
            aux_k=self.aux_k,
            aux_coeff=self.aux_coeff,
        )
        total_loss = total_loss + weighted_aux_loss
        loss_dict["aux_loss"] = aux_loss_for_logging

        return SAELoss(loss=total_loss, loss_dict=loss_dict)

    @property
    def dict_elements(self) -> torch.Tensor:
        """
        Column-wise unit-norm decoder (dictionary) – normalized every forward.
        """
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
