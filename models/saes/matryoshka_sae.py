# matryoshka_sae.py
"""
Matryoshka Batch TopK SAE - A variant of BatchTopKSAE with nested dictionary groups.

Features are divided into groups, and the loss is computed progressively at each group
boundary. This encourages the model to learn meaningful representations at multiple
scales - smaller prefixes of the dictionary should be useful on their own.

Reference: dictionary_learning repo MatryoshkaBatchTopKSAE implementation.
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, model_validator
from jaxtyping import Float
from math import isclose

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.utils import (
    init_decoder_orthogonal_cuda,
    update_dead_feature_stats,
    maybe_compute_auxk_features,
    compute_aux_loss_with_logging,
)
from utils.enums import SAEType


class MatryoshkaSAEConfig(SAEConfig):
    """
    Config for Matryoshka Batch Top-K SAE.

    Extends BatchTopKSAE with nested dictionary groups for progressive reconstruction.
    
    Notes:
    - Features are divided into groups via group_fractions (must sum to 1.0)
    - Loss is computed progressively at each group boundary
    - Each group can have a different weight in the total loss
    - Uses the same batch top-k selection and auxiliary loss as BatchTopKSAE
    """
    sae_type: SAEType = Field(default=SAEType.MATRYOSHKA, description="Type of SAE (automatically set to matryoshka)")
    k: int = Field(..., description="Number of active features to keep per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    
    # Group configuration for Matryoshka structure
    group_fractions: list[float] = Field(
        default=[1/32, 1/16, 1/8, 1/4, 1/2 + 1/32],
        description="Fractions of dict_size for each group (must sum to 1.0)"
    )
    group_weights: list[float] | None = Field(
        None, 
        description="Weights for loss at each group boundary (default: uniform)"
    )
    
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
            values["sae_type"] = SAEType.MATRYOSHKA
        return values

    @model_validator(mode="after")
    def validate_group_fractions(self) -> "MatryoshkaSAEConfig":
        """Validate that group_fractions sum to 1.0."""
        if not isclose(sum(self.group_fractions), 1.0, rel_tol=1e-5):
            raise ValueError(f"group_fractions must sum to 1.0, got {sum(self.group_fractions)}")
        if any(f <= 0 for f in self.group_fractions):
            raise ValueError("All group_fractions must be positive")
        return self

    @model_validator(mode="after")
    def validate_group_weights(self) -> "MatryoshkaSAEConfig":
        """Validate group_weights if provided."""
        if self.group_weights is not None:
            if len(self.group_weights) != len(self.group_fractions):
                raise ValueError(
                    f"group_weights length ({len(self.group_weights)}) must match "
                    f"group_fractions length ({len(self.group_fractions)})"
                )
        return self


class MatryoshkaSAEOutput(SAEOutput):
    """
    Matryoshka SAE output extending SAEOutput with useful intermediates.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (after ReLU)
    mask: Float[torch.Tensor, "... c"]     # binary mask of selected Top-K indices
    x_mean: torch.Tensor | None = None     # mean for unit norm (if applied)
    x_std: torch.Tensor | None = None      # std for unit norm (if applied)
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents


class MatryoshkaSAE(BaseSAE):
    """
    Matryoshka Batch Top-K Sparse Autoencoder:
      - Linear encoder/decoder (no bias on the linear layers)
      - Single learned decoder_bias used to center input and add back after decode
      - Top-K selection applied across the entire batch (flattened activations)
      - Progressive MSE loss at each group boundary with configurable weights
      - Optional auxiliary loss for dead features
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        k: int,
        group_fractions: list[float] | None = None,
        group_weights: list[float] | None = None,
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
            group_fractions: Fractions of dict_size for each group (must sum to 1.0).
                Default: [1/32, 1/16, 1/8, 1/4, 1/2 + 1/32] (5 groups).
            group_weights: Weights for loss at each group boundary (default: uniform).
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

        # Set default group fractions if not provided
        if group_fractions is None:
            group_fractions = [1/32, 1/16, 1/8, 1/4, 1/2 + 1/32]

        # Validate group fractions
        assert isclose(sum(group_fractions), 1.0, rel_tol=1e-5), "group_fractions must sum to 1.0"
        assert all(f > 0 for f in group_fractions), "all group_fractions must be positive"

        # Compute group sizes from fractions
        # Calculate all groups except the last one
        group_sizes = [int(f * n_dict_components) for f in group_fractions[:-1]]
        # Put remainder in the last group to ensure exact sum
        group_sizes.append(n_dict_components - sum(group_sizes))
        
        self.num_groups = len(group_sizes)
        self.register_buffer("group_sizes", torch.tensor(group_sizes, dtype=torch.long))
        
        # Compute cumulative group boundaries: [0, size_0, size_0+size_1, ...]
        boundaries = [0]
        for size in group_sizes:
            boundaries.append(boundaries[-1] + size)
        self.register_buffer("group_boundaries", torch.tensor(boundaries, dtype=torch.long))
        
        # Set group weights (default: uniform)
        if group_weights is None:
            group_weights = [1.0 / self.num_groups] * self.num_groups
        assert len(group_weights) == self.num_groups, "group_weights must match number of groups"
        self.register_buffer("group_weights", torch.tensor(group_weights, dtype=torch.float32))

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0  # not used
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0
        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None

        # Input preprocessing flag
        self.input_unit_norm = input_unit_norm
        
        # Running threshold for inference (use double precision for numerical stability)
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
        mask = (acts_topk > 0).float()
        
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
        
        # Disable autocast to prevent numerical issues
        with torch.autocast(self.running_threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = acts[positive_mask].min().to(self.running_threshold.dtype)
                lr = 1 - self.threshold_momentum  # Convert momentum to learning rate
                self.running_threshold = (1 - lr) * self.running_threshold + lr * min_positive

    def _apply_threshold(self, acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply learned threshold during inference (JumpReLU-style).
        
        NOTE: This method is DEPRECATED and should not be used.
        The threshold-based approach doesn't preserve L_0 = k during inference.
        Use _apply_per_sample_topk() instead.
        
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

    def _apply_per_sample_topk(self, acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-sample Top-K selection during inference.
        
        This maintains consistent L_0 = k behavior between training and inference,
        unlike the threshold-based approach which can result in variable L_0.
        
        Args:
            acts: Pre-activations (after ReLU), shape (batch, dict_size)
            
        Returns:
            acts_topk: Activations with only top-k values per sample
            mask: Binary mask of active features
        """
        # Get top-k values and indices per sample
        topk_values, topk_indices = torch.topk(acts, k=self.k, dim=-1)
        
        # Create sparse tensor with selected values
        acts_topk = torch.zeros_like(acts)
        acts_topk.scatter_(-1, topk_indices, topk_values)
        
        # Create mask (with ReLU, values are non-negative so > 0 works)
        mask = (acts_topk > 0).float()
        
        return acts_topk, mask

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> MatryoshkaSAEOutput:
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
        
        # Apply batch top-k sparsification (same for training and inference)
        # Using batch top-k for inference maintains consistency with training and gives better MSE
        acts_topk, mask = self._apply_batch_topk(preacts, batch_size)
        
        # Update running threshold during training (kept for backwards compatibility/logging)
        if self.training:
            self._update_running_threshold(acts_topk)
        
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
        # For forward output, we use the full reconstruction (all groups)
        x_hat = F.linear(acts_topk, self.dict_elements, bias=self.decoder_bias)
        
        # Postprocess if unit norm was applied
        x_hat = self._postprocess_output(x_hat, x_mean, x_std)
        
        # Reshape outputs back to original shape
        x_hat = x_hat.reshape(original_shape)
        acts_topk = acts_topk.reshape(*original_shape[:-1], self.n_dict_components)
        preacts = preacts.reshape(*original_shape[:-1], self.n_dict_components)
        mask = mask.reshape(*original_shape[:-1], self.n_dict_components)
        
        return MatryoshkaSAEOutput(
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

    def compute_loss(self, output: MatryoshkaSAEOutput) -> SAELoss:
        """
        Matryoshka progressive loss computation.
        
        Loss = sum over groups: group_weight[i] * MSE(x, reconstruction_up_to_group_i)
        
        For each group boundary, we reconstruct using features [0:boundary] and compute
        the weighted MSE. This encourages meaningful representations at each prefix.
        """
        # Need to flatten for loss computation
        input_for_loss = output.input.reshape(-1, self.input_size)
        acts = output.c.reshape(-1, self.n_dict_components)
        
        # Handle unit norm preprocessing for fair comparison
        if self.input_unit_norm and output.x_mean is not None and output.x_std is not None:
            x_mean_flat = output.x_mean.reshape(-1, 1)
            x_std_flat = output.x_std.reshape(-1, 1)
            input_normalized = (input_for_loss - x_mean_flat) / (x_std_flat + 1e-5)
            input_for_loss = input_normalized
        
        # Get normalized decoder weights
        dict_elements = self.dict_elements  # shape: (input_size, n_dict_components)
        
        # Progressive reconstruction at each group boundary
        # Start with just the decoder bias
        x_reconstruct = self.decoder_bias.unsqueeze(0).expand(acts.shape[0], -1).clone()
        
        total_mse_loss = torch.tensor(0.0, device=acts.device, dtype=acts.dtype)
        group_losses = []
        
        for i in range(self.num_groups):
            # Get this group's contribution (features from previous boundary to current)
            group_start = self.group_boundaries[i].item()
            group_end = self.group_boundaries[i + 1].item()
            
            # Slice activations and decoder for this group
            acts_slice = acts[:, group_start:group_end]  # (batch, group_size)
            W_dec_slice = dict_elements[:, group_start:group_end]  # (input_size, group_size)
            
            # Add this group's contribution to reconstruction
            # F.linear(input, weight) computes input @ weight.T
            # We have acts_slice @ W_dec_slice.T which gives (batch, input_size)
            x_reconstruct = x_reconstruct + F.linear(acts_slice, W_dec_slice)
            
            # Compute MSE at this group boundary (reconstruction uses features 0:end_idx)
            group_mse = F.mse_loss(x_reconstruct, input_for_loss)
            group_losses.append(group_mse.detach().clone())
            
            # Add weighted loss
            total_mse_loss = total_mse_loss + self.group_weights[i] * group_mse
        
        total_loss = self.mse_coeff * total_mse_loss
        
        # Build loss dict with group-wise losses for logging
        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": total_mse_loss.detach().clone(),
            "topk_threshold": self.running_threshold.detach().clone().float(),
        }
        
        # Add individual group losses for debugging/analysis
        for i, gl in enumerate(group_losses):
            loss_dict[f"mse_loss_group_{i}"] = gl

        # Optional auxiliary dead-feature loss using residual reconstruction
        # Use the final full reconstruction for aux loss
        x_hat_full = output.output.reshape(-1, self.input_size)
        weighted_aux_loss, aux_loss_for_logging = compute_aux_loss_with_logging(
            auxk_indices=output.auxk_indices,
            auxk_values=output.auxk_values,
            input_tensor=input_for_loss,
            output_tensor=x_hat_full,
            decoder_bias=self.decoder_bias,
            dict_elements=dict_elements,
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
