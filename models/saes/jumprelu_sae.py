# jumprelu_sae.py
"""
JumpReLU SAE implementation consistent with the original paper.

Key paper-consistent behaviors:
1. Reconstruction loss: SSE per example (sum over dims), then batch mean
2. Sparsity loss: Per-example L0 penalty, then batch mean  
3. ReLU on pre-activations before JumpReLU (as paper recommends)
4. Log-threshold parameterization for positive thresholds

Reference: "Scaling and evaluating sparse autoencoders" (Gao et al.)
"""
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
from models.saes.activations import StepFunction, JumpReLU
from utils.enums import SAEType


class JumpReLUSAEConfig(SAEConfig):
    """
    Config for JumpReLU SAE (paper-consistent implementation).
    
    Loss formulation (matching paper):
    - Reconstruction: E_x[ sum_d (x_d - x̂_d)² ]  (SSE per example, batch mean)
    - Sparsity: E_x[ λ · (L0(x) / target_l0 - 1)² ]  (per-example penalty, batch mean)
    
    Notes:
    - Uses JumpReLU activation with learned thresholds
    - Bandwidth parameter controls gradient smoothness for threshold learning
    - normalize_input is recommended when using small bandwidth (e.g., 0.001)
    """
    sae_type: SAEType = Field(default=SAEType.JUMP_RELU, description="Type of SAE (automatically set to jump_relu)")
    target_l0: float = Field(..., description="Target L0 sparsity (number of active features per sample)")
    bandwidth: float = Field(0.001, description="Bandwidth for JumpReLU gradient approximation (paper uses 0.001)")
    initial_threshold: float = Field(0.001, description="Initial threshold for JumpReLU activation")
    use_pre_enc_bias: bool = Field(False, description="Whether to subtract decoder bias before encoding")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    
    # Input normalization (recommended for consistent bandwidth behavior)
    normalize_input: bool = Field(False, description="Scale input to unit mean squared L2 norm (recommended when using bandwidth=0.001)")
    l0_ema_momentum: float = Field(0.99, description="Momentum for running mean squared norm statistics")
    
    # Dead feature tracking and auxiliary loss
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead (number of tokens)")
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss (select top aux_k from the inactive set)")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")
    
    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.JUMP_RELU
        return values


class JumpReLUSAEOutput(SAEOutput):
    """
    JumpReLU SAE output extending SAEOutput.
    """
    pre_activations: Float[torch.Tensor, "... c"]  # pre-JumpReLU activations
    l0: Float[torch.Tensor, "..."]  # L0 norm per sample
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents


class JumpReLUSAE(BaseSAE):
    """
    JumpReLU Sparse Autoencoder (paper-consistent implementation):
      - Linear encoder/decoder with bias
      - JumpReLU activation with learned thresholds
      - SSE reconstruction loss (sum over dims, mean over batch)
      - Per-example L0 sparsity loss (then batch mean)
    """
    
    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        target_l0: float,
        bandwidth: float = 0.001,
        initial_threshold: float = 0.001,
        use_pre_enc_bias: bool = False,
        sparsity_coeff: float | None = None,
        mse_coeff: float | None = None,
        dead_toks_threshold: int | None = None,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        normalize_input: bool = False,
        l0_ema_momentum: float = 0.99,
    ):
        """
        Args:
            input_size: Dimensionality of inputs.
            n_dict_components: Number of dictionary features.
            target_l0: Target L0 sparsity (number of active features per sample).
            bandwidth: Bandwidth for JumpReLU gradient approximation (paper uses 0.001).
            initial_threshold: Initial threshold for JumpReLU activation.
            use_pre_enc_bias: Whether to subtract decoder bias before encoding.
            sparsity_coeff: Coefficient for target L0 regularization loss.
            mse_coeff: Coefficient on reconstruction loss.
            dead_toks_threshold: Threshold for dead feature tracking.
            aux_k: If provided (>0), number of auxiliary features from the inactive set.
            aux_coeff: Coefficient on the auxiliary reconstruction loss.
            init_decoder_orthogonal: Initialize decoder weights to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
            normalize_input: Scale input to unit mean squared L2 norm (recommended for bandwidth=0.001).
            l0_ema_momentum: Momentum for running mean squared norm statistics.
        """
        super().__init__()
        assert n_dict_components > 0 and input_size > 0
        assert target_l0 > 0, "target_l0 must be positive"
        
        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.target_l0 = target_l0
        self.bandwidth = bandwidth
        self.initial_threshold = initial_threshold
        self.use_pre_enc_bias = use_pre_enc_bias
        self.normalize_input = normalize_input
        self.l0_ema_momentum = l0_ema_momentum
        
        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        
        # Dead feature tracking and auxiliary loss
        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None
        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0
        
        # Biases
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        self.encoder_bias = nn.Parameter(torch.zeros(n_dict_components))
        
        # Linear maps
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        
        # JumpReLU activation
        self.jumprelu = JumpReLU(
            feature_size=n_dict_components,
            bandwidth=bandwidth,
            initial_threshold=initial_threshold
        )
        
        # Initialize decoder, then (optionally) tie encoder init
        if init_decoder_orthogonal:
            self.decoder.weight.data = init_decoder_orthogonal_cuda(self.decoder.weight)
        else:
            dec_w = torch.randn_like(self.decoder.weight)
            dec_w = F.normalize(dec_w, dim=0)
            self.decoder.weight.data.copy_(dec_w)
        
        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)
        
        # Dead latent tracking
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dict_components, dtype=torch.long))
        
        # Running statistics for input normalization (if enabled)
        # Uses scale-only normalization: x_norm = x / scale to achieve mean squared L2 norm ≈ 1
        if self.normalize_input:
            self.register_buffer("running_msn", torch.tensor(1.0))  # mean squared norm
            self.register_buffer("input_stats_initialized", torch.tensor(False))
    
    def _normalize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Scale-only normalization to achieve mean squared L2 norm ≈ 1.
        
        Computes: x_normalized = x / scale, where scale = sqrt(E[||x||²/d])
        This preserves the mean/direction of activations while ensuring E[||x||²/d] ≈ 1.
        
        Returns the normalized input and scale factor for denormalization.
        """
        # Compute mean squared norm: E[||x||²/d] = mean of all squared elements
        batch_msn = (x ** 2).mean()
        
        if self.training:
            with torch.no_grad():
                if not self.input_stats_initialized:
                    self.running_msn.copy_(batch_msn)
                    self.input_stats_initialized.fill_(True)
                else:
                    # EMA update
                    momentum = self.l0_ema_momentum
                    self.running_msn.mul_(momentum).add_((1 - momentum) * batch_msn)
        
        # Scale factor: sqrt(running_msn)
        scale = (self.running_msn + 1e-8).sqrt()
        x_normalized = x / scale
        
        return x_normalized, scale

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> JumpReLUSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        """
        # Optional: scale-only normalization to achieve mean squared L2 norm ≈ 1
        if self.normalize_input:
            x_normalized, input_scale = self._normalize_input(x)
        else:
            x_normalized = x
            input_scale = None
        
        # Optional: subtract decoder bias before encoding
        if self.use_pre_enc_bias:
            x_enc = x_normalized - self.decoder_bias
        else:
            x_enc = x_normalized
        
        # Encode with ReLU pre-activation (paper: ensures negative pre-acts don't bias gradients)
        pre_activations = F.relu(self.encoder(x_enc) + self.encoder_bias)
        
        # Apply JumpReLU activation
        feature_activations = self.jumprelu(pre_activations)
        
        # Update dead latent statistics if training
        update_dead_feature_stats(
            activations=feature_activations,
            stats_last_nonzero=self.stats_last_nonzero,
            training=self.training,
            dead_toks_threshold=self.dead_toks_threshold,
        )
        
        # Compute auxiliary top-k indices and values for dead latents
        auxk_values, auxk_indices = maybe_compute_auxk_features(
            preacts=pre_activations,
            stats_last_nonzero=self.stats_last_nonzero,
            aux_k=self.aux_k,
            aux_coeff=self.aux_coeff,
            dead_toks_threshold=self.dead_toks_threshold,
        )
        
        # Decode
        x_reconstruct_normalized = F.linear(feature_activations, self.dict_elements) + self.decoder_bias
        
        # Denormalize output back to original scale (scale-only: just multiply)
        if self.normalize_input and input_scale is not None:
            x_reconstruct = x_reconstruct_normalized * input_scale
        else:
            x_reconstruct = x_reconstruct_normalized
        
        # Compute L0 for loss (using differentiable step function)
        threshold = torch.exp(self.jumprelu.log_threshold)
        l0 = StepFunction.apply(
            pre_activations, 
            threshold,
            self.bandwidth
        ).sum(dim=-1)
        
        return JumpReLUSAEOutput(
            input=x,
            c=feature_activations,
            output=x_reconstruct,
            logits=None,
            pre_activations=pre_activations,
            l0=l0,
            auxk_indices=auxk_indices,
            auxk_values=auxk_values,
        )
    
    def compute_loss(self, output: JumpReLUSAEOutput) -> SAELoss:
        """
        Paper-consistent loss computation:
        
        Loss = mse_coeff * E_x[sum_d (x_d - x̂_d)²] + sparsity_coeff * E_x[(L0/target - 1)²]
        
        - Reconstruction: SSE per example (sum over dims), then batch mean
        - Sparsity: Per-example L0 penalty, then batch mean
        """
        # Reconstruction loss: SSE per example, then batch mean (paper formulation)
        reconstruction_error = (output.output - output.input) ** 2
        flat_error = reconstruction_error.reshape(-1, self.input_size)
        reconstruction_loss = flat_error.sum(dim=-1).mean()
        
        # Sparsity loss: per-example penalty, then batch mean (paper formulation)
        per_example_penalty = ((output.l0 / self.target_l0) - 1.0) ** 2
        sparsity_loss = self.sparsity_coeff * per_example_penalty.mean()
        
        # Total loss
        total_loss = self.mse_coeff * reconstruction_loss + sparsity_loss
        
        # For logging, also compute mean L0
        mean_l0 = output.l0.mean()
        
        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": reconstruction_loss.detach().clone(),
            "sparsity_loss": sparsity_loss.detach().clone(),
            "l0_norm": mean_l0.detach().clone(),
        }
        
        # Optional auxiliary dead-feature loss
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
        """Column-wise unit-norm decoder (dictionary)."""
        return F.normalize(self.decoder.weight, dim=0)
    
    @property
    def device(self):
        return next(self.parameters()).device
