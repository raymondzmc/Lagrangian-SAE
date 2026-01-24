# jumprelu_sae.py
"""
JumpReLU SAE implementation.

Key behaviors:
1. Reconstruction loss: SSE in normalized space (sum over dims, mean over batch)
2. Sparsity loss: Mean-L0 penalty ((mean_l0 / target - 1)²) for faster convergence
3. ReLU on pre-activations before JumpReLU (as paper recommends)
4. Raw threshold parameterization (not log-space)
5. Scale-only input normalization to ensure bandwidth works correctly

Reference: "Scaling and evaluating sparse autoencoders" (Gao et al.)
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, model_validator
from jaxtyping import Float

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.activations import RectangleFunction, StepFunction, JumpReLUFunction, JumpReLU
from models.saes.utils import (
    init_decoder_orthogonal_cuda,
    update_dead_feature_stats,
    maybe_compute_auxk_features,
    compute_aux_loss_with_logging,
)
from utils.enums import SAEType


class JumpReLUSAEConfig(SAEConfig):
    """
    Config for JumpReLU SAE (Paper implementation).
    
    Loss formulation:
    - Reconstruction: SSE in normalized space (sum over dims, mean over batch)
    - Sparsity: λ · (E_x[L0(x)] / target_l0 - 1)²  (mean-L0 penalty, faster convergence)
    
    Notes:
    - Uses JumpReLU activation with learned thresholds (raw parameterization)
    - Bandwidth parameter controls gradient smoothness for threshold learning
    - normalize_activations=True is recommended for stable training
    - Loss is computed in normalized space for stability
    """
    sae_type: SAEType = Field(default=SAEType.JUMP_RELU, description="Type of SAE (automatically set to jump_relu)")
    target_l0: float = Field(..., description="Target L0 sparsity (number of active features per sample)")
    bandwidth: float = Field(0.001, description="Bandwidth for JumpReLU gradient approximation (paper uses 0.001)")
    initial_threshold: float = Field(0.001, description="Initial threshold for JumpReLU activation")
    use_pre_enc_bias: bool = Field(True, description="Whether to subtract decoder bias before encoding")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    
    # Activation normalization (recommended for stable training)
    normalize_activations: bool = Field(True, description="Normalize inputs to unit mean squared norm (recommended)")
    
    # Sparsity warmup (critical for fast L0 convergence!)
    sparsity_warmup_steps: int | None = Field(5000, description="Number of gradient steps to warmup sparsity penalty (dictionary_learning default: 5000)")
    
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
    
    When normalize_activations=True:
    - input: normalized input (for loss computation)
    - output: normalized reconstruction (for loss computation)
    - output_raw: denormalized reconstruction (for external use/evaluation)
    
    When normalize_activations=False:
    - input: raw input
    - output: raw reconstruction
    - output_raw: same as output
    """
    pre_activations: Float[torch.Tensor, "... c"]  # pre-JumpReLU activations
    l0: Float[torch.Tensor, "..."]  # L0 norm per sample
    output_raw: Float[torch.Tensor, "... dim"]  # denormalized reconstruction for external use
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents


class JumpReLUSAE(BaseSAE):
    """
    JumpReLU Sparse Autoencoder (paper-consistent implementation):
      - Linear encoder/decoder with bias
      - JumpReLU activation with learned thresholds
      - SSE reconstruction loss in normalized space (sum over dims, mean over batch)
      - Mean-L0 sparsity loss ((mean_l0 / target - 1)²) for faster convergence
      - Automatic input normalization for consistent bandwidth behavior
    """
    
    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        target_l0: float,
        bandwidth: float = 0.001,
        initial_threshold: float = 0.001,
        use_pre_enc_bias: bool = False,
        normalize_activations: bool = True,
        sparsity_warmup_steps: int | None = 5000,
        sparsity_coeff: float | None = None,
        mse_coeff: float | None = None,
        dead_toks_threshold: int | None = None,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
    ):
        """
        Args:
            input_size: Dimensionality of inputs.
            n_dict_components: Number of dictionary features.
            target_l0: Target L0 sparsity (number of active features per sample).
            bandwidth: Bandwidth for JumpReLU gradient approximation (paper uses 0.001).
            initial_threshold: Initial threshold for JumpReLU activation.
            use_pre_enc_bias: Whether to subtract decoder bias before encoding.
            normalize_activations: Normalize inputs to unit mean squared norm (recommended).
            sparsity_warmup_steps: Number of gradient steps to warmup sparsity penalty.
                Critical for convergence! dictionary_learning uses 5000 steps.
            sparsity_coeff: Coefficient for target L0 regularization loss.
            mse_coeff: Coefficient on reconstruction loss.
            dead_toks_threshold: Threshold for dead feature tracking.
            aux_k: If provided (>0), number of auxiliary features from the inactive set.
            aux_coeff: Coefficient on the auxiliary reconstruction loss.
            init_decoder_orthogonal: Initialize decoder weights to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
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
        self.normalize_activations = normalize_activations
        self.sparsity_warmup_steps = sparsity_warmup_steps if sparsity_warmup_steps is not None else 0
        
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
        
        # Activation normalization buffers
        # norm_factor = sqrt(E[||x||²]) so that normalized x has unit mean squared norm
        self.register_buffer("running_norm_factor", torch.tensor(1.0))
        self.register_buffer("norm_factor_initialized", torch.tensor(False))
        
        # Sparsity warmup step counter (updated via on_after_optimizer_step)
        self.register_buffer("sparsity_step", torch.tensor(0, dtype=torch.long))
    
    def _compute_norm_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute or update the norm factor for activation normalization.
        
        The norm factor is sqrt(E[||x||²]) so that x/norm_factor has unit mean squared norm.
        
        IMPORTANT: Even when an external accumulator is refining the norm_factor over multiple
        batches, we MUST use a reasonable estimate from the first batch. Otherwise, the
        threshold will learn wrong values during the accumulation period (since threshold 0.001
        is calibrated for normalized activations, not raw activations with magnitude ~50).
        
        Args:
            x: Input tensor with shape (..., input_size)
            
        Returns:
            The norm factor to use for this forward pass
        """
        # Check if we've computed an initial estimate (even if not "fully initialized")
        has_initial_estimate = getattr(self, '_has_initial_norm_estimate', False)
        
        if self.training and not self.norm_factor_initialized and not has_initial_estimate:
            with torch.no_grad():
                # Compute mean squared norm: E[||x||²]
                # Flatten batch dimensions for computation
                x_flat = x.reshape(-1, x.shape[-1])
                mean_squared_norm = (x_flat ** 2).sum(dim=-1).mean()
                norm_factor = mean_squared_norm.sqrt()
                
                self.running_norm_factor.copy_(norm_factor)
                self._has_initial_norm_estimate = True
                
                # Only mark as fully initialized if NOT using external accumulator
                # External accumulator will refine and mark initialized when ready
                defer_init = getattr(self, '_defer_norm_factor_init', False)
                if not defer_init:
                    self.norm_factor_initialized.fill_(True)
        
        return self.running_norm_factor
    
    def set_norm_factor(self, norm_factor: float) -> None:
        """Manually set the norm factor for activation normalization.
        
        This can be used to pre-compute norm_factor over multiple batches
        (matching dictionary_learning behavior which computes it once over 100 batches).
        
        Args:
            norm_factor: The norm factor to use (sqrt of mean squared activation norm)
        """
        with torch.no_grad():
            self.running_norm_factor.fill_(norm_factor)
            self.norm_factor_initialized.fill_(True)

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> JumpReLUSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        
        If normalize_activations=True, inputs are normalized to unit mean squared norm
        before encoding, and outputs are denormalized after decoding.
        """
        # Activation normalization (if enabled)
        if self.normalize_activations:
            norm_factor = self._compute_norm_factor(x)
            x_normalized = x / norm_factor
        else:
            norm_factor = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            x_normalized = x
        
        # Optional: subtract decoder bias before encoding
        if self.use_pre_enc_bias:
            x_enc = x_normalized - self.decoder_bias
        else:
            x_enc = x_normalized
        
        # Encode: linear transform + bias (no ReLU - matches SAE Bench for proper convergence)
        pre_activations = self.encoder(x_enc) + self.encoder_bias
        
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
        
        # Decode (in normalized space)
        x_hat_normalized = F.linear(feature_activations, self.dict_elements) + self.decoder_bias
        
        # Denormalize output for external use (evaluation, etc.)
        if self.normalize_activations:
            x_hat_raw = x_hat_normalized * norm_factor
        else:
            x_hat_raw = x_hat_normalized
        
        # Compute L0 for loss (using differentiable step function)
        l0 = StepFunction.apply(
            pre_activations, 
            self.jumprelu.threshold,
            self.bandwidth
        ).sum(dim=-1)
        
        # CRITICAL: Store normalized input/output for loss computation!
        # dictionary_learning computes loss in normalized space: both input and output
        # are scaled by 1/norm_factor. If we stored raw input and denormalized output,
        # the reconstruction loss would be scaled by norm_factor^2 (~2500x for Gemma-2),
        # completely breaking the balance between reconstruction and sparsity loss.
        return JumpReLUSAEOutput(
            input=x_normalized,       # normalized input for loss computation
            c=feature_activations,
            output=x_hat_normalized,  # normalized reconstruction for loss computation
            output_raw=x_hat_raw,     # denormalized reconstruction for external use
            logits=None,
            pre_activations=pre_activations,
            l0=l0,
            auxk_indices=auxk_indices,
            auxk_values=auxk_values,
        )
    
    def _get_sparsity_warmup_factor(self) -> float:
        """Get the current sparsity warmup factor (0 to 1).
        
        Linearly ramps sparsity penalty from 0 to 1 over sparsity_warmup_steps.
        This is critical for fast L0 convergence as it allows the threshold to
        grow before the full sparsity penalty kicks in.
        """
        if self.sparsity_warmup_steps <= 0:
            return 1.0
        return min(self.sparsity_step.item() / self.sparsity_warmup_steps, 1.0)
    
    def compute_loss(self, output: JumpReLUSAEOutput) -> SAELoss:
        """
        Loss computation:
        
        Loss = mse_coeff * SSE_norm + sparsity_coeff * (E_x[L0]/target - 1)² * warmup_factor
        
        - Reconstruction: SSE in normalized space (sum over dims, mean over batch)
        - Sparsity: Mean-L0 penalty ((mean_l0 / target - 1)²), scaled by warmup factor
        
        Note: output.input and output.output are already in normalized space.
        """
        # Reconstruction loss: SSE in normalized space (sum over dims, mean over batch)
        # This is the paper formulation: E_x[||x - x_hat||²]
        recon_sse = ((output.output - output.input) ** 2).sum(dim=-1).mean()
        
        # Sparsity warmup factor (critical for convergence!)
        sparsity_warmup_factor = self._get_sparsity_warmup_factor()
        
        # Sparsity loss: ((mean_l0 / target_l0) - 1)² * warmup_factor
        # Using mean-L0 penalty (not per-example) for faster convergence
        mean_l0 = output.l0.mean()
        sparsity_loss = self.sparsity_coeff * ((mean_l0 / self.target_l0) - 1.0) ** 2 * sparsity_warmup_factor
        
        # Total loss
        total_loss = self.mse_coeff * recon_sse + sparsity_loss
        
        # Compute band hit rate: fraction of (sample, feature) pairs in bandwidth window
        band_hit_rate = ((output.pre_activations - self.jumprelu.threshold).abs() < (self.bandwidth / 2)).float().mean()
        
        loss_dict: dict[str, torch.Tensor] = {
            "recon_loss": recon_sse.detach().clone(),
            "sparsity_loss": sparsity_loss.detach().clone(),
            "l0_norm": mean_l0.detach().clone(),
            "band_hit_rate": band_hit_rate.detach().clone(),
            "sparsity_warmup_factor": torch.tensor(sparsity_warmup_factor),
        }
        
        # Optional auxiliary dead-feature loss (computed in normalized space)
        weighted_aux_loss, aux_loss_for_logging = compute_aux_loss_with_logging(
            auxk_indices=output.auxk_indices,
            auxk_values=output.auxk_values,
            input_tensor=output.input,   # Use normalized input
            output_tensor=output.output,  # Use normalized output
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
    
    def on_before_optimizer_step(self) -> None:
        """Remove gradient component parallel to decoder directions (SAE Bench style).
        
        This projects the decoder gradient onto the tangent space of the unit sphere,
        which is orthogonal to the decoder column directions. This prevents the 
        optimizer from changing the norm of decoder columns (only the direction).
        """
        if self.decoder.weight.grad is None:
            return
        
        # decoder.weight has shape (input_size, n_dict_components)
        # For each column (dictionary element), remove gradient parallel to that column
        with torch.no_grad():
            dec = self.decoder.weight.data  # (input_size, n_dict_components)
            grad = self.decoder.weight.grad  # (input_size, n_dict_components)
            
            # Normalize decoder columns
            dec_normed = F.normalize(dec, dim=0)  # (input_size, n_dict_components)
            
            # Project out the parallel component for each column
            # parallel_component[i] = (grad[:, i] · dec_normed[:, i]) * dec_normed[:, i]
            # This is done vectorized: sum over input_size, keep n_dict_components
            parallel_coeff = (grad * dec_normed).sum(dim=0, keepdim=True)  # (1, n_dict_components)
            parallel_component = parallel_coeff * dec_normed  # (input_size, n_dict_components)
            
            # Remove parallel component from gradient
            self.decoder.weight.grad.sub_(parallel_component)
    
    def on_after_optimizer_step(self) -> None:
        """Normalize decoder weights and update step counter (SAE Bench style).
        
        After each optimizer step:
        1. Project decoder columns back to unit norm
        2. Increment sparsity warmup step counter
        """
        # decoder.weight has shape (input_size, n_dict_components)
        # Normalize each column to unit norm
        eps = torch.finfo(self.decoder.weight.dtype).eps
        norm = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data.div_(norm.clamp(min=eps))
        
        # Increment sparsity warmup step counter
        self.sparsity_step += 1
