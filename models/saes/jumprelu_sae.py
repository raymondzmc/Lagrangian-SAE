# jumprelu_sae.py
"""
JumpReLU SAE implementation following SAE Bench style.

Key behaviors:
1. Reconstruction loss: SSE (sum over dims, mean over batch)
2. Sparsity loss: coeff * ((L0 / target_l0) - 1)^2
3. Raw threshold parameterization (not log-threshold)
4. Optional activation normalization for hyperparameter transfer across models/layers
5. Optional auxiliary loss for dead feature mitigation

Reference: SAE Bench implementation (dictionary_learning)

Activation Normalization:
When normalize_activations=True (recommended), inputs are normalized to have unit mean 
squared norm before encoding. This makes hyperparameters like threshold (0.001) and 
bandwidth (0.001) meaningful regardless of the model's activation magnitude. Without 
normalization, these values may be too small compared to large activation magnitudes,
causing extremely high L0 values initially.
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, model_validator
from jaxtyping import Float

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.activations import StepFunction, JumpReLU
from models.saes.utils import (
    init_decoder_orthogonal_cuda,
    update_dead_feature_stats,
    maybe_compute_auxk_features,
    compute_aux_loss_with_logging,
)
from utils.enums import SAEType


class JumpReLUSAEConfig(SAEConfig):
    """
    Config for JumpReLU SAE (SAE Bench style).
    
    Loss formulation:
    - Reconstruction: SSE (sum over dims, mean over batch)
    - Sparsity: sparsity_coeff * ((L0 / target_l0) - 1)² * sparsity_warmup_factor
    
    Notes:
    - Uses JumpReLU activation with learned thresholds
    - Bandwidth parameter controls gradient smoothness for threshold learning
    - Raw threshold parameterization (not log-threshold)
    - normalize_activations=True is recommended for stable training
    - sparsity_warmup_steps gradually increases sparsity penalty (critical for convergence!)
    """
    sae_type: SAEType = Field(default=SAEType.JUMP_RELU, description="Type of SAE (automatically set to jump_relu)")
    target_l0: float = Field(..., description="Target L0 sparsity (number of active features per sample)")
    bandwidth: float = Field(0.001, description="Bandwidth for JumpReLU gradient approximation (paper uses 0.001)")
    initial_threshold: float = Field(0.001, description="Initial threshold for JumpReLU activation")
    use_pre_enc_bias: bool = Field(False, description="Whether to subtract decoder bias before encoding (SAE Bench default: False)")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    
    # Activation normalization (recommended for stable training)
    normalize_activations: bool = Field(True, description="Normalize inputs to unit mean squared norm (recommended)")
    norm_factor_num_batches: int = Field(100, description="Number of batches to compute norm factor over before training (dictionary_learning uses 100)")
    
    # Sparsity warmup (critical for fast L0 convergence!)
    # dictionary_learning uses 5000 steps = 5000 * 2048 = 10.24M samples
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
    JumpReLU Sparse Autoencoder (SAE Bench style implementation):
      - Linear encoder/decoder with bias
      - JumpReLU activation with learned thresholds (raw parameterization)
      - SSE reconstruction loss (sum over dims, mean over batch)
      - Sparsity loss: coeff * ((L0 / target_l0) - 1)²
      - Optional activation normalization for stable training (recommended)
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
        norm_factor_num_batches: int = 100,
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
            norm_factor_num_batches: Number of batches to compute norm factor over before
                training starts (dictionary_learning uses 100).
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
        self.norm_factor_num_batches = norm_factor_num_batches
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
        
        # JumpReLU activation with learned thresholds
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
    
    @torch.no_grad()
    def compute_norm_factor_from_data(
        self, 
        data_iterator, 
        num_batches: int = 100,
        device: torch.device | str | None = None
    ) -> float:
        """Compute norm factor from data iterator (matching dictionary_learning).
        
        dictionary_learning computes norm_factor once at the start over 100 batches:
        norm_factor = sqrt(average of mean_squared_norm over batches)
        
        Args:
            data_iterator: Iterator yielding activation batches
            num_batches: Number of batches to use for estimation (default: 100)
            device: Device to use for computation
            
        Returns:
            The computed norm factor
        """
        if device is None:
            device = self.device
            
        total_mean_squared_norm = 0.0
        count = 0
        
        for i, batch in enumerate(data_iterator):
            if i >= num_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch, dict):
                # Assume activations are stored under a key
                act = batch.get('activations', batch.get('input', None))
                if act is None:
                    raise ValueError("Cannot find activations in batch dict")
            elif isinstance(batch, (tuple, list)):
                act = batch[0]
            else:
                act = batch
            
            act = act.to(device)
            act_flat = act.reshape(-1, act.shape[-1])
            mean_squared_norm = (act_flat ** 2).sum(dim=-1).mean()
            total_mean_squared_norm += mean_squared_norm.item()
            count += 1
        
        if count == 0:
            raise ValueError("No batches processed for norm factor computation")
        
        average_mean_squared_norm = total_mean_squared_norm / count
        norm_factor = average_mean_squared_norm ** 0.5
        
        # Set the norm factor
        self.set_norm_factor(norm_factor)
        
        return norm_factor
    
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
        
        # Encode: linear transform + bias
        pre_activations = self.encoder(x_enc) + self.encoder_bias
        
        # Apply JumpReLU activation (with raw threshold)
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
        
        # Decode in normalized space
        x_hat_normalized = F.linear(feature_activations, self.dict_elements, bias=self.decoder_bias)
        
        # Denormalize output for external use (evaluation, etc.)
        if self.normalize_activations:
            x_hat_raw = x_hat_normalized * norm_factor
        else:
            x_hat_raw = x_hat_normalized

        l0 = StepFunction.apply(
            feature_activations, 
            self.jumprelu.threshold,
            self.bandwidth
        ).sum(dim=-1)

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
        Loss computation (SAE Bench style):
        
        Loss = mse_coeff * SSE + sparsity_coeff * ((L0 / target_l0) - 1)² * warmup_factor
        
        - Reconstruction: SSE (sum over dims, mean over batch)
        - Sparsity: Per-batch L0 penalty with warmup
        """
        # Reconstruction loss: SSE (sum over dims, mean over batch)
        recon_sse = ((output.output - output.input) ** 2).sum(dim=-1).mean()
        
        # Sparsity warmup factor (critical for convergence!)
        sparsity_warmup_factor = self._get_sparsity_warmup_factor()
        
        # Sparsity loss: ((mean_l0 / target_l0) - 1)² * warmup_factor
        mean_l0 = output.l0.mean()
        sparsity_loss = self.sparsity_coeff * ((mean_l0 / self.target_l0) - 1.0) ** 2 * sparsity_warmup_factor
        
        # Total loss
        total_loss = self.mse_coeff * recon_sse + sparsity_loss
        
        loss_dict: dict[str, torch.Tensor] = {
            "recon_loss": recon_sse.detach().clone(),
            "sparsity_loss": sparsity_loss.detach().clone(),
            "l0_norm": mean_l0.detach().clone(),
            "sparsity_warmup_factor": torch.tensor(sparsity_warmup_factor),
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
        """Decoder weight matrix (SAE Bench style: raw weights, normalized post-step)."""
        return self.decoder.weight
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def on_before_optimizer_step(self) -> None:
        """Remove gradient component parallel to decoder directions (SAE Bench style).
        
        This projects the decoder gradient onto the tangent space of the unit sphere,
        which is orthogonal to the decoder column directions. This prevents the 
        optimizer from changing the norm of decoder columns (only the direction).
        """
        if self.decoder.weight.grad is None:
            return
        
        # decoder.weight has shape (input_size, n_dict_components)
        W_dec = self.decoder.weight.data  # (D, F)
        W_dec_grad = self.decoder.weight.grad  # (D, F)
        
        # Normalize decoder columns
        normed_W_dec = W_dec / (torch.norm(W_dec, dim=0, keepdim=True) + 1e-6)
        
        # Compute parallel component: dot product of grad with normalized decoder
        # for each column (feature)
        parallel_component = (W_dec_grad * normed_W_dec).sum(dim=0, keepdim=True)  # (1, F)
        
        # Remove parallel component from gradient
        self.decoder.weight.grad -= parallel_component * normed_W_dec
    
    @torch.no_grad()
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
        self.decoder.weight.data /= norm + eps
        
        # Increment sparsity warmup step counter
        self.sparsity_step += 1
    
    def scale_biases(self, scale: float) -> None:
        """Scale biases and thresholds by a factor.
        
        This is used when saving a model that was trained with normalized activations.
        When normalize_activations=True, the model operates in normalized space internally.
        To save a model that works with raw (unnormalized) activations at inference time,
        call scale_biases(norm_factor) before saving.
        
        Args:
            scale: Factor to multiply biases and thresholds by.
        """
        with torch.no_grad():
            self.decoder_bias.data *= scale
            self.encoder_bias.data *= scale
            self.jumprelu.threshold.data *= scale
    
    def get_denormalized_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict with biases scaled for use with raw activations.
        
        If normalize_activations=True and norm_factor has been computed,
        returns a state dict with biases scaled by norm_factor so the saved
        model works with raw (unnormalized) activations at inference time.
        
        Returns:
            State dict with properly scaled biases.
        """
        state_dict = {k: v.clone() for k, v in self.state_dict().items()}
        
        if self.normalize_activations and self.norm_factor_initialized:
            norm_factor = self.running_norm_factor.item()
            state_dict['decoder_bias'] *= norm_factor
            state_dict['encoder_bias'] *= norm_factor
            state_dict['jumprelu.threshold'] *= norm_factor
            # Reset norm factor tracking for the saved model
            state_dict['running_norm_factor'] = torch.tensor(1.0)
            state_dict['norm_factor_initialized'] = torch.tensor(False)
        
        return state_dict
