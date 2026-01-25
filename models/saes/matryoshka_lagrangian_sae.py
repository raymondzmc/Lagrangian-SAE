# matryoshka_lagrangian_sae.py
"""
Matryoshka Lagrangian SAE - Combines Matryoshka's progressive group structure
with LagrangianSAE's adaptive sparsity control.

Key features:
- Progressive reconstruction loss at each group boundary (Matryoshka)
- JumpReLU activation with per-feature learned thresholds (Lagrangian)
- Dual ascent on alpha to achieve target L0 sparsity (Lagrangian)
- Differentiable L0 approximation for gradient computation
- Running mean (EMA) of L0 for stable constraint evaluation

============================================================================
IMPORTANT CAVEATS FOR COMPARISONS
============================================================================

When comparing MatryoshkaLagrangianSAE with other SAE implementations, be aware
of the following architectural differences that prevent direct apple-to-apple
comparisons:

1. ENCODER PRE-ACTIVATION (vs LagrangianSAE)
   - MatryoshkaLagrangianSAE: Applies ReLU BEFORE JumpReLU
     preacts = F.relu(self.encoder(x_enc) + self.encoder_bias)
   - LagrangianSAE: No ReLU, passes raw encoder outputs to JumpReLU
     preacts = self.encoder(x_enc) + self.encoder_bias
   
   Impact: JumpReLU thresholds have different semantics. With ReLU, all preacts
   are non-negative, changing how threshold values affect sparsity. The same
   threshold value will yield different L0 levels between implementations.

2. NORMALIZATION APPROACH (vs MatryoshkaSAE)
   - MatryoshkaLagrangianSAE: Uses `normalize_activations` (scale-only)
     x_normalized = x / norm_factor where norm_factor = sqrt(E[||x||²])
   - MatryoshkaSAE: Uses `input_unit_norm` (mean + std normalization)
     x_normalized = (x - mean) / std
   
   Impact: These are semantically different normalizations. MatryoshkaLagrangianSAE
   follows LagrangianSAE's approach, not MatryoshkaSAE's.

3. SPARSITY MECHANISM (vs MatryoshkaSAE)
   - MatryoshkaLagrangianSAE: Soft sparsity via JumpReLU + Lagrangian control
     L0 varies per sample, targets average L0 ≈ target_l0
   - MatryoshkaSAE: Hard sparsity via BatchTopK
     Guarantees exactly k * batch_size total active features
   
   Impact: Direct L0/MSE metric comparisons are not meaningful. 
   MatryoshkaLagrangianSAE may have samples with L0 >> target or L0 << target.

4. ENCODER BIAS (vs MatryoshkaSAE)
   - MatryoshkaLagrangianSAE: Has encoder_bias parameter (learned)
   - MatryoshkaSAE: No separate encoder bias (only decoder_bias for centering)
   
   Impact: Different parameter counts and optimization dynamics.

5. LOSS COMPOSITION
   - MatryoshkaLagrangianSAE: MSE + Lagrangian penalty + quadratic penalty
     Loss = Σ(group_weight[i] * MSE_i) + α * (L0 - target) + ρ/2 * (L0 - target)²
   - MatryoshkaSAE: Progressive MSE only
     Loss = Σ(group_weight[i] * MSE_i)
   
   Impact: Different optimization landscapes. The Lagrangian terms affect
   how the model balances reconstruction vs sparsity.

============================================================================
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
from models.saes.activations import StepFunction, JumpReLU
from utils.enums import SAEType


class MatryoshkaLagrangianSAEConfig(SAEConfig):
    """
    Config for Matryoshka Lagrangian SAE.

    Combines Matryoshka's progressive group structure with LagrangianSAE's
    adaptive sparsity control via dual ascent.
    
    Notes:
    - Features are divided into groups via group_fractions (must sum to 1.0)
    - Loss is computed progressively at each group boundary
    - Uses JumpReLU activation with per-feature learned thresholds
    - Dual ascent on alpha to achieve target L0 sparsity
    """
    sae_type: SAEType = Field(
        default=SAEType.MATRYOSHKA_LAGRANGIAN, 
        description="Type of SAE (automatically set to matryoshka_lagrangian)"
    )
    
    # Matryoshka group configuration
    group_fractions: list[float] = Field(
        default=[1/32, 1/16, 1/8, 1/4, 1/2 + 1/32],
        description="Fractions of dict_size for each group (must sum to 1.0)"
    )
    group_weights: list[float] | None = Field(
        None, 
        description="Weights for loss at each group boundary (default: uniform)"
    )
    
    # Lagrangian sparsity control
    target_l0: float = Field(..., description="Target L0 sparsity (number of active features per sample)")
    initial_alpha: float = Field(0.0, description="Initial alpha for Lagrangian multiplier")
    alpha_lr: float = Field(1e-2, description="Learning rate for alpha dual ascent")
    alpha_max: float = Field(100.0, description="Maximum alpha when L0 > target (upper bound)")
    alpha_min: float | None = Field(None, description="Maximum alpha magnitude when L0 < target (lower bound magnitude). If None, uses alpha_max for symmetric bounds.")
    rho_quadratic: float = Field(0.1, description="Coefficient for quadratic penalty in augmented Lagrangian")
    l0_ema_momentum: float = Field(0.99, description="Momentum for running mean of L0")
    equality_constraint: bool = Field(True, description="If True, enforce L0 ≈ target; if False, enforce L0 <= target")
    
    # JumpReLU threshold parameters
    initial_threshold: float = Field(0.5, description="Initial per-feature threshold value")
    bandwidth: float = Field(0.5, description="Bandwidth for step function gradient approximation")
    calibrate_thresholds: bool = Field(True, description="Auto-calibrate thresholds on first batch")
    
    # Standard SAE parameters
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    use_pre_enc_bias: bool = Field(False, description="Subtract decoder bias before encoding")
    
    # Activation normalization (recommended for consistent behavior across layers)
    # Uses same naming convention as JumpReLUSAE for compatibility with run.py pre-computation
    normalize_activations: bool = Field(False, description="Normalize inputs to unit mean squared norm (recommended for stable training)")
    norm_factor_num_batches: int = Field(100, description="Number of batches to compute norm factor over before training (dictionary_learning uses 100)")
    
    # Dead feature tracking and auxiliary loss
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead")
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.MATRYOSHKA_LAGRANGIAN
        return values

    @model_validator(mode="after")
    def validate_group_fractions(self) -> "MatryoshkaLagrangianSAEConfig":
        """Validate that group_fractions sum to 1.0."""
        if not isclose(sum(self.group_fractions), 1.0, rel_tol=1e-5):
            raise ValueError(f"group_fractions must sum to 1.0, got {sum(self.group_fractions)}")
        if any(f <= 0 for f in self.group_fractions):
            raise ValueError("All group_fractions must be positive")
        return self

    @model_validator(mode="after")
    def validate_group_weights(self) -> "MatryoshkaLagrangianSAEConfig":
        """Validate group_weights if provided."""
        if self.group_weights is not None:
            if len(self.group_weights) != len(self.group_fractions):
                raise ValueError(
                    f"group_weights length ({len(self.group_weights)}) must match "
                    f"group_fractions length ({len(self.group_fractions)})"
                )
        return self


class MatryoshkaLagrangianSAEOutput(SAEOutput):
    """
    Matryoshka Lagrangian SAE output combining both paradigms.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder outputs (after ReLU, before JumpReLU)
    l0: Float[torch.Tensor, "..."]  # L0 norm per sample (actual count)
    l0_differentiable: Float[torch.Tensor, "..."]  # Differentiable L0 for gradients
    alpha: torch.Tensor  # current Lagrangian multiplier value
    auxk_indices: torch.Tensor | None = None
    auxk_values: torch.Tensor | None = None


class MatryoshkaLagrangianSAE(BaseSAE):
    """
    Matryoshka Lagrangian Sparse Autoencoder:
      - Linear encoder/decoder with biases
      - JumpReLU activation with per-feature learned thresholds
      - Dual ascent on alpha to achieve target sparsity
      - Progressive MSE loss at each group boundary
      - Optional auxiliary loss for dead features
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        target_l0: float,
        group_fractions: list[float] | None = None,
        group_weights: list[float] | None = None,
        initial_alpha: float = 0.0,
        alpha_lr: float = 1e-2,
        alpha_max: float = 100.0,
        alpha_min: float | None = None,
        rho_quadratic: float = 0.1,
        l0_ema_momentum: float = 0.99,
        equality_constraint: bool = True,
        initial_threshold: float = 0.5,
        bandwidth: float = 0.5,
        calibrate_thresholds: bool = True,
        sparsity_coeff: float | None = None,  # unused; kept for API parity
        mse_coeff: float | None = None,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        dead_toks_threshold: int | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        use_pre_enc_bias: bool = False,
        normalize_activations: bool = False,
        norm_factor_num_batches: int = 100,
    ):
        """
        Args:
            input_size: Dimensionality of inputs (e.g., residual stream width).
            n_dict_components: Number of dictionary features (latent size).
            target_l0: Target number of active features per sample.
            group_fractions: Fractions of dict_size for each group (must sum to 1.0).
                Default: [1/32, 1/16, 1/8, 1/4, 1/2 + 1/32] (5 groups).
            group_weights: Weights for loss at each group boundary (default: uniform).
            initial_alpha: Initial value for Lagrangian multiplier.
            alpha_lr: Learning rate for alpha dual ascent.
            alpha_max: Maximum alpha when L0 > target (upper bound).
            alpha_min: Maximum alpha magnitude when L0 < target. If None, uses alpha_max.
            rho_quadratic: Coefficient for quadratic penalty in augmented Lagrangian.
            l0_ema_momentum: Momentum for running mean of L0.
            equality_constraint: If True, enforce L0 ≈ target; if False, L0 <= target.
            initial_threshold: Initial per-feature threshold value.
            bandwidth: Bandwidth for step function gradient approximation.
            calibrate_thresholds: Auto-calibrate thresholds on first batch.
            sparsity_coeff: Unused (present for interface compatibility).
            mse_coeff: Coefficient on MSE reconstruction loss (default 1.0).
            aux_k: Number of auxiliary features for dead latents.
            aux_coeff: Coefficient on the auxiliary reconstruction loss.
            dead_toks_threshold: Threshold for considering a feature as dead.
            init_decoder_orthogonal: Initialize decoder weight columns to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
            use_pre_enc_bias: Whether to subtract decoder bias before encoding.
            normalize_activations: Normalize inputs to unit mean squared norm (recommended for stable training).
            norm_factor_num_batches: Number of batches to compute norm factor over before training.
        """
        super().__init__()
        assert target_l0 > 0, "target_l0 must be positive"
        assert n_dict_components > 0 and input_size > 0
        assert bandwidth > 0, "bandwidth must be positive"
        assert 0.0 <= l0_ema_momentum < 1.0, "l0_ema_momentum must be in [0, 1)"

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.target_l0 = target_l0
        self.alpha_lr = alpha_lr
        self.alpha_max = alpha_max
        # alpha_min defaults to alpha_max for symmetric bounds; set smaller for asymmetric "soft inequality"
        self.alpha_min = alpha_min if alpha_min is not None else alpha_max
        self.rho_quadratic = rho_quadratic
        self.l0_ema_momentum = l0_ema_momentum
        self.bandwidth = bandwidth
        self.use_pre_enc_bias = use_pre_enc_bias
        self.equality_constraint = equality_constraint
        self.normalize_activations = normalize_activations
        self.norm_factor_num_batches = norm_factor_num_batches

        # Set default group fractions if not provided
        if group_fractions is None:
            group_fractions = [1/32, 1/16, 1/8, 1/4, 1/2 + 1/32]

        # Validate group fractions
        assert isclose(sum(group_fractions), 1.0, rel_tol=1e-5), "group_fractions must sum to 1.0"
        assert all(f > 0 for f in group_fractions), "all group_fractions must be positive"

        # Compute group sizes from fractions
        group_sizes = [int(f * n_dict_components) for f in group_fractions[:-1]]
        group_sizes.append(n_dict_components - sum(group_sizes))
        
        self.num_groups = len(group_sizes)
        self.register_buffer("group_sizes", torch.tensor(group_sizes, dtype=torch.long))
        
        # Compute cumulative group boundaries
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
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None
        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0

        # Lagrangian multiplier (alpha) as a buffer
        self.register_buffer("alpha", torch.tensor(initial_alpha, dtype=torch.float32))
        
        # Running mean of L0 for stable constraint evaluation (EMA)
        self.register_buffer("running_l0", torch.tensor(target_l0, dtype=torch.float32))

        # Biases
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        self.encoder_bias = nn.Parameter(torch.zeros(n_dict_components))

        # Linear maps (no bias in linear layers - biases are separate)
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        
        # JumpReLU activation with per-feature learned thresholds
        self.jumprelu = JumpReLU(
            feature_size=n_dict_components,
            bandwidth=bandwidth,
            initial_threshold=initial_threshold,
        )

        # Initialize decoder, then (optionally) tie encoder init to decoder^T
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
        
        # Running statistics for activation normalization (if enabled)
        # Uses scale-only normalization: x_norm = x / norm_factor where norm_factor = sqrt(E[||x||²])
        # Uses same naming convention as JumpReLUSAE for compatibility with run.py pre-computation
        if self.normalize_activations:
            self.register_buffer("running_norm_factor", torch.tensor(1.0))
            self.register_buffer("norm_factor_initialized", torch.tensor(False))
        
        # Threshold calibration state
        self.calibrate_thresholds = calibrate_thresholds
        if self.calibrate_thresholds:
            self.register_buffer("thresholds_calibrated", torch.tensor(False))

    @torch.no_grad()
    def _calibrate_thresholds_from_preacts(self, preacts: torch.Tensor) -> None:
        """
        Calibrate thresholds based on pre-activation statistics.
        
        Sets per-feature thresholds such that the expected L0 ≈ target_l0.
        """
        flat_preacts = preacts.reshape(-1, self.n_dict_components)
        n_samples = flat_preacts.shape[0]
        
        # Target: each feature should activate with probability target_l0 / n_dict_components
        target_activation_prob = self.target_l0 / self.n_dict_components
        target_percentile = 1.0 - target_activation_prob
        
        # Compute per-feature threshold at target percentile
        sorted_preacts, _ = torch.sort(flat_preacts, dim=0)
        percentile_idx = int(target_percentile * n_samples)
        percentile_idx = min(percentile_idx, n_samples - 1)
        
        threshold_values = sorted_preacts[percentile_idx, :]
        threshold_values = threshold_values.clamp(min=1e-4)
        
        self.jumprelu.threshold.data.copy_(threshold_values)
        self.thresholds_calibrated.fill_(True)
        
        mean_threshold = threshold_values.mean().item()
        min_threshold = threshold_values.min().item()
        max_threshold = threshold_values.max().item()
        print(f"[MatryoshkaLagrangianSAE] Thresholds calibrated: mean={mean_threshold:.4f}, "
              f"min={min_threshold:.4f}, max={max_threshold:.4f}")

    def _compute_norm_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute or update the norm factor for activation normalization.
        
        The norm factor is sqrt(E[||x||²]) so that x/norm_factor has unit mean squared norm.
        
        IMPORTANT: Even when an external accumulator is refining the norm_factor over multiple
        batches, we MUST use a reasonable estimate from the first batch. Otherwise, the
        threshold will learn wrong values during the accumulation period (since threshold 0.001
        is calibrated for normalized activations, not raw activations with magnitude ~50).
        
        This method matches JumpReLUSAE's _compute_norm_factor for compatibility with run.py.
        
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
        
        This method is called by run.py's compute_norm_factors_for_saes() function.
        
        Args:
            norm_factor: The norm factor to use (sqrt of mean squared activation norm)
        """
        with torch.no_grad():
            self.running_norm_factor.fill_(norm_factor)
            self.norm_factor_initialized.fill_(True)

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> MatryoshkaLagrangianSAEOutput:
        """
        Forward pass with JumpReLU activation and learned thresholds.
        
        If normalize_activations is enabled:
        - Input is scaled to achieve mean squared L2 norm ≈ 1 (preserves mean/direction)
        - SAE operates in normalized space (encoding, thresholding, decoding)
        - Output is denormalized back to original scale
        """
        # Activation normalization (if enabled)
        if self.normalize_activations:
            norm_factor = self._compute_norm_factor(x)
            x_normalized = x / norm_factor
        else:
            norm_factor = None
            x_normalized = x
        
        # Optional: subtract decoder bias before encoding
        if self.use_pre_enc_bias:
            x_enc = x_normalized - self.decoder_bias
        else:
            x_enc = x_normalized

        # Encoder with ReLU pre-activation
        preacts = F.relu(self.encoder(x_enc) + self.encoder_bias)
        
        # Threshold calibration on first batch
        if self.calibrate_thresholds and self.training and not self.thresholds_calibrated:
            with torch.no_grad():
                self._calibrate_thresholds_from_preacts(preacts)
        
        # Apply JumpReLU with per-feature learned thresholds
        c = self.jumprelu(preacts)

        # Update dead latent statistics
        update_dead_feature_stats(
            activations=c,
            stats_last_nonzero=self.stats_last_nonzero,
            training=self.training,
            dead_toks_threshold=self.dead_toks_threshold,
        )

        # Compute auxiliary top-k for dead latents
        auxk_values, auxk_indices = maybe_compute_auxk_features(
            preacts=preacts,
            stats_last_nonzero=self.stats_last_nonzero,
            aux_k=self.aux_k,
            aux_coeff=self.aux_coeff,
            dead_toks_threshold=self.dead_toks_threshold,
        )

        # Decode using normalized dictionary elements + bias (in normalized space)
        x_hat_normalized = F.linear(c, self.dict_elements, bias=self.decoder_bias)
        
        # Denormalize output back to original scale (scale-only: just multiply)
        if self.normalize_activations and norm_factor is not None:
            x_hat = x_hat_normalized * norm_factor
        else:
            x_hat = x_hat_normalized

        # Compute true L0
        l0 = (c > 0).float().sum(dim=-1)
        
        # Compute differentiable L0 using StepFunction
        l0_differentiable = StepFunction.apply(
            preacts, 
            self.jumprelu.threshold, 
            self.bandwidth
        ).sum(dim=-1)

        return MatryoshkaLagrangianSAEOutput(
            input=x,
            c=c,
            output=x_hat,
            logits=None,
            preacts=preacts,
            l0=l0,
            l0_differentiable=l0_differentiable,
            alpha=self.alpha,
            auxk_indices=auxk_indices,
            auxk_values=auxk_values,
        )

    def compute_loss(self, output: MatryoshkaLagrangianSAEOutput) -> SAELoss:
        """
        Computes the combined Matryoshka progressive loss + Lagrangian sparsity penalty.
        
        Loss = Σ(group_weight[i] * MSE_i) + α * (L0 - target) + ρ/2 * (L0 - target)²
        """
        # Flatten for loss computation
        input_for_loss = output.input.reshape(-1, self.input_size)
        acts = output.c.reshape(-1, self.n_dict_components)
        
        # Get normalized decoder weights
        dict_elements = self.dict_elements
        
        # Progressive reconstruction at each group boundary (Matryoshka)
        x_reconstruct = self.decoder_bias.unsqueeze(0).expand(acts.shape[0], -1).clone()
        
        total_mse_loss = torch.tensor(0.0, device=acts.device, dtype=acts.dtype)
        group_losses = []
        
        for i in range(self.num_groups):
            group_start = self.group_boundaries[i].item()
            group_end = self.group_boundaries[i + 1].item()
            
            acts_slice = acts[:, group_start:group_end]
            W_dec_slice = dict_elements[:, group_start:group_end]
            
            x_reconstruct = x_reconstruct + F.linear(acts_slice, W_dec_slice)
            
            group_mse = F.mse_loss(x_reconstruct, input_for_loss)
            group_losses.append(group_mse.detach().clone())
            
            total_mse_loss = total_mse_loss + self.group_weights[i] * group_mse
        
        # Compute mean L0 across batch
        mean_l0 = output.l0.mean()
        
        # Update running L0 (EMA)
        if self.training:
            with torch.no_grad():
                # NOTE: Compute in float32 to avoid bfloat16 precision issues where EMA
                # updates can get lost due to limited mantissa bits (e.g., 74.5 is a "sticky"
                # value in bfloat16 where small updates round back to the same value)
                running_l0_f32 = self.running_l0.float()
                running_l0_f32 = (
                    self.l0_ema_momentum * running_l0_f32
                    + (1.0 - self.l0_ema_momentum) * mean_l0.detach().float()
                )
                self.running_l0.copy_(running_l0_f32)
        
        # Store constraint violation for dual ascent
        running_constraint_violation = self.running_l0 - self.target_l0
        self._last_constraint_violation = running_constraint_violation.detach().clone()
        
        # Differentiable L0 for gradient computation
        mean_l0_diff = output.l0_differentiable.mean()
        differentiable_violation = mean_l0_diff - self.target_l0
        
        # Get current alpha value (detached)
        alpha_value = self.alpha.detach().clone()
        
        # Compute normalized violation based on constraint type
        if self.equality_constraint:
            normalized_violation = differentiable_violation / self.target_l0
        else:
            normalized_violation = F.relu(differentiable_violation) / self.target_l0
        
        # Augmented Lagrangian sparsity penalty
        sparsity_loss = alpha_value * normalized_violation
        quadratic_penalty = self.rho_quadratic * (normalized_violation ** 2)
        
        # Total loss
        total_loss = self.mse_coeff * total_mse_loss + sparsity_loss + quadratic_penalty
        
        # Build loss dict
        mean_threshold = self.jumprelu.threshold.mean()
        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": total_mse_loss.detach().clone(),
            "running_l0": self.running_l0.detach().clone(),
            "alpha": alpha_value,
            "quadratic_penalty": quadratic_penalty.detach().clone(),
            "mean_threshold": mean_threshold.detach().clone(),
        }
        
        # Add individual group losses
        for i, gl in enumerate(group_losses):
            loss_dict[f"mse_loss_group_{i}"] = gl

        # Auxiliary dead-feature loss
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
    
    @torch.no_grad()
    def update_alpha(self) -> None:
        """
        Perform dual ascent update on alpha.
        
        Should be called after optimizer.step() to avoid autograd conflicts.
        
        Uses asymmetric bounds [-alpha_min, +alpha_max] for equality constraint.
        """
        if hasattr(self, '_last_constraint_violation'):
            self.alpha.add_(self.alpha_lr * self._last_constraint_violation)
            if self.equality_constraint:
                # Asymmetric bounds: [-alpha_min, +alpha_max]
                self.alpha.clamp_(min=-self.alpha_min, max=self.alpha_max)
            else:
                self.alpha.clamp_(min=0, max=self.alpha_max)

    @property
    def dict_elements(self) -> torch.Tensor:
        """Column-wise unit-norm decoder (dictionary)."""
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def _apply(self, fn):
        """Override _apply to keep control buffers (running_l0, alpha) in float32.
        
        When .to(dtype=bfloat16) is called, all buffers get converted to bfloat16.
        However, running_l0 and alpha need float32 precision for EMA updates to work
        correctly. In bfloat16, certain values like 74.5 are "sticky" where small
        updates round back to the same value, causing the control system to fail.
        """
        # Apply the function to all parameters and buffers normally
        super()._apply(fn)
        
        # Restore float32 precision for control buffers
        # These buffers need full precision for accurate EMA and dual ascent updates
        if hasattr(self, 'running_l0') and self.running_l0.dtype != torch.float32:
            self.running_l0 = self.running_l0.float()
        if hasattr(self, 'alpha') and self.alpha.dtype != torch.float32:
            self.alpha = self.alpha.float()
        
        return self
