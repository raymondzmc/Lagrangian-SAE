# lagrangian_sae.py
"""
Lagrangian Sparse Autoencoder using dual ascent to learn sparsity.

This SAE uses a Lagrangian multiplier (alpha) to adaptively control the sparsity penalty,
targeting an L0 sparsity level. Unlike TopK/BatchTopK which enforce exact
sparsity, this approach learns to achieve a target sparsity through optimization.

Key design choices:
- Supports two constraint types via `equality_constraint` option:
  - Equality (default, equality_constraint=True): penalizes when L0 ≠ target_l0 (both directions)
  - Inequality (equality_constraint=False): penalizes only when L0 > target_l0 (single direction)
- Equality mode: alpha can be negative (incentivizes more features when L0 < target)
- Inequality mode: alpha is non-negative (only penalizes excess features)
- Uses running mean (EMA) of L0 for stable constraint evaluation
- Uses per-feature learned thresholds (like JumpReLU) for flexible sparsity control
- Uses differentiable L0 approximation for gradient computation
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


class LagrangianSAEConfig(SAEConfig):
    """
    Config for Lagrangian SAE.

    Notes:
    - Uses JumpReLU activation with per-feature learned thresholds
    - Differentiable L0 approximation for gradient computation  
    - Dual ascent on alpha to achieve target sparsity
    - Supports two constraint modes via `equality_constraint`:
      - True (default): Equality constraint (L0 ≈ target_l0), alpha in [-alpha_min, alpha_max]
      - False: Inequality constraint (L0 ≤ target_l0), alpha in [0, alpha_max]
    - Supports asymmetric bounds via `alpha_min` for "soft inequality" behavior
    - Uses running mean (EMA) of L0 for stable constraint evaluation
    - Comparable to TopK/BatchTopK for analyzing global expected L0
    """
    sae_type: SAEType = Field(default=SAEType.LAGRANGIAN, description="Type of SAE (automatically set to lagrangian)")
    target_l0: float = Field(..., description="Target L0 sparsity (number of active features per sample)")
    initial_alpha: float = Field(0.0, description="Initial alpha for Lagrangian multiplier (can be positive or negative)")
    alpha_lr: float = Field(1e-2, description="Learning rate for alpha dual ascent")
    alpha_lr_down: float | None = Field(None, description="Unused for equality constraint (kept for backward compatibility)")
    alpha_max: float = Field(100.0, description="Maximum alpha when L0 > target (upper bound)")
    alpha_min: float | None = Field(None, description="Maximum alpha magnitude when L0 < target (lower bound magnitude). If None, uses alpha_max for symmetric bounds. Set smaller than alpha_max for asymmetric 'soft inequality' behavior.")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    use_pre_enc_bias: bool = Field(False, description="Subtract decoder bias before encoding")
    
    # Activation normalization (recommended for consistent behavior across layers)
    # Uses same naming convention as JumpReLUSAE for compatibility with run.py pre-computation
    normalize_activations: bool = Field(False, description="Normalize inputs to unit mean squared norm (recommended for stable training)")
    norm_factor_num_batches: int = Field(100, description="Number of batches to compute norm factor over before training (dictionary_learning uses 100)")
    
    # Running mean (EMA) parameters for stable L0 estimation
    l0_ema_momentum: float = Field(0.99, description="Momentum for running mean of L0 (higher = smoother)")
    
    # Per-feature threshold parameters (like JumpReLU)
    # NOTE: initial_threshold is a fallback; use calibrate_thresholds=True for automatic calibration
    initial_threshold: float = Field(0.5, description="Initial per-feature threshold value (fallback if calibration disabled)")
    bandwidth: float = Field(0.5, description="Bandwidth for step function gradient approximation (larger = more gradient signal)")
    
    # Threshold calibration (RECOMMENDED: ensures initial L0 ≈ target_l0)
    calibrate_thresholds: bool = Field(True, description="Auto-calibrate thresholds on first batch to achieve target L0")
    
    # Dead feature tracking and auxiliary loss (optional)
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead (number of tokens)")
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss (select top aux_k from the inactive set)")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")
    
    # Quadratic penalty coefficient for constraint violation (augmented Lagrangian)
    rho_quadratic: float = Field(0.1, description="Coefficient for quadratic penalty term in augmented Lagrangian")
    
    # Constraint type: equality (L0 ≈ target) vs inequality (L0 ≤ target)
    equality_constraint: bool = Field(True, description="If True, enforce L0 ≈ target (equality); if False, enforce L0 <= target (inequality)")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.LAGRANGIAN
        return values


class LagrangianSAEOutput(SAEOutput):
    """
    Lagrangian SAE output extending SAEOutput with useful intermediates.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (before thresholding)
    l0: Float[torch.Tensor, "..."]  # L0 norm per sample (actual count of non-zero activations)
    l0_differentiable: Float[torch.Tensor, "..."]  # Differentiable L0 approximation for gradient
    alpha: torch.Tensor  # current Lagrangian multiplier value
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents (shape: ... x aux_k)
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents (shape: ... x aux_k)


class LagrangianSAE(BaseSAE):
    """
    Lagrangian Sparse Autoencoder:
      - Linear encoder/decoder with bias
      - JumpReLU activation with per-feature learned thresholds
      - Differentiable L0 approximation for gradient computation
      - Dual ascent on alpha to achieve target sparsity
      - MSE reconstruction loss
      
    Supports two constraint modes via `equality_constraint`:
    
    Equality constraint (equality_constraint=True, default):
        min_θ E[MSE(x, x̂)]  s.t. E[L0(c)] = target_l0
        L = MSE + α · (L0_diff - target) + ρ/2 · (L0 - target)²
        α ← α + α_lr · (L0 - target_l0), α ∈ [-α_min, α_max]
        
    Inequality constraint (equality_constraint=False):
        min_θ E[MSE(x, x̂)]  s.t. E[L0(c)] ≤ target_l0
        L = MSE + α · max(0, L0_diff - target) + ρ/2 · max(0, L0 - target)²
        α ← α + α_lr · (L0 - target_l0), α ∈ [0, α_max]
    
    Asymmetric bounds (alpha_min < alpha_max):
        Creates "soft inequality" behavior where L0 > target is penalized more
        strongly than L0 < target. Default: alpha_min = alpha_max (symmetric).
    
    Unlike fixed-threshold approaches, this SAE learns per-feature thresholds,
    allowing each dictionary element to find its optimal activation threshold.
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        target_l0: float,
        initial_alpha: float = 0.0,
        alpha_lr: float = 1e-2,
        alpha_lr_down: float | None = None,
        alpha_max: float = 100.0,
        alpha_min: float | None = None,
        rho_quadratic: float = 0.1,
        l0_ema_momentum: float = 0.99,
        initial_threshold: float = 0.5,
        bandwidth: float = 0.1,
        sparsity_coeff: float | None = None,  # unused; kept for API parity
        mse_coeff: float | None = None,
        dead_toks_threshold: int | None = None,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        use_pre_enc_bias: bool = False,
        normalize_activations: bool = False,
        norm_factor_num_batches: int = 100,
        calibrate_thresholds: bool = True,
        equality_constraint: bool = True,
    ):
        """
        Args:
            input_size: Dimensionality of inputs (e.g., residual stream width).
            n_dict_components: Number of dictionary features (latent size).
            target_l0: Target number of active features per sample.
            initial_alpha: Initial value for Lagrangian multiplier (can be positive or negative).
            alpha_lr: Learning rate for alpha dual ascent.
            alpha_lr_down: Unused (kept for backward compatibility).
            alpha_max: Maximum alpha when L0 > target (upper bound).
            alpha_min: Maximum alpha magnitude when L0 < target (lower bound magnitude).
                       If None, uses alpha_max for symmetric bounds. Set smaller than
                       alpha_max for asymmetric "soft inequality" behavior.
            rho_quadratic: Coefficient for quadratic penalty in augmented Lagrangian.
            l0_ema_momentum: Momentum for running mean of L0 (0.99 = smooth, 0.0 = no smoothing).
            initial_threshold: Initial per-feature threshold value (fallback if calibration disabled).
            bandwidth: Bandwidth for step function gradient approximation.
            sparsity_coeff: Unused for Lagrangian (present for interface compatibility).
            mse_coeff: Coefficient on MSE reconstruction loss (default 1.0).
            dead_toks_threshold: Threshold for considering a feature as dead.
            aux_k: If provided (>0), number of auxiliary features from the inactive set.
            aux_coeff: Coefficient on the auxiliary reconstruction loss (default 0.0 if aux_k is None).
            init_decoder_orthogonal: Initialize decoder weight columns to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
            use_pre_enc_bias: Whether to subtract decoder bias before encoding.
            normalize_activations: Normalize inputs to unit mean squared norm (recommended for stable training).
            norm_factor_num_batches: Number of batches to compute norm factor over before training.
            calibrate_thresholds: Auto-calibrate thresholds on first batch to achieve target L0.
            equality_constraint: If True, enforce L0 ≈ target (equality); if False, enforce L0 <= target (inequality).
        """
        super().__init__()
        assert target_l0 > 0, "target_l0 must be positive"
        assert n_dict_components > 0 and input_size > 0
        assert bandwidth > 0, "bandwidth must be positive"
        assert 0.0 <= l0_ema_momentum < 1.0, "l0_ema_momentum must be in [0, 1)"
        # initial_alpha can be positive or negative for equality constraint
        assert initial_threshold > 0, "initial_threshold must be positive"

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.target_l0 = target_l0
        self.alpha_lr = alpha_lr
        # alpha_lr_down is unused but kept for backward compatibility
        self.alpha_lr_down = alpha_lr_down if alpha_lr_down is not None else 0.1 * alpha_lr
        self.alpha_max = alpha_max
        # alpha_min defaults to alpha_max for symmetric bounds; set smaller for asymmetric "soft inequality"
        self.alpha_min = alpha_min if alpha_min is not None else alpha_max
        self.rho_quadratic = rho_quadratic
        self.l0_ema_momentum = l0_ema_momentum
        self.bandwidth = bandwidth
        self.use_pre_enc_bias = use_pre_enc_bias
        self.normalize_activations = normalize_activations
        self.norm_factor_num_batches = norm_factor_num_batches
        self.equality_constraint = equality_constraint

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0  # not used directly
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None
        
        # Auxiliary loss for dead features (same as TopK/BatchTopK)
        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0

        # Lagrangian multiplier (alpha) as a buffer (not a parameter - updated via dual ascent)
        # Alpha can be positive or negative for equality constraint
        self.register_buffer("alpha", torch.tensor(initial_alpha, dtype=torch.float32))
        
        # Running mean of L0 for stable constraint evaluation (EMA)
        # Initialize to target_l0 so the constraint starts satisfied
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
            # Random unit-norm columns
            dec_w = torch.randn_like(self.decoder.weight)
            dec_w = F.normalize(dec_w, dim=0)
            self.decoder.weight.data.copy_(dec_w)

        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)

        # Dead latent tracking - counts tokens since last activation
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
        Calibrate thresholds based on accumulated pre-activation statistics.
        
        Sets per-feature thresholds such that the expected L0 ≈ target_l0.
        
        The key insight: if we want L0 = target_l0 features active per sample on average,
        we need to set each feature's threshold at a percentile such that:
            P(preact > threshold) ≈ target_l0 / n_dict_components
        
        This means threshold should be at the (1 - target_l0/n_dict_components) percentile.
        """
        # Flatten to (N, n_dict_components)
        flat_preacts = preacts.reshape(-1, self.n_dict_components)
        n_samples = flat_preacts.shape[0]
        
        # Target: each feature should activate with probability target_l0 / n_dict_components
        target_activation_prob = self.target_l0 / self.n_dict_components
        target_percentile = 1.0 - target_activation_prob  # e.g., 0.998 for L0=32, n=24576
        
        # Compute per-feature threshold at target percentile
        # For each feature, find the value at target_percentile
        sorted_preacts, _ = torch.sort(flat_preacts, dim=0)
        percentile_idx = int(target_percentile * n_samples)
        percentile_idx = min(percentile_idx, n_samples - 1)  # Clamp to valid range
        
        # Get threshold values at target percentile for each feature
        threshold_values = sorted_preacts[percentile_idx, :]  # Shape: (n_dict_components,)
        
        # Clamp thresholds to be positive (JumpReLU requires positive thresholds)
        threshold_values = threshold_values.clamp(min=1e-4)
        
        # Set threshold directly (raw parameterization)
        self.jumprelu.threshold.data.copy_(threshold_values)
        
        # Mark as calibrated
        self.thresholds_calibrated.fill_(True)
        
        # Log calibration info
        mean_threshold = threshold_values.mean().item()
        min_threshold = threshold_values.min().item()
        max_threshold = threshold_values.max().item()
        print(f"[LagrangianSAE] Thresholds calibrated: mean={mean_threshold:.4f}, "
              f"min={min_threshold:.4f}, max={max_threshold:.4f}, "
              f"target_percentile={target_percentile:.6f}")

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

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> LagrangianSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        
        If normalize_activations is enabled:
        - Input is scaled to achieve mean squared L2 norm ≈ 1 (preserves mean/direction)
        - SAE operates in normalized space (encoding, thresholding, decoding)
        - Output is denormalized back to original scale
        - MSE loss is computed in original space for proper comparison
        """
        # Activation normalization (if enabled)
        if self.normalize_activations:
            norm_factor = self._compute_norm_factor(x)
            x_normalized = x / norm_factor
        else:
            norm_factor = None
            x_normalized = x
        
        # Optional: subtract decoder bias before encoding (matches apply_b_dec_to_input in dictionary_learning)
        if self.use_pre_enc_bias:
            x_enc = x_normalized - self.decoder_bias
        else:
            x_enc = x_normalized

        # Encoder pre-activations (no ReLU - matches dictionary_learning's jumprelu.py)
        # JumpReLU will handle the activation directly on these pre-activations
        preacts = self.encoder(x_enc) + self.encoder_bias
        
        # Threshold calibration on first batch (if enabled and not yet calibrated)
        if self.calibrate_thresholds and self.training and not self.thresholds_calibrated:
            with torch.no_grad():
                self._calibrate_thresholds_from_preacts(preacts)
        
        # Apply JumpReLU with per-feature learned thresholds
        c = self.jumprelu(preacts)

        # Update dead latent statistics if training
        update_dead_feature_stats(
            activations=c,
            stats_last_nonzero=self.stats_last_nonzero,
            training=self.training,
            dead_toks_threshold=self.dead_toks_threshold,
        )

        # Compute auxiliary top-k indices and values for dead latents
        auxk_values, auxk_indices = maybe_compute_auxk_features(
            preacts=preacts,
            stats_last_nonzero=self.stats_last_nonzero,
            aux_k=self.aux_k,
            aux_coeff=self.aux_coeff,
            dead_toks_threshold=self.dead_toks_threshold,
        )

        # Decode using normalized dictionary elements + add bias back (in normalized space)
        x_hat_normalized = F.linear(c, self.dict_elements, bias=self.decoder_bias)
        
        # Denormalize output back to original scale (scale-only: just multiply)
        if self.normalize_activations and norm_factor is not None:
            x_hat = x_hat_normalized * norm_factor
        else:
            x_hat = x_hat_normalized

        # Compute true L0 (for constraint evaluation and logging)
        l0 = (c > 0).float().sum(dim=-1)
        
        # Compute differentiable L0 using StepFunction (gradient flows to learned thresholds)
        l0_differentiable = StepFunction.apply(
            preacts, 
            self.jumprelu.threshold, 
            self.bandwidth
        ).sum(dim=-1)

        return LagrangianSAEOutput(
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

    def compute_loss(self, output: LagrangianSAEOutput) -> SAELoss:
        """
        Computes the augmented Lagrangian loss for sparsity control.
        
        Equality constraint (equality_constraint=True, default):
            Loss = MSE + α * (L0_diff - target) + ρ/2 * (L0_diff - target)²
            Penalizes both directions: L0 < target and L0 > target.
            
        Inequality constraint (equality_constraint=False):
            Loss = MSE + α * max(0, L0_diff - target) + ρ/2 * max(0, L0_diff - target)²
            Only penalizes when L0 > target; no penalty when L0 ≤ target.
        
        Uses:
        - L0_diff: Differentiable L0 approximation (provides gradient to learned thresholds)
        - L0: True L0 (used for constraint evaluation and dual ascent)
        - running_l0: Exponential moving average of L0 for stable dual ascent
        
        Note: Dual ascent on alpha is performed via update_alpha() which should be
        called after optimizer.step() to avoid autograd conflicts.
        """
        # Reconstruction loss
        mse_loss = F.mse_loss(output.output, output.input)
        
        # Compute mean L0 across the batch (true L0 for monitoring)
        mean_l0 = output.l0.mean()
        
        # Update running L0 (EMA) for stable constraint evaluation
        if self.training:
            with torch.no_grad():
                # EMA update: running_l0 = momentum * running_l0 + (1 - momentum) * batch_l0
                # NOTE: Compute in float32 to avoid bfloat16 precision issues where EMA
                # updates can get lost due to limited mantissa bits (e.g., 74.5 is a "sticky"
                # value in bfloat16 where small updates round back to the same value)
                running_l0_f32 = self.running_l0.float()
                running_l0_f32 = (
                    self.l0_ema_momentum * running_l0_f32
                    + (1.0 - self.l0_ema_momentum) * mean_l0.detach().float()
                )
                self.running_l0.copy_(running_l0_f32)
        
        # Use RUNNING L0 for dual ascent (smoother updates)
        running_constraint_violation = self.running_l0 - self.target_l0
        
        # Store RAW constraint violation for dual ascent (can be negative)
        self._last_constraint_violation = running_constraint_violation.detach().clone()
        
        # Differentiable L0 for gradient computation
        mean_l0_diff = output.l0_differentiable.mean()
        differentiable_violation = mean_l0_diff - self.target_l0
        
        # Get current alpha value (detached to avoid gradient through alpha)
        alpha_value = self.alpha.detach().clone()
        
        # Compute normalized violation based on constraint type
        if self.equality_constraint:
            # Equality constraint: penalize both directions (no ReLU)
            normalized_violation = differentiable_violation / self.target_l0
        else:
            # Inequality constraint: only penalize when L0 > target (use ReLU)
            normalized_violation = F.relu(differentiable_violation) / self.target_l0
        
        # Augmented Lagrangian formulation:
        # Equality: L = MSE + α * (L0_diff - target) + ρ/2 * (L0_diff - target)²
        # Inequality: L = MSE + α * max(0, L0_diff - target) + ρ/2 * max(0, L0_diff - target)²
        sparsity_loss = alpha_value * normalized_violation
        quadratic_penalty = self.rho_quadratic * (normalized_violation ** 2)
        
        total_loss = self.mse_coeff * mse_loss + sparsity_loss + quadratic_penalty

        # Compute mean threshold for logging
        mean_threshold = self.jumprelu.threshold.mean()

        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": mse_loss.detach().clone(),
            "running_l0": self.running_l0.detach().clone(),  # Running mean L0 (expected L0)
            "alpha": alpha_value,
            "quadratic_penalty": quadratic_penalty.detach().clone(),
            "mean_threshold": mean_threshold.detach().clone(),  # Track learned thresholds
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
    
    @torch.no_grad()
    def update_alpha(self) -> None:
        """
        Perform dual ascent update on alpha.
        
        Should be called after optimizer.step() to avoid autograd conflicts.
        
        Update rule: α ← α + α_lr · (L0 - target_l0)
        
        Clamping depends on constraint type:
        - Equality (equality_constraint=True, default): α ∈ [-α_min, α_max]
          Alpha can be negative to incentivize more features when L0 < target.
          Use alpha_min < alpha_max for asymmetric "soft inequality" behavior.
        - Inequality (equality_constraint=False): α ∈ [0, α_max]
          Alpha is non-negative; only penalizes when L0 > target.
        """
        if hasattr(self, '_last_constraint_violation'):
            # Dual ascent update
            self.alpha.add_(self.alpha_lr * self._last_constraint_violation)
            # Clamp alpha based on constraint type
            if self.equality_constraint:
                # Equality: alpha can be negative (to incentivize more active features)
                # Uses asymmetric bounds [-alpha_min, +alpha_max] for "soft inequality" when alpha_min < alpha_max
                self.alpha.clamp_(min=-self.alpha_min, max=self.alpha_max)
            else:
                # Inequality: alpha must be non-negative (only penalizes, never incentivizes)
                self.alpha.clamp_(min=0, max=self.alpha_max)

    @property
    def dict_elements(self) -> torch.Tensor:
        """
        Column-wise unit-norm decoder (dictionary) – normalized every forward.
        This mirrors common SAE practice and avoids degenerate scaling solutions.
        """
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
