# topk_sae.py
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


class TopKSAEConfig(SAEConfig):
    """
    Config for Top-K SAE.

    Notes (faithful to Gao et al., "Scaling and Evaluating Sparse Autoencoders"):
    - Enforce exact sparsity via a Top-K activation (no explicit L1 penalty).
    - Bias is a single learned vector used to center inputs before encoding and
      added back after decoding.
    - We support tied encoder initialization (encoder.weight = decoder.weight.T).
    - (Optional) Auxiliary loss to mitigate dead features by giving gradient signal
      to non-selected latents (implements a simple Aux-K).
    """
    sae_type: SAEType = Field(default=SAEType.TOPK, description="Type of SAE (automatically set to topk)")
    k: int = Field(..., description="Number of active features to keep per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")

    # Optional: dead-feature mitigation via auxiliary Top-K on the *inactive* set
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss (select top aux_k from the inactive set)")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead (number of tokens)")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.TOPK
        return values


class TopKSAEOutput(SAEOutput):
    """
    TopK SAE output extending SAEOutput with useful intermediates for loss/analysis.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (after centering)
    mask: Float[torch.Tensor, "... c"]     # binary mask of selected Top-K indices
    auxk_indices: torch.Tensor | None = None  # auxiliary top-k indices for dead latents (shape: ... x aux_k)
    auxk_values: torch.Tensor | None = None   # auxiliary top-k values for dead latents (shape: ... x aux_k)


class TopKSAE(BaseSAE):
    """
    Top-K Sparse Autoencoder (PyTorch) faithful to Gao et al.:
      - Linear encoder/decoder (no bias on the linear layers)
      - Single learned decoder_bias used to center input and add back after decode
      - ReLU (optional) + Top-K selection per sample in the latent space
      - MSE reconstruction loss only (sparsity enforced by Top-K), with an optional Aux-K loss
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
    ):
        """
        Args:
            input_size: Dimensionality of inputs (e.g., residual stream width).
            n_dict_components: Number of dictionary features (latent size).
            k: Number of active features to keep per sample (Top-K).
            sparsity_coeff: Unused for Top-K (present for interface compatibility).
            mse_coeff: Coefficient on MSE reconstruction loss (default 1.0).
            aux_k: If provided (>0), number of auxiliary features from the inactive set.
            aux_coeff: Coefficient on the auxiliary reconstruction loss (default 0.0 if aux_k is None).
            dead_toks_threshold: Threshold for considering a feature as dead (number of tokens).
            init_decoder_orthogonal: Initialize decoder weight columns to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
        """
        super().__init__()
        assert k >= 0, "k must be non-negative"
        assert n_dict_components > 0 and input_size > 0

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.k = int(k)
        assert self.k <= n_dict_components, "k must be less than or equal to n_dict_components"

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0  # not used, but kept for logs
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0
        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None

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

    def _apply_topk(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply (optional) ReLU then Top-K selection along the last dimension.
        Returns:
            code: sparse activations after masking to Top-K
            mask: binary mask (same shape as z) with ones at Top-K indices
        """
        # Compute Top-K per sample along last dim
        topk_idx = torch.topk(z, k=self.k, dim=-1)[1]
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_idx, 1.0)
        code = z * mask
        return code, mask

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> TopKSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        """
        # Center input
        x_centered = x - self.decoder_bias
        # Encoder preactivations
        preacts = self.encoder(x_centered)  # (..., n_dict_components)
        # Top-K sparsification
        c, mask = self._apply_topk(preacts)
        
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
        
        # Decode using normalized dictionary elements + add bias back
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)
        
        return TopKSAEOutput(
            input=x, 
            c=c, 
            output=x_hat, 
            logits=None, 
            preacts=preacts, 
            mask=mask,
            auxk_indices=auxk_indices,
            auxk_values=auxk_values
        )
    
    def sample_hard_concrete(self, log_alpha: torch.Tensor, tau: float = 0.5,
                             limit_a: float = -0.1, limit_b: float = 1.1):
        # Maddison/Jang (Concrete) + Louizos et al. (Hard-Concrete)
        u = torch.rand_like(log_alpha).clamp_(1e-6, 1-1e-6)
        s = torch.sigmoid((log_alpha + torch.log(u) - torch.log(1 - u)) / tau)
        s_bar = s * (limit_b - limit_a) + limit_a
        z = s_bar.clamp(0.0, 1.0)  # gate in [0,1]
        return z

    def compute_loss(self, output: TopKSAEOutput) -> SAELoss:
        """
        Loss = mse_coeff * MSE + aux_coeff * AuxK (optional)

        - No explicit L1 sparsity term (sparsity enforced by Top-K).
        - AuxK: Reconstruct the residual error (input - main_reconstruction) using dead latents
          to provide gradient signal to features that haven't been active recently.
        """
        # Reconstruction loss
        mse_loss = F.mse_loss(output.output, output.input)
        total_loss = self.mse_coeff * mse_loss
        loss_dict: dict[str, torch.Tensor] = {"mse_loss": mse_loss.detach().clone()}

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
        This mirrors common SAE practice and avoids degenerate scaling solutions.
        """
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
