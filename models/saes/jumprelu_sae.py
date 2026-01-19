# jumprelu_sae.py
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
    Config for JumpReLU SAE.
    
    Notes:
    - Uses JumpReLU activation with learned thresholds
    - Target L0 regularization via differentiable step function: loss = ((L0 / target_l0) - 1)²
    - Bandwidth parameter controls gradient smoothness
    - Supports auxiliary loss for dead feature mitigation (like TopK/BatchTopK)
    """
    sae_type: SAEType = Field(default=SAEType.JUMP_RELU, description="Type of SAE (automatically set to jump_relu)")
    target_l0: float = Field(..., description="Target L0 sparsity (number of active features per sample)")
    bandwidth: float = Field(0.01, description="Bandwidth for JumpReLU gradient approximation")
    use_pre_enc_bias: bool = Field(False, description="Whether to subtract decoder bias before encoding")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    
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
    JumpReLU Sparse Autoencoder:
      - Linear encoder/decoder with bias
      - JumpReLU activation with learned thresholds
      - L0 regularization via differentiable step function
      - MSE reconstruction loss
    """
    
    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        target_l0: float,
        bandwidth: float = 0.01,
        use_pre_enc_bias: bool = False,
        sparsity_coeff: float | None = None,  # Coefficient for target L0 loss
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
            bandwidth: Bandwidth for JumpReLU gradient approximation.
            use_pre_enc_bias: Whether to subtract decoder bias before encoding.
            sparsity_coeff: Coefficient for target L0 regularization loss.
            mse_coeff: Coefficient on MSE reconstruction loss.
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
        self.use_pre_enc_bias = use_pre_enc_bias
        
        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0  # Target L0 loss coefficient
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
            device='cpu'  # Will be moved with the model
        )
        
        # Initialize decoder, then (optionally) tie encoder init
        if init_decoder_orthogonal:
            self.decoder.weight.data = init_decoder_orthogonal_cuda(self.decoder.weight)
        else:
            # Random unit-norm columns
            dec_w = torch.randn_like(self.decoder.weight)
            dec_w = F.normalize(dec_w, dim=0)
            self.decoder.weight.data.copy_(dec_w)
        
        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)
        
        # Dead latent tracking - counts tokens since last activation (like TopK/BatchTopK)
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dict_components, dtype=torch.long))
    
    def forward(self, x: Float[torch.Tensor, "... dim"]) -> JumpReLUSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        """
        # Optional: subtract decoder bias before encoding
        if self.use_pre_enc_bias:
            x_enc = x - self.decoder_bias
        else:
            x_enc = x
        
        # Encode with ReLU pre-activation
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
        x_reconstruct = F.linear(feature_activations, self.dict_elements) + self.decoder_bias
        
        # Compute L0 for loss
        l0 = StepFunction.apply(
            pre_activations, 
            self.jumprelu.log_threshold, 
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
        Loss = mse_coeff * MSE + sparsity_coeff * ((L0 / target_l0) - 1)² + aux_coeff * AuxK (optional)
        
        L0 is computed using a differentiable step function approximation.
        The sparsity loss penalizes deviation from the target L0 in either direction.
        AuxK: Reconstruct the residual error using dead latents to provide gradient signal.
        """
        # MSE reconstruction loss
        mse_loss = F.mse_loss(output.output, output.input)
        
        # Target L0 regularization: penalize deviation from target
        mean_l0 = output.l0.mean()
        sparsity_loss = self.sparsity_coeff * ((mean_l0 / self.target_l0) - 1.0) ** 2
        
        # Total loss
        total_loss = self.mse_coeff * mse_loss + sparsity_loss
        
        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": mse_loss.detach().clone(),
            "sparsity_loss": sparsity_loss.detach().clone(),
            "l0_norm": mean_l0.detach().clone(),
            "target_l0": torch.tensor(self.target_l0),
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
