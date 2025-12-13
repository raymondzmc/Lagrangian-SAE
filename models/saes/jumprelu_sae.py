# jumprelu_sae.py
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, model_validator
from jaxtyping import Float

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.utils import init_decoder_orthogonal_cuda
from models.saes.activations import StepFunction, JumpReLU
from utils.enums import SAEType


class JumpReLUSAEConfig(SAEConfig):
    """
    Config for JumpReLU SAE.
    
    Notes:
    - Uses JumpReLU activation with learned thresholds
    - L0 regularization via differentiable step function
    - Bandwidth parameter controls gradient smoothness
    """
    sae_type: SAEType = Field(default=SAEType.JUMP_RELU, description="Type of SAE (automatically set to jump_relu)")
    bandwidth: float = Field(0.01, description="Bandwidth for JumpReLU gradient approximation")
    use_pre_enc_bias: bool = Field(False, description="Whether to subtract decoder bias before encoding")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")
    
    # Dead feature tracking
    dead_toks_threshold: int | None = Field(None, description="Threshold for considering a feature as dead (number of tokens)")
    
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
        bandwidth: float = 0.01,
        use_pre_enc_bias: bool = False,
        sparsity_coeff: float | None = None,  # Used for L0 coefficient
        mse_coeff: float | None = None,
        dead_toks_threshold: int | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
    ):
        """
        Args:
            input_size: Dimensionality of inputs.
            n_dict_components: Number of dictionary features.
            bandwidth: Bandwidth for JumpReLU gradient approximation.
            use_pre_enc_bias: Whether to subtract decoder bias before encoding.
            sparsity_coeff: Coefficient for L0 regularization.
            mse_coeff: Coefficient on MSE reconstruction loss.
            dead_toks_threshold: Threshold for dead feature tracking.
            init_decoder_orthogonal: Initialize decoder weights to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
        """
        super().__init__()
        assert n_dict_components > 0 and input_size > 0
        
        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.bandwidth = bandwidth
        self.use_pre_enc_bias = use_pre_enc_bias
        
        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0  # L0 coefficient
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        
        self.dead_toks_threshold = int(dead_toks_threshold) if dead_toks_threshold is not None else None
        
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
        
        # Dead latent tracking
        self.register_buffer("num_batches_not_active", torch.zeros(n_dict_components, dtype=torch.float32))
    
    def _update_dead_features(self, acts: torch.Tensor) -> None:
        """Update dead feature tracking statistics."""
        # Increment counter for all features
        self.num_batches_not_active += 1.0
        
        # Reset counter for features that are active
        active_mask = (acts.sum(dim=tuple(range(acts.ndim - 1))) > 0)
        self.num_batches_not_active[active_mask] = 0.0
    
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
        
        # Decode
        x_reconstruct = F.linear(feature_activations, self.dict_elements) + self.decoder_bias
        
        # Compute L0 for loss
        l0 = StepFunction.apply(
            pre_activations, 
            self.jumprelu.log_threshold, 
            self.bandwidth
        ).sum(dim=-1)
        
        # Update dead feature statistics if training
        if self.training and self.dead_toks_threshold is not None:
            self._update_dead_features(feature_activations)
        
        return JumpReLUSAEOutput(
            input=x,
            c=feature_activations,
            output=x_reconstruct,
            logits=None,
            pre_activations=pre_activations,
            l0=l0
        )
    
    def compute_loss(self, output: JumpReLUSAEOutput) -> SAELoss:
        """
        Loss = mse_coeff * MSE + sparsity_coeff * L0
        
        L0 is computed using a differentiable step function approximation.
        """
        # MSE reconstruction loss
        mse_loss = F.mse_loss(output.output, output.input)
        
        # L0 regularization
        l0_loss = output.l0.mean()
        
        # Total loss
        total_loss = self.mse_coeff * mse_loss + self.sparsity_coeff * l0_loss
        
        # Compute number of dead features
        num_dead_features = (
            self.num_batches_not_active > self.dead_toks_threshold
        ).sum() if self.dead_toks_threshold is not None else torch.tensor(0)
        
        loss_dict: dict[str, torch.Tensor] = {
            "mse_loss": mse_loss.detach().clone(),
            "l0_loss": l0_loss.detach().clone(),
            "l0_norm": output.l0.mean().detach().clone(),
            "num_dead_features": num_dead_features.float(),
        }
        
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
