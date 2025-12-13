import torch
import torch.nn.functional as F
from torch import nn
from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from pydantic import ConfigDict, Field
from utils.enums import EncoderType


class GatedSAEConfig(SAEConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary loss")
    magnitude_encoder: EncoderType = Field(EncoderType.SCALE, description="Type of magnitude encoder")
    magnitude_activation: str | None = Field("relu", description="Activation function for magnitude ('relu', 'softplus', etc.) or None")


class GatedSAEOutput(SAEOutput):
    gates: torch.Tensor
    magnitudes: torch.Tensor
    mask: torch.Tensor


class GatedSAE(BaseSAE):
    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        magnitude_encoder: str = "scale",
        magnitude_activation: str | None = "relu",
        sparsity_coeff: float | None = None,
        mse_coeff: float | None = None,
        aux_coeff: float | None = None,
    ):
        """
        Gated Sparse Autoencoder with tied encoder weights.
        input_dim: dimensionality of input x.
        hidden_dim: number of sparse features (dictionary size).
        """
        super().__init__()
        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.magnitude_encoder_type = magnitude_encoder
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        self.aux_coeff = aux_coeff if aux_coeff is not None else 1.0

        # Decoder bias for input centering
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        
        # Encoder (no bias, tied to decoder transpose)
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        
        # Magnitude network parameters
        self.r_mag = nn.Parameter(torch.zeros(n_dict_components))
        self.mag_bias = nn.Parameter(torch.zeros(n_dict_components))

        # Gating network parameters
        self.gate_bias = nn.Parameter(torch.zeros(n_dict_components))

        # Decoder (no bias, bias handled separately)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        
        # Initialize weights properly
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are already initialized to zero in __init__, no need to re-zero them
        
        # decoder weights are initialized to random unit vectors
        dec_weight = torch.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        # tie encoder weights to decoder transpose
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)

    def forward(self, x: torch.Tensor):
        """
        Forward pass returns reconstruction and intermediate codes.
        """
        # Center input by subtracting decoder bias (treated as input mean)
        x_enc = self.encoder(x - self.decoder_bias)  # (batch, hidden_dim)

        # Gating network: simple bias addition (no scaling)
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).float()  # Heaviside step -> {0,1}

        # Magnitude network: exponential scaling + bias + ReLU
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias

        f_mag = F.relu(pi_mag)

        # Combine gating and magnitude
        code = f_gate * f_mag
        
        # Decode
        recon = self.decoder(code) + self.decoder_bias  # (batch, input_dim)
        
        return GatedSAEOutput(
            input=x, 
            c=code, 
            output=recon, 
            logits=None, 
            gates=pi_gate,  # gating pre-activations
            magnitudes=f_mag,  # magnitude activations
            mask=f_gate  # binary gating mask
        )

    def compute_loss(self, output: GatedSAEOutput) -> SAELoss:
        """
        Compute the Gated SAE loss based on the reference implementation:
        L_recon + L_sparsity + L_aux
        """
        # L_recon: Reconstruction loss (MSE)
        L_recon = F.mse_loss(output.output, output.input)
        
        # L_sparsity: Sparsity loss using L1 norm on gate activations (ReLU of gate pre-activations)
        # In reference: f_gate = ReLU(pi_gate), then lp_norm(f_gate, p=1)
        f_gate = F.relu(output.gates)
        L_sparsity = torch.norm(f_gate, p=1.0, dim=-1).mean()
        
        # L_aux: Auxiliary reconstruction loss using gate activations with detached decoder
        with torch.no_grad():
            # Detach decoder weights and bias to stop gradients
            dec_weight_detached = self.decoder.weight.detach()
            dec_bias_detached = self.decoder_bias.detach()
        
        # Reconstruct using gate activations: x_hat_gate = f_gate @ W_dec^T + b_dec
        x_hat_gate = f_gate @ dec_weight_detached.T + dec_bias_detached
        L_aux = F.mse_loss(x_hat_gate, output.input)
        
        # Total loss: L_recon + alpha * L_sparsity + L_aux
        # Note: Reference implementation doesn't use separate mse_coeff, typically mse_coeff=1.0
        total_loss = self.mse_coeff * L_recon + self.sparsity_coeff * L_sparsity + self.aux_coeff * L_aux
        
        loss_dict = {
            "mse_loss": L_recon.detach().clone(),
            "sparsity_loss": L_sparsity.detach().clone(), 
            "aux_loss": L_aux.detach().clone(),
        }
        
        return SAELoss(loss=total_loss, loss_dict=loss_dict)

    @property
    def device(self):
        return next(self.parameters()).device
