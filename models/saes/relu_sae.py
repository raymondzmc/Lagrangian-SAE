import torch
import torch.nn.functional as F
from torch import nn
from pydantic import Field, model_validator
from typing import Any
from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from models.saes.utils import init_decoder_orthogonal_cuda
from utils.enums import SAEType


class ReLUSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.RELU, description="Type of SAE (automatically set to relu)")
    
    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure sae_type is set to relu."""
        if isinstance(values, dict):
            values["sae_type"] = SAEType.RELU
        return values


class ReluSAE(BaseSAE):
    """
    Sparse AutoEncoder with ReLU activation.
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        sparsity_coeff: float | None = None,
        mse_coeff: float | None = None,
        init_decoder_orthogonal: bool = True,
    ):
        """Initialize the ReluSAE.

        Args:
            input_size: Dimensionality of input data
            n_dict_components: Number of dictionary components
            init_decoder_orthogonal: Initialize the decoder weights to be orthonormal
        """

        super().__init__()
        # self.encoder[0].weight has shape: (n_dict_components, input_size)
        # self.decoder.weight has shape:    (input_size, n_dict_components)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, n_dict_components, bias=True),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(n_dict_components, input_size, bias=True)
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        if init_decoder_orthogonal:
            # Initialize so that there are n_dict_components orthonormal vectors
            self.decoder.weight.data = init_decoder_orthogonal_cuda(self.decoder.weight)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        """Pass input through the encoder and normalized decoder."""
        c = self.encoder(x)
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder.bias)
        return SAEOutput(input=x, c=c, output=x_hat, logits=None)

    def compute_loss(self, output: SAEOutput) -> SAELoss:
        """Compute the loss for the ReluSAE.

        Args:
            output: The output of the ReluSAE.
        """
        mse_loss = F.mse_loss(output.output, output.input)
        sparsity_loss = torch.norm(output.c, p=1.0, dim=-1).mean() / self.input_size
        loss = self.sparsity_coeff * sparsity_loss + self.mse_coeff * mse_loss
        return SAELoss(
            loss=loss,
            loss_dict={
                "mse_loss": mse_loss.detach().clone(),
                "sparsity_loss": sparsity_loss.detach().clone(),
            },
        )

    @property
    def dict_elements(self):
        """Dictionary elements are simply the normalized decoder weights."""
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
