import torch
from pathlib import Path
from typing import Annotated
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, BeforeValidator
from utils.enums import SAEType


class SAEConfig(BaseModel):
    name: str
    model_config = ConfigDict(extra="ignore", frozen=False)
    sae_type: SAEType = Field(..., description="Type of SAE to use")
    dict_size_to_input_ratio: PositiveFloat = 1.0
    pretrained_sae_paths: Annotated[
        list[Path] | None, BeforeValidator(lambda x: [x] if isinstance(x, (str, Path)) else x)
    ] = Field(None, description="Path to a pretrained SAEs to load. If None, don't load any.")
    retrain_saes: bool = Field(False, description="Whether to retrain the pretrained SAEs.")
    sae_positions: Annotated[
        list[str], BeforeValidator(lambda x: [x] if isinstance(x, str) else x)
    ] = Field(
        ...,
        description="The names of the hook positions to train SAEs on. E.g. 'hook_resid_post' or "
        "['hook_resid_post', 'hook_mlp_out']. Each entry gets matched to all hook positions that "
        "contain the given string.",
    )
    init_decoder_orthogonal: bool = Field(True, description="Whether to initialize the decoder weights to be orthonormal")
    sparsity_coeff: float | None = Field(None, description="Coefficient for the sparsity loss")
    mse_coeff: float | None = Field(None, description="Coefficient for the MSE loss")


class SAEOutput(BaseModel):
    """Base class for SAE outputs using Pydantic BaseModel for clean inheritance."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    input: Float[torch.Tensor, "... dim"]
    c: Float[torch.Tensor, "... c"]
    output: Float[torch.Tensor, "... dim"]
    logits: Float[torch.Tensor, "... c"] | None = None


class SAELoss(BaseModel):
    """Base class for SAE loss outputs using Pydantic BaseModel for clean inheritance."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    loss: Float[torch.Tensor, ""]
    loss_dict: dict[str, Float[torch.Tensor, ""] | float]


class BaseSAE(torch.nn.Module):
    """Base class for SAEs"""
    n_dict_components: int
    input_size: int

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> SAEOutput:
        raise NotImplementedError("Subclasses must implement forward pass")

    def compute_loss(self, output: SAEOutput, *args, **kwargs) -> SAELoss:
        raise NotImplementedError("Subclasses must implement compute_loss")
