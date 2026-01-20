"""
SAEBench Wrapper - Adapts our SAE implementations to SAEBench interface.

This module provides wrapper classes that convert SAEs trained with this codebase
to be compatible with SAEBench evaluation framework.

SAEBench expects:
- W_enc: shape (d_in, d_sae) - encoder weights
- W_dec: shape (d_sae, d_in) - decoder weights with unit-norm rows
- b_enc: shape (d_sae,) - encoder bias
- b_dec: shape (d_in,) - decoder bias
- encode(x), decode(feature_acts), forward(x) methods
- cfg: CustomSAEConfig object
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig

from utils.enums import SAEType
from models.saes import (
    ReluSAE,
    TopKSAE,
    LagrangianSAE,
    JumpReLUSAE,
    GatedSAE,
    BatchTopKSAE,
    MatryoshkaSAE,
    MatryoshkaLagrangianSAE,
)


class SAEBenchWrapper(nn.Module):
    """
    Wrapper class that adapts our SAE implementations to SAEBench interface.
    
    This class extracts and remaps parameters from our SAE implementations to match
    the expected SAEBench interface (W_enc, W_dec, b_enc, b_dec).
    """
    
    def __init__(
        self,
        sae: nn.Module,
        sae_type: SAEType,
        model_name: str,
        hook_layer: int,
        hook_name: str,
        device: torch.device,
        dtype: torch.dtype,
        training_tokens: int = -1,
    ):
        """
        Initialize the SAEBench wrapper.
        
        Args:
            sae: The original SAE module from our implementation.
            sae_type: The type of SAE (from SAEType enum).
            model_name: The TransformerLens model name (e.g., "gpt2", "pythia-70m-deduped").
            hook_layer: The layer number where the SAE is applied.
            hook_name: The full hook name (e.g., "blocks.8.hook_resid_post").
            device: The device to place the wrapper on.
            dtype: The dtype for the wrapper parameters.
            training_tokens: Number of training tokens (for plotting metadata).
        """
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self._sae = sae
        self._sae_type = sae_type
        
        # Extract dimensions from the original SAE
        d_in = sae.input_size
        d_sae = sae.n_dict_components
        
        # Extract and map parameters based on SAE type
        W_enc, W_dec, b_enc, b_dec = self._extract_parameters(sae, sae_type, d_in, d_sae)
        
        # Create SAEBench-compatible parameters
        self.W_enc = nn.Parameter(W_enc.clone().to(dtype=dtype, device=device))
        self.W_dec = nn.Parameter(W_dec.clone().to(dtype=dtype, device=device))
        self.b_enc = nn.Parameter(b_enc.clone().to(dtype=dtype, device=device))
        self.b_dec = nn.Parameter(b_dec.clone().to(dtype=dtype, device=device))
        
        # Ensure W_dec has unit-norm rows
        self._normalize_decoder()
        
        # Create CustomSAEConfig for SAEBench
        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_layer=hook_layer,
            hook_name=hook_name,
        )
        self.cfg.dtype = str(dtype).split(".")[-1]  # e.g., "float32"
        self.cfg.architecture = f"lagrangian_sae_{sae_type.value}"
        self.cfg.training_tokens = training_tokens
        
        self.to(device=device, dtype=dtype)
    
    def _extract_parameters(
        self, 
        sae: nn.Module, 
        sae_type: SAEType,
        d_in: int,
        d_sae: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract and map parameters from different SAE types to SAEBench format.
        
        Returns:
            W_enc: (d_in, d_sae) encoder weights
            W_dec: (d_sae, d_in) decoder weights (will be normalized later)
            b_enc: (d_sae,) encoder bias
            b_dec: (d_in,) decoder bias
        """
        with torch.no_grad():
            if sae_type == SAEType.RELU:
                # ReluSAE: encoder is nn.Sequential with Linear + ReLU
                # encoder[0].weight: (n_dict_components, input_size)
                # decoder.weight: (input_size, n_dict_components)
                W_enc = sae.encoder[0].weight.data.T.clone()  # (d_in, d_sae)
                W_dec = sae.dict_elements.T.clone()  # (d_sae, d_in)
                b_enc = sae.encoder[0].bias.data.clone()  # (d_sae,)
                b_dec = sae.decoder.bias.data.clone()  # (d_in,)
                
            elif sae_type in [SAEType.TOPK, SAEType.BATCH_TOPK, SAEType.MATRYOSHKA]:
                # TopK/BatchTopK/Matryoshka: encoder and decoder are nn.Linear without bias
                # encoder.weight: (n_dict_components, input_size)
                # decoder.weight: (input_size, n_dict_components)
                W_enc = sae.encoder.weight.data.T.clone()  # (d_in, d_sae)
                W_dec = sae.dict_elements.T.clone()  # (d_sae, d_in)
                b_enc = torch.zeros(d_sae, device=sae.device)  # No encoder bias
                b_dec = sae.decoder_bias.data.clone()  # (d_in,)
                
            elif sae_type in [SAEType.LAGRANGIAN, SAEType.JUMP_RELU, SAEType.MATRYOSHKA_LAGRANGIAN]:
                # Lagrangian/JumpReLU/MatryoshkaLagrangian: separate encoder_bias and decoder_bias
                # encoder.weight: (n_dict_components, input_size)
                # decoder.weight: (input_size, n_dict_components)
                W_enc = sae.encoder.weight.data.T.clone()  # (d_in, d_sae)
                W_dec = sae.dict_elements.T.clone()  # (d_sae, d_in)
                b_enc = sae.encoder_bias.data.clone()  # (d_sae,)
                b_dec = sae.decoder_bias.data.clone()  # (d_in,)
                
            elif sae_type == SAEType.GATED:
                # GatedSAE: encoder is nn.Linear without bias, has separate gate and magnitude
                # encoder.weight: (n_dict_components, input_size)
                # decoder.weight: (input_size, n_dict_components)
                W_enc = sae.encoder.weight.data.T.clone()  # (d_in, d_sae)
                # GatedSAE doesn't use dict_elements property, use raw decoder weight
                W_dec = sae.decoder.weight.data.T.clone()  # (d_sae, d_in)
                b_enc = torch.zeros(d_sae, device=sae.device)  # No encoder bias
                b_dec = sae.decoder_bias.data.clone()  # (d_in,)
                
            else:
                raise ValueError(f"Unsupported SAE type: {sae_type}")
        
        return W_enc, W_dec, b_enc, b_dec
    
    def _normalize_decoder(self) -> None:
        """Normalize decoder rows to unit norm (SAEBench requirement)."""
        with torch.no_grad():
            norms = torch.norm(self.W_dec.data, dim=1, keepdim=True)
            norms = norms.clamp(min=1e-8)  # Avoid division by zero
            self.W_dec.data /= norms
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to feature space.
        
        This method delegates to the original SAE to ensure identical encoding behavior,
        including sparsity mechanisms (TopK, JumpReLU thresholding, gating, etc.).
        
        Args:
            x: Input tensor of shape (batch, d_in) or (batch, seq, d_in)
            
        Returns:
            Feature activations of shape (batch, d_sae) or (batch, seq, d_sae)
        """
        # Move input to the same device as the original SAE
        orig_device = next(self._sae.parameters()).device
        x_orig_device = x.to(orig_device)
        
        # Use the original SAE's forward pass to get the feature activations
        # This ensures we get the exact same encoding including sparsity mechanisms
        with torch.no_grad():
            sae_output = self._sae(x_orig_device)
            # The feature activations are in the 'c' field of SAEOutput
            encoded = sae_output.c
        
        # Move back to wrapper's device if needed
        return encoded.to(self.device)
    
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to activation space.
        
        Args:
            feature_acts: Feature tensor of shape (batch, d_sae) or (batch, seq, d_sae)
            
        Returns:
            Reconstructed activations of shape (batch, d_in) or (batch, seq, d_in)
        """
        return feature_acts @ self.W_dec + self.b_dec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode then decode.
        
        This method delegates to the original SAE to ensure identical behavior,
        including all sparsity mechanisms and reconstruction.
        
        Args:
            x: Input tensor of shape (batch, d_in) or (batch, seq, d_in)
            
        Returns:
            Reconstructed activations of same shape as input
        """
        # Move input to the same device as the original SAE
        orig_device = next(self._sae.parameters()).device
        x_orig_device = x.to(orig_device)
        
        # Use the original SAE's forward pass for faithful reconstruction
        with torch.no_grad():
            sae_output = self._sae(x_orig_device)
            reconstructed = sae_output.output
        
        # Move back to wrapper's device if needed
        return reconstructed.to(self.device)
    
    def to(self, *args, **kwargs) -> "SAEBenchWrapper":
        """Handle device and dtype updates for both wrapper and original SAE."""
        super().to(*args, **kwargs)
        
        # Also move the original SAE
        self._sae.to(*args, **kwargs)
        
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        
        # Handle positional args
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, str):
                device = torch.device(arg)
        
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self


def wrap_sae_for_saebench(
    sae: nn.Module,
    sae_type: SAEType,
    model_name: str,
    hook_name: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    training_tokens: int = -1,
) -> SAEBenchWrapper:
    """
    Wrap an SAE for SAEBench evaluation.
    
    Args:
        sae: The original SAE module from our implementation.
        sae_type: The type of SAE (from SAEType enum).
        model_name: The TransformerLens model name (e.g., "gpt2", "pythia-70m-deduped").
        hook_name: The full hook name (e.g., "blocks.8.hook_resid_post").
        device: The device to place the wrapper on.
        dtype: The dtype for the wrapper parameters.
        training_tokens: Number of training tokens (for plotting metadata).
        
    Returns:
        SAEBenchWrapper instance ready for SAEBench evaluation.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # Extract hook layer from hook_name
    # e.g., "blocks.8.hook_resid_post" -> 8
    hook_layer = int(hook_name.split(".")[1])
    
    return SAEBenchWrapper(
        sae=sae,
        sae_type=sae_type,
        model_name=model_name,
        hook_layer=hook_layer,
        hook_name=hook_name,
        device=device,
        dtype=dtype,
        training_tokens=training_tokens,
    )


def create_saebench_saes_from_transformer(
    sae_transformer,
    config,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    training_tokens: int = -1,
) -> list[tuple[str, SAEBenchWrapper]]:
    """
    Create SAEBench-compatible SAE wrappers from an SAETransformer.
    
    Args:
        sae_transformer: The SAETransformer model containing trained SAEs.
        config: The training config with model info.
        device: The device to place the wrappers on.
        dtype: The dtype for the wrapper parameters.
        training_tokens: Number of training tokens (for plotting metadata).
        
    Returns:
        List of (sae_id, wrapped_sae) tuples for SAEBench evaluation.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # Get model name from config
    model_name = config.tlens_model_name
    sae_type = config.saes.sae_type
    
    wrapped_saes = []
    
    # Iterate over all SAE positions
    for raw_pos in sae_transformer.raw_sae_positions:
        # Get the SAE module (positions have dots replaced with hyphens in ModuleDict)
        pos_key = raw_pos.replace(".", "-")
        sae = sae_transformer.saes[pos_key]
        
        # Create wrapper
        wrapped_sae = wrap_sae_for_saebench(
            sae=sae,
            sae_type=sae_type,
            model_name=model_name,
            hook_name=raw_pos,
            device=device,
            dtype=dtype,
            training_tokens=training_tokens,
        )
        
        # Create unique ID for this SAE
        sae_id = f"{model_name}_{raw_pos}_{sae_type.value}"
        
        wrapped_saes.append((sae_id, wrapped_sae))
    
    return wrapped_saes
