import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss
from models import SAETransformerOutput
from utils.enums import SAEType


def layer_norm(x: Float[Tensor, "... dim"], eps: float = 1e-5) -> Float[Tensor, "... dim"]:
    """Layernorm without the affine transformation."""
    x = x - x.mean(dim=-1, keepdim=True)
    scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return x / scale


def explained_variance(
    pred: Float[Tensor, "... dim"], target: Float[Tensor, "... dim"], layer_norm_flag: bool = False
) -> Float[Tensor, "..."]:
    """Calculate the explained variance of the pred and target.

    Args:
        pred: The prediction to compare to the target.
        target: The target to compare the prediction to.
        layer_norm_flag: Whether to apply layer norm to the pred and target before calculating the loss.

    Returns:
        The explained variance between the prediction and target for each batch and sequence pos.
    """
    if layer_norm_flag:
        pred = layer_norm(pred)
        target = layer_norm(target)
    
    # Calculate residual sum of squares (squared errors)
    residual_sum_of_squares = (pred - target).pow(2).sum(dim=-1)
    
    # Calculate total sum of squares (variance in target)
    # We want to compute variance across the feature dimension only
    target_mean = target.mean(dim=-1, keepdim=True)
    total_sum_of_squares = (target - target_mean).pow(2).sum(dim=-1)
    
    # Explained variance: 1 - (RSS / TSS)
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    explained_var = 1 - residual_sum_of_squares / (total_sum_of_squares + eps)
    
    return explained_var


def get_activations_for_sae_type(sae_output, sae_type: SAEType) -> torch.Tensor:
    """Get the appropriate activations tensor based on SAE type.
    
    All SAE types use sae_output.c as the main activations tensor.
    """
    return sae_output.c


def compute_alive_dictionary_indices(activations: torch.Tensor) -> list[int]:
    """Compute indices of alive (non-zero) dictionary components from activations.
    
    Args:
        activations: Tensor of shape (batch, seq_len, dict_size)
        
    Returns:
        List of indices of dictionary components that have non-zero activations
    """
    assert activations.ndim == 3, "Activations must be a 3D tensor of shape (batch, seq_len, dict_size)"
    nonzero_indices = activations.sum(0).sum(0).nonzero().squeeze().cpu()
    if nonzero_indices.numel() == 0:
        return []
    elif nonzero_indices.numel() == 1:
        return [nonzero_indices.item()]
    else:
        return nonzero_indices.tolist()


@torch.inference_mode()
def reconstruction_metrics(output: SAETransformerOutput, sae_type: SAEType) -> dict[str, float]:
    """Get metrics on the outputs of the SAE-augmented model and the original model.

    Args:
        output: The output of the SAE-augmented Transformer.
        sae_type: The type of SAE being used.

    Returns:
        Dictionary of output metrics
    """
    reconstruction_metrics = {}
    for name, sae_output in output.sae_outputs.items():
        # Use raw output for metrics (for JumpReLU with normalization, output_raw is denormalized)
        raw_output = getattr(sae_output, 'output_raw', sae_output.output)
        # For normalized SAEs, we need the original input too
        # Note: For JumpReLU, sae_output.input is normalized, but we want metrics in original space
        # The activations dict has the original activations
        original_input = output.activations.get(name, sae_output.input)
        
        # Compute MSE loss (same as evaluation.py)
        mse = mse_loss(
            raw_output,
            original_input,
            reduction='mean'
        )
        
        # Fix parameter order: pred first, target second
        var = explained_variance(
            raw_output, original_input.detach().clone(), layer_norm_flag=False
        )
        var_ln = explained_variance(
            raw_output, original_input.detach().clone(), layer_norm_flag=True
        )
        
        # Add MSE to the metrics
        reconstruction_metrics[f"mse/{name}"] = mse.item()
        reconstruction_metrics[f"explained_variance/{name}"] = var.mean().item()
        reconstruction_metrics[f"explained_variance_ln/{name}"] = var_ln.mean().item()
    return reconstruction_metrics


@torch.inference_mode()
def sparsity_metrics(output: SAETransformerOutput, sae_type: SAEType) -> dict[str, float]:
    """Collect sparsity metrics for logging.

    Args:
        output: The output of the SAE-augmented Transformer.
        sae_type: The type of SAE being used.

    Returns:
        Dictionary of sparsity metrics.
    """
    sparsity_metrics = {}
    for name, sae_output in output.sae_outputs.items():
        # Get activations based on SAE type (consistent with evaluation.py)
        acts = get_activations_for_sae_type(sae_output, sae_type)
        
        # Use consistent L0 norm calculation for all SAE types
        l_0_norm = torch.norm(acts, p=0, dim=-1).mean().item()
        
        # Compute alive dictionary components (always computed now)
        alive_indices = compute_alive_dictionary_indices(acts)
        num_alive = len(alive_indices)
        sparsity_metrics[f"alive_dict_components/{name}"] = num_alive

        # Store with consistent naming for cross-SAE comparison
        sparsity_metrics[f"L_0/{name}"] = l_0_norm

    return sparsity_metrics


def loss_metrics(output: SAETransformerOutput) -> dict[str, float]:
    """Calculate loss metrics for the output of the SAE-augmented Transformer.
    """
    loss_metrics = {}
    for name, loss_output in output.loss_outputs.items():
        for loss_name, loss_value in loss_output.loss_dict.items():
            loss_metrics[f"loss/{loss_name}/{name}"] = loss_value if isinstance(loss_value, float) else loss_value.item()
    return loss_metrics


def all_metrics(output: SAETransformerOutput, train: bool = True, sae_type: SAEType | None = None) -> dict[str, float]:
    """Calculate all metrics for the output of the SAE-augmented Transformer.

    Args:
        output: The output of the SAE-augmented Transformer.
        train: Whether this is for training (True) or evaluation (False).
        sae_type: The type of SAE being used (needed for correct activation selection).

    Returns:
        Dictionary of all metrics.
    """
    prefix = "train" if train else "eval"
    
    # Use SAEType.RELU as default if not provided (for backwards compatibility)
    if sae_type is None:
        sae_type = SAEType.RELU
    
    return {
        **{f"{prefix}/{k}": v for k, v in reconstruction_metrics(output, sae_type).items()},
        **{f"{prefix}/{k}": v for k, v in sparsity_metrics(output, sae_type).items()},
        **{f"{prefix}/{k}": v for k, v in loss_metrics(output).items()},
    }
