import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def init_decoder_orthogonal_cuda(decoder_weight: torch.Tensor) -> torch.Tensor:
    """
    Initialize decoder weights to be orthogonal using CUDA if available, otherwise CPU.
    
    This is equivalent to: torch.nn.init.orthogonal_(decoder_weight.data.T).T
    but performs the initialization on CUDA to potentially speed up the computation
    for large matrices, then moves the result back to the original device.
    
    Args:
        decoder_weight: Decoder weight tensor to initialize (shape: input_size x n_dict_components)
        
    Returns:
        Orthogonally initialized decoder weight tensor on the original device
    """
    original_device = decoder_weight.device
    
    # Use CUDA if available, otherwise use the original device
    if torch.cuda.is_available():
        W = decoder_weight.data.clone().to("cuda")
        W = torch.nn.init.orthogonal_(W.T).T
        return W.to(original_device).clone()
    else:
        # If CUDA is not available, perform initialization on the original device
        W = decoder_weight.data.clone()
        W = torch.nn.init.orthogonal_(W.T).T
        return W


def get_dead_latent_mask(
    stats_last_nonzero: torch.Tensor,
    dead_toks_threshold: int,
) -> torch.Tensor:
    """
    Get a boolean mask indicating which latents are "dead" (haven't fired recently).
    
    Args:
        stats_last_nonzero: Tensor of shape (n_dict_components,) tracking tokens since last activation
        dead_toks_threshold: Threshold for considering a feature as dead
        
    Returns:
        Boolean mask of shape (n_dict_components,) where True indicates a dead latent
    """
    return stats_last_nonzero > dead_toks_threshold


def mask_dead_latents(
    x: torch.Tensor,
    stats_last_nonzero: torch.Tensor,
    dead_toks_threshold: Optional[int],
) -> torch.Tensor:
    """
    Mask out alive latents by zeroing those that have been active recently.
    
    Args:
        x: Tensor of shape (..., n_dict_components) - preactivations or similar
        stats_last_nonzero: Tensor of shape (n_dict_components,) tracking tokens since last activation
        dead_toks_threshold: Threshold for considering a feature as dead. If None, returns x unchanged.
        
    Returns:
        Tensor of same shape as x with alive latents zeroed out
    """
    if dead_toks_threshold is None:
        return x
    dead_mask = stats_last_nonzero > dead_toks_threshold
    # Expand dead_mask to match x dimensions
    dead_mask = dead_mask.view(1, -1).expand_as(x)
    return x * dead_mask.to(x.dtype)


def compute_auxk_features(
    preacts: torch.Tensor,
    stats_last_nonzero: torch.Tensor,
    dead_toks_threshold: int,
    aux_k: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute top-k auxiliary features among dead latents for auxiliary loss.
    
    This function identifies "dead" latents (those that haven't fired recently) and
    selects the top-k among them based on their pre-activation values. This provides
    gradient signal to features that would otherwise receive no learning signal.
    
    Args:
        preacts: Pre-activation tensor of shape (batch*seq, n_dict_components) - should be flattened
        stats_last_nonzero: Tensor of shape (n_dict_components,) tracking tokens since last activation
        dead_toks_threshold: Threshold for considering a feature as dead
        aux_k: Number of auxiliary features to select
        
    Returns:
        Tuple of (auxk_values, auxk_indices) each of shape (batch*seq, aux_k), or (None, None) if no dead latents
    """
    # Apply mask to get only dead latents
    masked_preacts = mask_dead_latents(preacts, stats_last_nonzero, dead_toks_threshold)
    
    # Get top-k among dead latents
    if masked_preacts.abs().max() > 0:  # Only if there are dead latents with non-zero preacts
        auxk_values, auxk_indices = torch.topk(
            masked_preacts,
            k=min(aux_k, masked_preacts.shape[-1]),
            dim=-1
        )
        return auxk_values, auxk_indices
    
    return None, None


def compute_auxiliary_reconstruction_loss(
    auxk_indices: torch.Tensor,
    auxk_values: torch.Tensor,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    decoder_bias: torch.Tensor,
    dict_elements: torch.Tensor,
    n_dict_components: int,
) -> torch.Tensor:
    """
    Compute auxiliary reconstruction loss for dead latent mitigation.
    
    This loss reconstructs the residual error (input - main_reconstruction) using
    dead latents to provide gradient signal to features that haven't been active recently.
    
    The loss is computed as normalized MSE between:
    - The auxiliary reconstruction (from dead latents)
    - The residual target (input - main_output + decoder_bias)
    
    Both are normalized to unit norm before MSE computation to balance contributions
    across different magnitude scales.
    
    Args:
        auxk_indices: Indices of top-k dead latents, shape (batch*seq, aux_k)
        auxk_values: Values at those indices, shape (batch*seq, aux_k)
        input_tensor: Original input tensor, shape (batch*seq, input_size) - should be flattened
        output_tensor: Main reconstruction output, shape (batch*seq, input_size) - should be flattened
        decoder_bias: Decoder bias, shape (input_size,)
        dict_elements: Normalized decoder weights, shape (input_size, n_dict_components)
        n_dict_components: Number of dictionary components
        
    Returns:
        Scalar auxiliary loss tensor
        
    Note:
        All batch tensors should be flattened to 2D before calling this function.
        The caller is responsible for handling NaN safety if needed (this function
        includes built-in NaN handling).
    """
    batch_size = input_tensor.shape[0]
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Create sparse representation for auxiliary latents
    aux_c = torch.zeros(batch_size, n_dict_components, device=device, dtype=dtype)
    aux_c.scatter_(-1, auxk_indices, auxk_values)
    
    # Decode auxiliary latents (no bias, as we're reconstructing residual)
    x_hat_aux = F.linear(aux_c, dict_elements)
    
    # Compute residual target: input - main_reconstruction + bias
    # The bias is added back because we want auxiliary latents to help reconstruct
    # the part not captured by main latents
    residual_target = input_tensor - output_tensor.detach() + decoder_bias.detach()
    
    # Normalized MSE for auxiliary loss
    # Avoid division by zero with small epsilon
    residual_norm = torch.norm(residual_target, p=2, dim=-1, keepdim=True)
    aux_recon_norm = torch.norm(x_hat_aux, p=2, dim=-1, keepdim=True)
    
    normalized_aux_loss = F.mse_loss(
        x_hat_aux / (aux_recon_norm + 1e-8),
        residual_target / (residual_norm + 1e-8)
    )
    
    # Safety: Replace NaN with 0 to prevent training instability
    aux_loss = normalized_aux_loss.nan_to_num(0.0)
    
    # Additional check: if aux_loss is still NaN or inf, zero it out
    if torch.isnan(aux_loss).any() or torch.isinf(aux_loss).any():
        aux_loss = torch.zeros_like(aux_loss)
    
    return aux_loss


def update_dead_feature_stats(
    activations: torch.Tensor,
    stats_last_nonzero: torch.Tensor,
    training: bool,
    dead_toks_threshold: Optional[int],
    activation_threshold: float = 1e-3,
) -> None:
    """
    Update dead feature tracking statistics in-place.
    
    A latent is considered "activated" if |activation| > activation_threshold
    for ANY token in the batch. The counter is reset for activated latents
    and incremented by the number of tokens for all latents.
    
    Args:
        activations: Activation tensor with shape (..., n_dict_components)
        stats_last_nonzero: Buffer of shape (n_dict_components,) tracking tokens since last activation
        training: Whether model is in training mode
        dead_toks_threshold: Threshold for dead features. If None, no update is performed.
        activation_threshold: Threshold for considering a feature as active (default: 1e-3)
    """
    if not training or dead_toks_threshold is None:
        return
    
    with torch.no_grad():
        # Flatten batch dimensions
        flat_acts = activations.reshape(-1, activations.shape[-1])
        n_tokens = flat_acts.shape[0]
        
        # A latent is active if |activation| > threshold for any token in batch
        activated_mask = (flat_acts.abs() > activation_threshold).any(dim=0)
        
        # Reset counter for activated latents, increment for all
        stats_last_nonzero *= (~activated_mask).long()
        stats_last_nonzero += n_tokens


def maybe_compute_auxk_features(
    preacts: torch.Tensor,
    stats_last_nonzero: torch.Tensor,
    aux_k: int,
    aux_coeff: float,
    dead_toks_threshold: Optional[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute auxiliary top-k features for dead latents if conditions are met.
    
    This is a convenience wrapper around compute_auxk_features that handles
    the condition checking for whether aux loss should be computed.
    
    Args:
        preacts: Pre-activation tensor with shape (..., n_dict_components)
        stats_last_nonzero: Buffer tracking tokens since last activation
        aux_k: Number of auxiliary features to select
        aux_coeff: Coefficient for auxiliary loss
        dead_toks_threshold: Threshold for considering a feature as dead
        
    Returns:
        Tuple of (auxk_values, auxk_indices) if conditions are met, else (None, None)
    """
    if aux_k > 0 and aux_coeff > 0.0 and dead_toks_threshold is not None:
        # Flatten batch dimensions for consistent computation
        flat_preacts = preacts.reshape(-1, preacts.shape[-1])
        return compute_auxk_features(
            preacts=flat_preacts,
            stats_last_nonzero=stats_last_nonzero,
            dead_toks_threshold=dead_toks_threshold,
            aux_k=aux_k,
        )
    return None, None


def compute_aux_loss_with_logging(
    auxk_indices: Optional[torch.Tensor],
    auxk_values: Optional[torch.Tensor],
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    decoder_bias: torch.Tensor,
    dict_elements: torch.Tensor,
    n_dict_components: int,
    input_size: int,
    aux_k: int,
    aux_coeff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute weighted auxiliary loss and the unweighted version for logging.
    
    This handles the full auxiliary loss computation pattern used in compute_loss,
    including condition checking, tensor flattening, and proper handling of the
    case when no aux loss should be computed.
    
    Args:
        auxk_indices: Indices of top-k dead latents from forward pass, or None
        auxk_values: Values at those indices from forward pass, or None
        input_tensor: Original input tensor (any shape, will be flattened)
        output_tensor: Main reconstruction output (same shape as input)
        decoder_bias: Decoder bias parameter
        dict_elements: Normalized decoder weights
        n_dict_components: Number of dictionary components
        input_size: Size of input/output last dimension
        aux_k: Number of auxiliary features
        aux_coeff: Coefficient for auxiliary loss
        
    Returns:
        Tuple of:
            - weighted_aux_loss: aux_coeff * aux_loss (to add to total_loss)
            - aux_loss_for_logging: detached unweighted aux_loss (for loss_dict)
    """
    device = input_tensor.device
    
    if (aux_k > 0 and aux_coeff > 0.0 and 
        auxk_indices is not None and auxk_values is not None):
        
        # Flatten tensors for auxiliary loss computation
        input_flat = input_tensor.reshape(-1, input_size)
        output_flat = output_tensor.reshape(-1, input_size)
        
        aux_loss = compute_auxiliary_reconstruction_loss(
            auxk_indices=auxk_indices,
            auxk_values=auxk_values,
            input_tensor=input_flat,
            output_tensor=output_flat,
            decoder_bias=decoder_bias,
            dict_elements=dict_elements,
            n_dict_components=n_dict_components,
        )
        
        weighted_aux_loss = aux_coeff * aux_loss
        aux_loss_for_logging = aux_loss.detach().clone()
        
        return weighted_aux_loss, aux_loss_for_logging
    
    # No auxiliary loss - return zeros
    return torch.zeros((), device=device), torch.zeros((), device=device)