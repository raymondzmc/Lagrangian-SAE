import random
from collections import deque
import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_interp.explainers.features import Example


def split_activation_quantiles(
    examples: list["Example"], 
    n_quantiles: int,
    n_samples: int,
    seed: int = 22
):
    random.seed(seed)

    max_activation = examples[0].max_activation
    thresholds = [max_activation * i / n_quantiles for i in range(1, n_quantiles)]

    samples = []
    examples = deque(examples)

    for threshold in thresholds:
        quantile = []
        while examples and examples[0].max_activation < threshold:
            quantile.append(examples.popleft())

        sample = random.sample(quantile, n_samples)
        samples.append(sample)

    sample = random.sample(examples, n_samples)
    samples.append(sample)
    
    return samples


def stratified_sample_by_max_activation(
    neuron_activations: torch.Tensor,
    n_samples: int,
    n_quantiles: int = 20,
    seed: int = 22
) -> torch.Tensor:
    """
    Perform stratified sampling based on max activation values.
    
    Args:
        neuron_activations: Tensor of shape (n_examples, seq_len) containing activations
        n_samples: Total number of samples to select
        n_quantiles: Number of quantiles to split the data into (default: 4)
        seed: Random seed for reproducibility (default: 22)
        
    Returns:
        Tensor of indices for selected examples, sorted by activation value (descending)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Calculate max activation per example (row)
    max_activations_per_example = neuron_activations.max(dim=1).values  # (n_examples,)
    
    # Get sorted indices (descending order)
    sorted_indices = torch.argsort(max_activations_per_example, descending=True)    
    n_examples = len(sorted_indices)
    
    # Handle edge cases
    if n_samples >= n_examples:
        return sorted_indices[:n_samples] if n_samples <= n_examples else sorted_indices
    
    # Calculate samples per quantile
    samples_per_quantile = n_samples // n_quantiles
    remaining_samples = n_samples % n_quantiles
    
    # Calculate quantile boundaries based on position in sorted array
    quantile_boundaries = []
    for i in range(n_quantiles):
        if i == 0:
            start_idx = 0
        else:
            start_idx = quantile_boundaries[-1][1]
        
        if i == n_quantiles - 1:
            end_idx = n_examples
        else:
            # Split based on equal portions of the sorted data
            end_idx = min((i + 1) * n_examples // n_quantiles, n_examples)
        
        quantile_boundaries.append((start_idx, end_idx))
    
    # Sample from each quantile
    selected_indices = []
    
    for i, (start_idx, end_idx) in enumerate(quantile_boundaries):
        quantile_size = end_idx - start_idx
        
        # Calculate how many samples to take from this quantile
        if i < remaining_samples:
            # Distribute remaining samples to first quantiles (highest activations)
            n_to_sample = samples_per_quantile + 1
        else:
            n_to_sample = samples_per_quantile
        
        # Ensure we don't sample more than available in the quantile
        n_to_sample = min(n_to_sample, quantile_size)
        
        if n_to_sample > 0:
            # Get indices within this quantile
            quantile_indices = sorted_indices[start_idx:end_idx]
            
            # Randomly sample from this quantile
            if n_to_sample >= len(quantile_indices):
                sampled = quantile_indices
            else:
                # Random sampling without replacement
                perm = torch.randperm(len(quantile_indices))[:n_to_sample]
                sampled = quantile_indices[perm]
            
            selected_indices.append(sampled)
    
    # Concatenate all selected indices
    if selected_indices:
        all_selected = torch.cat(selected_indices)
        
        # Re-sort the final indices by their activation values (descending)
        # This maintains consistency with the original behavior where top_indices were sorted
        selected_activations = max_activations_per_example[all_selected]
        sorted_order = torch.argsort(selected_activations, descending=True)
        return all_selected[sorted_order]
    else:
        return torch.tensor([], dtype=torch.long)
