import torch
import random
from dataclasses import dataclass
from typing import List, Callable
from auto_interp.explainers.features import Example


@dataclass
class Sample:
    text: str
    clean: str
    label: int
    activations: list[float]


def _highlight(tokens: list[str], condition: Callable[[int], bool]) -> str:
    highlighted_tokens = []
    i = 0
    while i < len(tokens):
        if condition(i):
            highlighted_tokens.append("<<")
            while i < len(tokens) and condition(i):
                highlighted_tokens.append(tokens[i])
                i += 1

            highlighted_tokens.append(">>")
        else:
            highlighted_tokens.append(tokens[i])
            i += 1
    return "".join(highlighted_tokens)


def _prepare_text(
    example: Example,
    n_incorrect: int = 0,
    threshold: float = 0.0,
    highlighted: bool = False,
) -> tuple[str, str]:
    """
    Prepare text from an example, optionally highlighting activated tokens.
    
    Args:
        example: Example object with tokens and activations
        n_incorrect: Number of incorrect tokens to add (for fuzzing)
        threshold: Threshold for highlighting tokens (as fraction of max activation)
        highlighted: Whether to highlight tokens
    
    Returns:
        Tuple of (highlighted_text, clean_text)
    """
    # Tokens are already strings
    tokens = example.tokens
    
    assert len(tokens) == len(example.activations), "Number of tokens and activation values must match"
    
    if highlighted:
        activation_threshold = threshold * example.max_activation
        highlighted_tokens = []

        # If no incorrect tokens are needed, highlight consecutive tokens with activations above threshold
        if n_incorrect == 0:
            highlighted_tokens = _highlight(tokens, lambda i: example.activations[i] > activation_threshold)
        else:
            below_threshold = torch.nonzero(example.activations <= activation_threshold).squeeze()
            n_incorrect = min(n_incorrect, len(below_threshold))
            random_indices = set(random.sample(below_threshold.tolist(), n_incorrect))
            highlighted_tokens = _highlight(tokens, lambda i: i in random_indices)

        highlighted_text = "".join(highlighted_tokens)
        clean_text = "".join(tokens)
        return highlighted_text, clean_text
    else:
        text = "".join(tokens)
        return text, text


def examples_to_samples(
    examples: List[Example],
    label: int,
    highlighted: bool = False,
    n_incorrect: int = 0,
    threshold: float = 0,
) -> List[Sample]:
    """
    Convert a list of Example objects to Sample objects.
    
    Args:
        examples: List of Example objects
        label: Label for all samples (1 for positive, 0 for negative)
        highlighted: Whether to highlight tokens based on activations
        n_incorrect: Number of incorrect tokens to add (for fuzzing)
        threshold: Threshold for highlighting tokens
    
    Returns:
        List of Sample objects
    """
    samples = []
    
    for example in examples:
        if highlighted:
            text, clean = _prepare_text(example, n_incorrect, threshold, highlighted)
        else:
            # Tokens are already strings, just join them
            text = "".join(example.tokens)
            clean = text
        
        if isinstance(example.activations, torch.Tensor):
            activations = example.activations.tolist()
        else:
            activations = example.activations
            
        samples.append(Sample(
            text=text,
            clean=clean,
            label=label,
            activations=activations
        ))
    
    return samples
