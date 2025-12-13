from pathlib import Path

import torch
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig
from utils.types import convert_str_to_torch_dtype
from utils.constants import CONFIG_FILE


def load_tlens_model(
    tlens_model_name: str | None, tlens_model_path: Path | None, device: torch.device | None = None
) -> HookedTransformer:
    """Load transformerlens model from either HuggingFace or local path."""
    if tlens_model_name is not None:
        # Load to CPU first to avoid default cuda:0 allocation
        tlens_model = HookedTransformer.from_pretrained(tlens_model_name, device="cpu")
        # Then move to target device if specified
        if device is not None:
            tlens_model = tlens_model.to(device)
    else:
        assert tlens_model_path is not None, "tlens_model_path is None."
        # Load the tlens_config
        assert (
            tlens_model_path.parent / CONFIG_FILE
        ).exists(), f"{CONFIG_FILE} does not exist."
        with open(tlens_model_path.parent / CONFIG_FILE) as f:
            config_data = yaml.safe_load(f)["tlens_config"]
        
        # Handle dtype conversion if present
        if "dtype" in config_data and config_data["dtype"] is not None:
            config_data["dtype"] = convert_str_to_torch_dtype(config_data["dtype"])

        # Load the model with HookedTransformerConfig
        hooked_transformer_config = HookedTransformerConfig(**config_data)
        tlens_model = HookedTransformer(hooked_transformer_config)
        tlens_model.load_state_dict(torch.load(tlens_model_path, map_location="cpu"))
        
        # Move to target device if specified
        if device is not None:
            tlens_model = tlens_model.to(device)

    assert tlens_model.tokenizer is not None
    return tlens_model


def load_pretrained_saes(
    saes: torch.nn.ModuleDict,
    pretrained_sae_paths: list[Path],
    all_param_names: list[str],
    retrain_saes: bool,
) -> list[str]:
    """Load in the pretrained SAEs to saes (in place) and return the trainable param names.

    Args:
        saes: The base SAEs to load the pretrained SAEs into. Updated in place.
        pretrained_sae_paths: List of paths to the pretrained SAEs.
        all_param_names: List of all the parameter names in saes.
        retrain_saes: Whether to retrain the pretrained SAEs.

    Returns:
        The updated all_param_names.
    """
    pretrained_sae_params = {}
    for pretrained_sae_path in pretrained_sae_paths:
        # Add new sae params (note that this will overwrite existing SAEs with the same name)
        pretrained_sae_params = {**pretrained_sae_params, **torch.load(pretrained_sae_path)}
    sae_state_dict = {**dict(saes.named_parameters()), **pretrained_sae_params}

    saes.load_state_dict(sae_state_dict)
    if not retrain_saes:
        # Don't retrain the pretrained SAEs
        trainable_param_names = [
            name for name in all_param_names if name not in pretrained_sae_params
        ]
    else:
        trainable_param_names = all_param_names

    return trainable_param_names
