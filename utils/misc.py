import torch
import random
import numpy as np
from typing import Any

from config import Config
from utils.types import BaseModelType


def deep_update(base_dict: dict, update_dict: dict) -> dict:
    result = base_dict.copy()
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def replace_pydantic_model(model: BaseModelType, *updates: dict[str, Any]) -> BaseModelType:
    updated_data = model.model_dump()
    for update in updates:
        updated_data = deep_update(updated_data, update)
    return model.__class__(**updated_data)


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def filter_names(all_names: list[str], filter_names: list[str] | str) -> list[str]:
    """Use filter_names to filter `all_names` by partial match.

    The filtering is done by checking if any of the filter_names are in the all_names. Partial
    matches are allowed. E.g. "hook_resid_pre" matches ["blocks.0.hook_resid_pre",
    "blocks.1.hook_resid_pre", ...].

    Args:
        all_names: The names to filter.
        filter_names: The names to use to filter all_names by partial match.
    Returns:
        The filtered names.
    """
    if isinstance(filter_names, str):
        filter_names = [filter_names]
    return [name for name in all_names if any(filter_name in name for filter_name in filter_names)]


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_name = config.wandb_run_name
    else:
        run_name = ""
        if config.tlens_model_name:
            run_name += f"{config.tlens_model_name.split('/')[-1]}_"
        
        run_name += f"{config.saes.sae_type.value}_"

        if config.saes.sparsity_coeff is not None:
            run_name += f"sparsity-{config.saes.sparsity_coeff}_"
        if config.saes.mse_coeff is not None:
            run_name += f"mse-{config.saes.mse_coeff}_"

        run_name += f"ratio-{config.saes.dict_size_to_input_ratio}_"
        run_name += f"lr-{config.lr}_"
        run_name += f"{'-'.join(config.saes.sae_positions)}"
    return run_name


class RunningAverage:
    def __init__(self):
        self.sum = 0.0
        self.weight = 0.0

    def add(self, value, weight=1.0):
        self.sum += float(value) * float(weight)
        self.weight += float(weight)

    def mean(self):
        return self.sum / max(self.weight, 1e-8)

    def reset(self):
        self.sum = 0.0
        self.weight = 0.0
