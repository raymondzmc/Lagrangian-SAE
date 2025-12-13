import torch
from typing import TypeVar
from pydantic import BaseModel


BaseModelType = TypeVar("BaseModelType", bound=BaseModel)

# Define torch dtype mapping for manual conversion
TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

def convert_str_to_torch_dtype(dtype_str):
    """Convert dtype from str to torch dtype."""
    if dtype_str in TORCH_DTYPES:
        return TORCH_DTYPES[dtype_str]
    elif dtype_str in TORCH_DTYPES.values():
        return dtype_str
    else:
        raise ValueError(f"Invalid dtype: {dtype_str}")