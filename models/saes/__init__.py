from models.saes.base import SAEConfig, SAEOutput, BaseSAE, SAELoss
from models.saes.relu_sae import ReluSAE, ReLUSAEConfig
from models.saes.lagrangian_sae import LagrangianSAE, LagrangianSAEConfig, LagrangianSAEOutput
from models.saes.gated_sae import GatedSAE, GatedSAEConfig, GatedSAEOutput
from models.saes.topk_sae import TopKSAE, TopKSAEConfig, TopKSAEOutput
from models.saes.batch_topk_sae import BatchTopKSAE, BatchTopKSAEConfig, BatchTopKSAEOutput
from models.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig, JumpReLUSAEOutput
from models.saes.matryoshka_sae import MatryoshkaSAE, MatryoshkaSAEConfig, MatryoshkaSAEOutput
from models.saes.matryoshka_lagrangian_sae import MatryoshkaLagrangianSAE, MatryoshkaLagrangianSAEConfig, MatryoshkaLagrangianSAEOutput
from utils.enums import SAEType
from typing import Any, Union
import inspect


ALL_SAE_CONFIGS = [
    cls for name, cls in globals().items() 
    if inspect.isclass(cls) and issubclass(cls, SAEConfig) and cls is not SAEConfig
]

# Union type for type annotations (Python 3.9 compatible)
AllSAEConfigs = Union[
    ReLUSAEConfig,
    LagrangianSAEConfig,
    GatedSAEConfig,
    TopKSAEConfig,
    BatchTopKSAEConfig,
    JumpReLUSAEConfig,
    MatryoshkaSAEConfig,
    MatryoshkaLagrangianSAEConfig,
]

SAE_TYPE_TO_CONFIG = {
    SAEType.LAGRANGIAN: LagrangianSAEConfig,
    SAEType.RELU: ReLUSAEConfig,
    SAEType.GATED: GatedSAEConfig,
    SAEType.TOPK: TopKSAEConfig,
    SAEType.BATCH_TOPK: BatchTopKSAEConfig,
    SAEType.JUMP_RELU: JumpReLUSAEConfig,
    SAEType.MATRYOSHKA: MatryoshkaSAEConfig,
    SAEType.MATRYOSHKA_LAGRANGIAN: MatryoshkaLagrangianSAEConfig,
}


SAE_TYPE_TO_CLS = {
    SAEType.LAGRANGIAN: LagrangianSAE,
    SAEType.RELU: ReluSAE,
    SAEType.GATED: GatedSAE,
    SAEType.TOPK: TopKSAE,
    SAEType.BATCH_TOPK: BatchTopKSAE,
    SAEType.JUMP_RELU: JumpReLUSAE,
    SAEType.MATRYOSHKA: MatryoshkaSAE,
    SAEType.MATRYOSHKA_LAGRANGIAN: MatryoshkaLagrangianSAE,
}

assert set(SAE_TYPE_TO_CONFIG.keys()) == set(SAE_TYPE_TO_CLS.keys()), f"SAE_TYPE_TO_CONFIG.keys(): {SAE_TYPE_TO_CONFIG.keys()} != SAE_TYPE_TO_CLS.keys(): {SAE_TYPE_TO_CLS.keys()}"
assert set(ALL_SAE_CONFIGS) == set(SAE_TYPE_TO_CONFIG.values()), f"ALL_SAE_CONFIGS: {ALL_SAE_CONFIGS} != SAE_TYPE_TO_CONFIG.values(): {SAE_TYPE_TO_CONFIG.values()}"


def create_sae_config(config_dict: dict[str, Any]) -> SAEConfig:
    """Factory function to create the appropriate SAE config based on sae_type.
    
    Args:
        config_dict: Dictionary containing SAE configuration parameters
        
    Returns:
        Appropriate SAEConfig subclass instance
        
    Raises:
        NotImplementedError: If sae_type is not supported
        ValueError: If sae_type is missing from config_dict
    """
    if "sae_type" not in config_dict:
        raise ValueError("sae_type must be specified in SAE config")
    
    try:
        sae_type = SAEType(config_dict["sae_type"])
    except ValueError:
        raise ValueError(f"Invalid sae_type: {config_dict['sae_type']}")
    
    return SAE_TYPE_TO_CONFIG[sae_type].model_validate(config_dict)


__all__ = [
    "BaseSAE",
    "SAEConfig", 
    "SAELoss",
    "SAEOutput",
    "ReluSAE",
    "ReLUSAEConfig",
    "LagrangianSAE",
    "LagrangianSAEConfig",
    "LagrangianSAEOutput",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedSAEOutput",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKSAEOutput",
    "BatchTopKSAE",
    "BatchTopKSAEConfig",
    "BatchTopKSAEOutput",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "JumpReLUSAEOutput",
    "MatryoshkaSAE",
    "MatryoshkaSAEConfig",
    "MatryoshkaSAEOutput",
    "MatryoshkaLagrangianSAE",
    "MatryoshkaLagrangianSAEConfig",
    "MatryoshkaLagrangianSAEOutput",
    "create_sae_config",
    "AllSAEConfigs",
]
