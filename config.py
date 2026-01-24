from pathlib import Path
from typing import Any, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from data import DataConfig
from models import create_sae_config
from models.saes import AllSAEConfigs
from utils.enums import SAEType
from settings import settings


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None  # If None, don't log to Weights & Biases
    wandb_run_name: str | None = Field(
        None,
        description="If None, a run_name is generated based on (typically) important config "
        "parameters.",
    )
    wandb_tags: list[str] | None = Field(None, description="Tags to add to the wandb run.")
    seed: NonNegativeInt = Field(
        0,
        description="Seed set at start of script. Also used for train_data.seed and eval_data.seed "
        "if they are not set explicitly.",
    )
    tlens_model_name: str | None = None
    tlens_model_path: Path | None = Field(
        None,
        description="Path to '.pt' checkpoint. The directory housing this file should also contain `CONFIG_FILE`."
    )
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        "float32",
        description="Data type for model and SAE training. Use 'bfloat16' for larger models like Gemma-2."
    )
    use_torch_compile: bool = Field(
        False,
        description="Enable torch.compile for the transformer model. Uses PyTorch's inductor "
        "backend which can optimize attention to use flash attention when available."
    )
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        "reduce-overhead",
        description="torch.compile mode. 'reduce-overhead' is good for inference-heavy workloads, "
        "'max-autotune' provides best performance but longer initial compilation."
    )
    save_dir: Path | None = settings.output_dir

    save_every_n_samples: PositiveInt | None = None
    eval_every_n_samples: PositiveInt | None = Field(
        None, description="If None, don't evaluate. If 0, only evaluate at the end."
    )
    gradient_accumulation_steps: PositiveInt = 1
    lr: PositiveFloat
    lr_schedule: Literal["linear", "cosine", "dictionary_learning"] = "cosine"
    min_lr_factor: NonNegativeFloat = Field(
        0.1,
        description="The minimum learning rate as a factor of the initial learning rate. Used "
        "in the cooldown phase of a linear or cosine schedule.",
    )
    warmup_samples: NonNegativeInt = 0
    cooldown_samples: NonNegativeInt = 0
    decay_start_fraction: float = Field(
        0.8,
        description="Fraction of training at which LR decay starts (for dictionary_learning schedule). "
        "E.g., 0.8 means decay starts at 80% of training.",
    )
    max_grad_norm: PositiveFloat | None = None
    log_every_n_grad_steps: PositiveInt = 20
    # collect_act_frequency_every_n_samples: NonNegativeInt = Field(
    #     20_000,
    #     description="Metrics such as activation frequency and alive neurons are calculated over "
    #     "fixed number of batches. This parameter specifies how often to calculate these metrics.",
    # )
    # act_frequency_n_tokens: PositiveInt = Field(
    #     100_000, description="The number of tokens to caclulate activation frequency metrics over."
    # )
    data: DataConfig = Field(..., description="Data configuration with train/eval sample counts")
    saes: AllSAEConfigs = Field(..., description="SAE configuration")
    
    @model_validator(mode="before")
    @classmethod
    def validate_sae_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate and create the appropriate SAE config based on sae_type."""
        if "saes" in values and isinstance(values["saes"], dict):
            # Use factory function to create the correct SAE config type
            values["saes"] = create_sae_config(values["saes"])
            
        return values

    @model_validator(mode="before")
    @classmethod
    def check_only_one_model_definition(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert (values.get("tlens_model_name") is not None) + (
            values.get("tlens_model_path") is not None
        ) == 1, "Must specify exactly one of tlens_model_name or tlens_model_path."
        return values

    @model_validator(mode="after")
    def verify_valid_eval_settings(self) -> "Config":
        """User can't provide eval_every_n_samples without both eval_n_samples and data.n_eval_samples."""
        if self.eval_every_n_samples is not None:
            assert (
                self.data.n_eval_samples is not None and self.data.n_eval_samples > 0
            ), "Must provide data.n_eval_samples when using eval_every_n_samples."
        return self

    @model_validator(mode="after")
    def cosine_schedule_requirements(self) -> "Config":
        """Cosine schedule must have data.n_train_samples set in order to define the cosine curve."""
        if self.lr_schedule == "cosine":
            assert self.data.n_train_samples is not None, "Cosine schedule requires data.n_train_samples."
            assert self.cooldown_samples == 0, "Cosine schedule must not have cooldown_samples."
        return self

    @model_validator(mode="after")
    def validate_gradient_accumulation_divisibility(self) -> "Config":
        """Ensure train_batch_size is divisible by gradient_accumulation_steps."""
        if self.data.train_batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"train_batch_size ({self.data.train_batch_size}) must be divisible by "
                f"gradient_accumulation_steps ({self.gradient_accumulation_steps})"
            )
        return self

    @property
    def mini_batch_size(self) -> int:
        """Mini-batch size for each forward pass (train_batch_size / gradient_accumulation_steps)."""
        return self.data.train_batch_size // self.gradient_accumulation_steps

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size is now just the train_batch_size (total batch size per gradient update)."""
        return self.data.train_batch_size
