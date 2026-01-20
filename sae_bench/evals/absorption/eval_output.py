from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass

from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)

EVAL_TYPE_ID_ABSORPTION = "absorption_first_letter"


# Define the metrics for each metric category, and include a title and description for each.
@dataclass
class AbsorptionMeanMetrics(BaseMetrics):
    mean_absorption_fraction_score: float = Field(
        title="Mean Absorption Fraction Score",
        description="Average of the absorption fraction scores across all letters",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    mean_full_absorption_score: float = Field(
        title="Mean Full Absorption Score",
        description="Average of the full absorption scores across all letters",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    mean_num_split_features: float = Field(
        title="Mean Number of Split Features",
        description="Average number of split features across all letters",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    std_dev_absorption_fraction_score: float = Field(
        title="Standard Deviation of Absorption Fraction Score",
        description="Standard deviation of the absorption fraction scores across all letters",
    )
    std_dev_full_absorption_score: float = Field(
        title="Standard Deviation of Full Absorption Score",
        description="Standard deviation of the full absorption scores across all letters",
    )
    std_dev_num_split_features: float = Field(
        title="Standard Deviation of Number of Split Features",
        description="Standard deviation of the number of split features across all letters",
    )


# Define the categories themselves, and include a title and description for each.
@dataclass
class AbsorptionMetricCategories(BaseMetricCategories):
    mean: AbsorptionMeanMetrics = Field(
        title="Mean",
        description="Mean metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )


# Define a result detail, which in this case is an absorption result for a single letter.
@dataclass
class AbsorptionResultDetail(BaseResultDetail):
    first_letter: str = Field(title="First Letter", description="")

    @field_validator("first_letter")
    @classmethod
    def validate_single_letter(cls, value: str) -> str:
        if len(value) == 1 and value.isalpha():
            return value
        raise ValueError("First letter must be a single letter")

    mean_absorption_fraction: float = Field(
        title="Mean Absorption Fraction", description=""
    )
    full_absorption_rate: float = Field(title="Rate of Full Absorption", description="")
    num_full_absorption: int = Field(title="Num Full Absorption", description="")
    num_probe_true_positives: int = Field(
        title="Num Probe True Positives", description=""
    )
    num_split_features: int = Field(title="Num Split Features", description="")


# Define the eval output, which includes the eval config, metrics, and result details.
# The title will end up being the title of the eval in the UI.
@dataclass(config=ConfigDict(title="Absorption"))
class AbsorptionEvalOutput(
    BaseEvalOutput[
        AbsorptionEvalConfig, AbsorptionMetricCategories, AbsorptionResultDetail
    ]
):
    # This will end up being the description of the eval in the UI.
    """
    The feature absorption evaluation looking at the first letter.
    """

    eval_config: AbsorptionEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: AbsorptionMetricCategories
    eval_result_details: list[AbsorptionResultDetail] = Field(
        default_factory=list,
        title="Per-Letter Absorption Results",
        description="Each object is a stat on the first letter of the absorption.",
    )
    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_ABSORPTION,
        title="Eval Type ID",
        description="The type of the evaluation",
    )
