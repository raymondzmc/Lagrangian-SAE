from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)

EVAL_TYPE_ID_AUTOINTERP = "autointerp"


@dataclass
class AutoInterpMetrics(BaseMetrics):
    autointerp_score: float = Field(
        title="AutoInterp Score",
        description="AutoInterp detection score, using methodology similar to Eleuther's 'Open Source Automated Interpretability for Sparse Autoencoder Features'",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    autointerp_std_dev: float = Field(
        title="AutoInterp Standard Deviation",
        description="AutoInterp detection score standard deviation over all tested features",
    )


# Define the categories themselves
@dataclass
class AutoInterpMetricCategories(BaseMetricCategories):
    autointerp: AutoInterpMetrics = Field(
        title="AutoInterp",
        description="Metrics related to autointerp",
    )


# Define the eval output
@dataclass(config=ConfigDict(title="AutoInterp"))
class AutoInterpEvalOutput(
    BaseEvalOutput[AutoInterpEvalConfig, AutoInterpMetricCategories, BaseResultDetail]  # type: ignore
):
    """
    An evaluation of the interpretability of SAE latents. This evaluation is based on Eleuther's 'Open Source Automated Interpretability for Sparse Autoencoder Features'
    """

    eval_config: AutoInterpEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: AutoInterpMetricCategories

    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_AUTOINTERP,
        title="Eval Type ID",
        description="The type of the evaluation",
    )
