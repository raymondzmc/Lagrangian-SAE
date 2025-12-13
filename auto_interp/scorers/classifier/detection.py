from typing import List
from pydantic import BaseModel, Field

from auto_interp.clients import Client
from auto_interp.explainers.explainer import ExplainerResult
from auto_interp.scorers.classifier.classifier import Classifier
from auto_interp.scorers.classifier.prompts.detection_prompt import detection_prompt
from auto_interp.scorers.classifier.sample import Sample, examples_to_samples


class ClassificationOutput(BaseModel):
    predictions: List[int] = Field(
        description="List of 1s and 0s indicating classification results for each example"
    )


class DetectionScorer(Classifier):
    name = "detection"

    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        batch_size: int = 10,
        log_prob: bool = False,
        use_structured_output: bool = True,
        **generation_kwargs,
    ):
        super().__init__(
            client=client,
            verbose=verbose,
            batch_size=batch_size,
            log_prob=log_prob,
            use_structured_output=use_structured_output,
            **generation_kwargs,
        )
        self.prompt = detection_prompt  # Store the prompt function

    def _prepare(self, result: ExplainerResult) -> List[Sample]:
        """
        Prepare samples for detection scoring.
        
        Args:
            result: ExplainerResult containing the feature record and explanation
        
        Returns:
            List of Sample objects ready for classification
        """
        record = result.record
        
        # Convert positive examples to samples with label 1
        positive_samples = examples_to_samples(
            examples=record.positive_examples,
            label=1,
            highlighted=False,
        )
        
        # Convert negative examples to samples with label 0
        negative_samples = examples_to_samples(
            examples=record.negative_examples,
            label=0,
            highlighted=False,
        )
        
        # Combine all samples
        samples = positive_samples + negative_samples
        
        return samples

    def _build_prompt(self, explanation: str, batch: List[Sample]) -> List[dict]:
        """
        Build the full prompt messages for detection scoring.
        
        Args:
            explanation: The explanation to evaluate
            batch: List of samples to classify
        
        Returns:
            List of message dictionaries for the API
        """
        # Format examples text
        examples = self._format_examples(batch)
        
        # Call the prompt function to get the full message list
        messages = self.prompt(examples=examples, explanation=explanation)
        
        return messages
