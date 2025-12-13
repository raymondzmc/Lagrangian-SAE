from math import ceil
from typing import List

import torch

from auto_interp.clients import Client
from auto_interp.explainers.explainer import ExplainerResult
from auto_interp.explainers.features import Example
from auto_interp.scorers.scorer import Scorer
from auto_interp.scorers.classifier.classifier import Classifier
from auto_interp.scorers.classifier.prompts.fuzz_prompt import fuzz_prompt as prompt
from auto_interp.scorers.classifier.sample import Sample, examples_to_samples

class FuzzingScorer(Classifier, Scorer):
    name = "fuzz"

    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        batch_size: int = 1,
        threshold: float = 0.3,
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
        self.threshold = threshold
        self.prompt = prompt
    
    def average_n_activations(self, examples: list[Example], threshold: float) -> float:
        avg = sum(
            len(torch.nonzero(example.activations > (threshold * example.max_activation))) for example in examples
        ) / len(examples)

        return ceil(avg)

    def _prepare(self, result: ExplainerResult) -> List[Sample]:
        """
        Prepare samples for fuzzing scoring.
        
        The fuzzing test checks if the model can identify correctly vs incorrectly
        highlighted tokens. We create:
        - Positive examples with correct highlights (should return 1)
        - Negative examples with random/incorrect highlights (should return 0)
        
        Args:
            result: ExplainerResult containing the feature record and explanation
        
        Returns:
            List of Sample objects ready for classification
        """
        record = result.record
        n_incorrect = self.average_n_activations(record.positive_examples, self.threshold)
        negative_samples = examples_to_samples(
            record.negative_examples,
            label=0,
            highlighted=True,
            n_incorrect=n_incorrect,
            threshold=self.threshold,
        )
        positive_samples = examples_to_samples(
            examples=record.positive_examples,
            label=1,
            highlighted=True,
            n_incorrect=0,
            threshold=self.threshold,
        )

        samples = positive_samples + negative_samples
        return samples

    def _build_prompt(self, explanation: str, batch: List[Sample]) -> List[dict]:
        """
        Build the full prompt messages for fuzzing scoring.
        
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
