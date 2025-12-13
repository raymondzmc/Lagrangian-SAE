import json
import random
import re
import numpy as np
from abc import abstractmethod
from typing import List

from pydantic import BaseModel, Field

from auto_interp.clients import Client, LogProb
from auto_interp.explainers.explainer import ExplainerResult
from auto_interp.scorers.classifier.sample import Sample
from auto_interp.scorers.scorer import Scorer, ScorerResult
from utils.logging import logger


class ClassificationOutput(BaseModel):
    """Structured output model for classification results."""
    predictions: List[int] = Field(
        description="List of 1s and 0s indicating classification results for each example"
    )


class Classifier(Scorer):
    """Base class for classification-based scorers."""
    
    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        batch_size: int = 10,
        log_prob: bool = False,
        use_structured_output: bool = True,
        **generation_kwargs,
    ):
        """
        Initialize the classifier.
        
        Args:
            client: Client for generating predictions
            verbose: Whether to print verbose output
            batch_size: Number of samples to process at once
            log_prob: Whether to use log probabilities for scoring
            use_structured_output: Whether to use structured output (Pydantic model)
            **generation_kwargs: Additional generation parameters
        """
        self.client = client
        self.verbose = verbose
        self.batch_size = batch_size
        self.log_prob = log_prob
        self.use_structured_output = use_structured_output
        self.generation_kwargs = generation_kwargs

    @abstractmethod
    def _prepare(self, result: ExplainerResult) -> List[Sample]:
        """Prepare samples from the explainer result."""
        raise NotImplementedError

    @abstractmethod
    def _build_prompt(self, explanation: str, batch: List[Sample]) -> List[dict]:
        """
        Build the full prompt messages for the classifier.
        
        Args:
            explanation: The explanation to evaluate
            batch: List of samples to classify
            
        Returns:
            List of message dictionaries for the API
        """
        raise NotImplementedError

    async def __call__(self, result: ExplainerResult) -> ScorerResult:
        """
        Score the explanation using classification.
        
        Args:
            result: ExplainerResult containing the explanation and examples
        
        Returns:
            ScorerResult with accuracy score
        """
        # Prepare samples
        samples = self._prepare(result)
        
        if not samples:
            logger.warning("No samples to classify")
            return ScorerResult(record=result.record, score=0.0)
        
        # Shuffle samples for better mixing of positive/negative
        random.shuffle(samples)
        
        # Process in batches
        all_predictions = []
        all_labels = []
        
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            
            # Build the full prompt messages
            messages = self._build_prompt(result.explanation, batch)
            
            # Generate predictions
            if self.use_structured_output:
                response = await self.client.generate(
                    messages=messages,
                    response_model=ClassificationOutput,
                    **self.generation_kwargs
                )
                
                if response.structured_response:
                    predictions = response.structured_response.predictions
                    # Convert to float for consistency
                    predictions = [float(p) for p in predictions]
                else:
                    # Fallback to parsing text response
                    predictions = self._parse_predictions(response.text, len(batch))
                
                # If we also want logprobs for confidence scoring
                if self.log_prob and response.logprobs:
                    # We can use logprobs to get confidence scores
                    confidences = self._extract_probabilities_from_logprobs(response.logprobs)
                    # Optionally combine structured predictions with confidence scores
                    # For now, just use the structured predictions
            else:
                response = await self.client.generate(
                    messages=messages,
                    **self.generation_kwargs
                )
                
                if self.log_prob and response.logprobs:
                    # Extract probabilities from logprobs
                    predictions = self._extract_probabilities_from_logprobs(response.logprobs)
                else:
                    # Parse text predictions
                    predictions = self._parse_predictions(response.text, len(batch))
            
            # Ensure we have the right number of predictions
            if len(predictions) < len(batch):
                # Pad with uncertain predictions
                predictions.extend([0.5] * (len(batch) - len(predictions)))
            elif len(predictions) > len(batch):
                # Truncate to batch size
                predictions = predictions[:len(batch)]
            
            # Collect predictions and labels
            all_predictions.extend(predictions)
            all_labels.extend([s.label for s in batch])
            
            if self.verbose:
                for sample, pred in zip(batch, predictions):
                    correct = "✓" if (pred > 0.5 and sample.label == 1) or (pred <= 0.5 and sample.label == 0) else "✗"
                    print(f"  {correct} Predicted: {pred:.2f}, Actual: {sample.label}, Text: {sample.text[:50]}...")
        
        # Calculate accuracy
        correct = sum(
            1 for pred, label in zip(all_predictions, all_labels)
            if (pred > 0.5 and label == 1) or (pred <= 0.5 and label == 0)
        )
        accuracy = correct / len(all_labels) if all_labels else 0.0
        
        if self.verbose:
            print(f"Accuracy: {accuracy:.3f} ({correct}/{len(all_labels)})")
        
        return ScorerResult(record=result.record, score=accuracy)

    def _format_examples(self, samples: List[Sample]) -> str:
        """Format samples into text for the prompt."""
        examples = []
        for i, sample in enumerate(samples):
            examples.append(f"Example {i}: {sample.text}")
        return "\n".join(examples)

    def _parse_predictions(self, text: str, expected_count: int) -> List[float]:
        """
        Parse predictions from text response.
        
        Args:
            text: Response text containing predictions
            expected_count: Expected number of predictions
        
        Returns:
            List of prediction probabilities (0.0 or 1.0)
        """
        # Try to extract list format first [1, 0, 1, ...]
        list_match = re.search(r'\[([^\]]+)\]', text)
        if list_match:
            try:
                values = json.loads(f"[{list_match.group(1)}]")
                if len(values) == expected_count:
                    return [float(v) for v in values]
            except:
                pass
        
        # Try to extract individual numbers
        numbers = re.findall(r'\b[01]\b', text)
        if len(numbers) == expected_count:
            return [float(n) for n in numbers]
        
        # Fallback: return 0.5 for all (uncertain)
        logger.warning(f"Could not parse predictions from response: {text[:100]}...")
        return [0.5] * expected_count

    def _extract_probabilities_from_logprobs(self, logprobs: List[LogProb]) -> List[float]:
        """
        Extract classification probabilities from logprobs.
        
        This method analyzes the log probabilities of tokens to determine
        the probability of positive classification (1) vs negative (0).
        
        Args:
            logprobs: List of LogProb objects from the response
        
        Returns:
            List of probabilities for positive classification
        """
        probabilities = []
        
        for logprob in logprobs:
            # Get the token that was actually generated
            token = logprob.token.strip()
            
            # Calculate P(1|context) by looking at alternatives
            p_positive = 0.0
            p_negative = 0.0
            
            for alt in logprob.top_logprobs:
                alt_token = alt.token.strip()
                prob = np.exp(alt.logprob)
                
                if alt_token == "1":
                    p_positive += prob
                elif alt_token == "0":
                    p_negative += prob
            
            # Normalize to get probability of positive class
            total = p_positive + p_negative
            if total > 0:
                prob_positive = p_positive / total
            else:
                # If we couldn't determine from logprobs, use the actual token
                prob_positive = 1.0 if token == "1" else 0.0
            
            probabilities.append(prob_positive)
        
        return probabilities
