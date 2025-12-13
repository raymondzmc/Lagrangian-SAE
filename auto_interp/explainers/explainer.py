import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from transformers import PreTrainedTokenizer

from auto_interp.clients import Client, Response
from auto_interp.explainers.features import Example, FeatureRecord
from auto_interp.explainers.prompts import build_prompt
from utils.logging import logger


@dataclass
class ExplainerResult:
    record: FeatureRecord
    explanation: str


class Explainer(ABC):
    @abstractmethod
    async def __call__(self, record: FeatureRecord) -> ExplainerResult:
        raise NotImplementedError


class DefaultExplainer(Explainer):
    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer | None = None,
        activations: bool = True,
        cot: bool = False,
        threshold: float = 0.3,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.activations = activations
        self.cot = cot
        self.threshold = threshold
        self.generation_kwargs = generation_kwargs
    
    def _highlight(self, index: int, example: Example) -> str:
        result = [f"Example {index}: "]

        threshold = example.max_activation * self.threshold
        activations = example.activations

        if all(isinstance(token, str) for token in example.tokens):
            str_toks = example.tokens
        elif self.tokenizer is not None:
            str_toks = self.tokenizer.batch_decode(example.tokens)
        else:
            raise ValueError("Tokenizer is required to decode examples")

        i = 0
        while i < len(str_toks):
            if activations[i] > threshold:
                result.append("<<")
                while i < len(str_toks) and activations[i] > threshold:
                    result.append(str_toks[i])
                    i += 1
                result.append(">>")
            else:
                result.append(str_toks[i])
                i += 1

        return "".join(result)

    def _join_activations(self, example: Example) -> str:
        activations = []
        threshold = 0.3
        for i, normalized in enumerate(example.normalized_activations):
            if example.normalized_activations[i] > threshold:
                # Use tokens instead of str_toks (str_toks is often None)
                activations.append((example.tokens[i], int(normalized)))

        acts = ", ".join(f'("{item[0]}" : {item[1]})' for item in activations)
        return "Activations: " + acts

    def _build_prompt(self, examples: list[Example]) -> list[dict[str, str]]:
        highlighted_examples: list[str] = []

        for i, example in enumerate(examples):
            highlighted_examples.append(self._highlight(i + 1, example))

            if self.activations:
                highlighted_examples.append(self._join_activations(example))

        highlighted_examples = "\n".join(highlighted_examples)
        return build_prompt(
            examples=highlighted_examples,
            activations=self.activations,
            cot=self.cot,
        )

    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing regex failed: {e}")
            raise e

    async def __call__(self, record: FeatureRecord) -> ExplainerResult:
        messages = self._build_prompt(record.explanation_examples)
        response: Response = await self.client.generate(messages, **self.generation_kwargs)
        return ExplainerResult(
            record=record,
            explanation=self.parse_explanation(response.text)
        )
