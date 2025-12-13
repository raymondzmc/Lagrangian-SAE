import openai
import together
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Type, List
from pydantic import BaseModel, ValidationError
from together.types.chat_completions import ChatCompletionChoicesData, LogprobsPart
import together.error
import asyncio
import random
import logging
from functools import wraps

T = TypeVar('T', bound=BaseModel)

# Set up logging
logger = logging.getLogger(__name__)


async def retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: The async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
    
    Returns:
        The result of the function call
    
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            
            # Check if this is a retryable error
            is_retryable = False
            error_message = str(e).lower()
            
            # Together AI errors
            if hasattr(together.error, 'ServiceUnavailableError') and isinstance(e, together.error.ServiceUnavailableError):
                is_retryable = True
            elif hasattr(together.error, 'RateLimitError') and isinstance(e, together.error.RateLimitError):
                is_retryable = True
            elif hasattr(together.error, 'APITimeoutError') and isinstance(e, together.error.APITimeoutError):
                is_retryable = True
            elif hasattr(together.error, 'APIConnectionError') and isinstance(e, together.error.APIConnectionError):
                is_retryable = True
            # Fallback to checking class name for Together AI errors
            elif hasattr(e, '__class__'):
                error_name = e.__class__.__name__
                if any(err in error_name for err in ['ServiceUnavailableError', 'RateLimitError', 'APITimeoutError', 'APIConnectionError']):
                    is_retryable = True
            
            # OpenAI errors  
            if isinstance(e, openai.RateLimitError):
                is_retryable = True
            elif isinstance(e, openai.APITimeoutError):
                is_retryable = True
            elif isinstance(e, openai.APIConnectionError):
                is_retryable = True
            elif isinstance(e, openai.InternalServerError):
                is_retryable = True
            elif isinstance(e, openai.APIError):
                # Check for specific status codes
                if hasattr(e, 'status_code'):
                    if e.status_code in [429, 500, 502, 503, 504]:
                        is_retryable = True
                        
            # Check error message for common patterns
            if '503' in error_message or 'service unavailable' in error_message:
                is_retryable = True
            elif '429' in error_message or 'rate limit' in error_message:
                is_retryable = True
            elif 'timeout' in error_message:
                is_retryable = True
            elif 'connection' in error_message and 'error' in error_message:
                is_retryable = True
                
            if not is_retryable:
                logger.error(f"Non-retryable error: {e}")
                raise
            
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                raise
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter if enabled
            if jitter:
                delay = delay * (0.5 + random.random())
            
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed with {error_name if 'error_name' in locals() else type(e).__name__}: {e}. "
                f"Retrying in {delay:.2f} seconds..."
            )
            
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in retry logic")


@dataclass
class TopLogProb:
    """Represents a single token alternative with its log probability."""
    token: str
    logprob: float


@dataclass  
class LogProb:
    """Represents a token with its alternatives."""
    token: str
    logprob: float
    top_logprobs: List[TopLogProb]


@dataclass
class Response:
    text: str
    logprobs: List[LogProb]
    cost: float
    input_tokens: int
    output_tokens: int
    structured_response: Optional[BaseModel] = None


class Client(ABC):
    api_key: str
    model: str
    max_logprobs: int
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    @abstractmethod
    async def generate(
        self, 
        messages: list[dict[str, str]],
        response_model: Optional[Type[T]] = None,
        **kwargs
    ) -> Response:
        raise NotImplementedError

# OpenAI model mappings and pricing (per 1M tokens)
OPENAI_MODELS = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-11-20",
}

# OpenAI pricing per 1M tokens (input, output)
OPENAI_PRICING = {
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
}


class OpenAIClient(Client):
    max_logprobs: int = 20

    def __init__(self, api_key: str, model: str, max_retries: int = 5, 
                 base_delay: float = 1.0, max_delay: float = 60.0,
                 exponential_base: float = 2.0, jitter: bool = True):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        if model not in OPENAI_MODELS:
            raise ValueError(f"Invalid model: {model}. Available models: {list(OPENAI_MODELS.keys())}")
        self.model_name = OPENAI_MODELS[model]
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    async def generate(self, messages: list[dict[str, str]], response_model: Optional[Type[T]] = None, **kwargs) -> Response:
        """Async generation with optional structured output."""
        structured_response = None
        logprobs = []

        if "logprobs" not in kwargs:
            kwargs["logprobs"] = True
        if "top_logprobs" not in kwargs and kwargs.get("logprobs"):
            kwargs["top_logprobs"] = self.max_logprobs

        # Wrap API calls in retry logic
        async def _make_api_call():
            if response_model:
                return await self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_model,
                    **kwargs
                )
            else:
                return await self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages, 
                    **kwargs
                )
        
        # Execute with retry
        response = await retry_with_exponential_backoff(
            _make_api_call,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            jitter=self.jitter
        )
        
        if response_model:
            structured_response = response.choices[0].message.parsed
        
        text = response.choices[0].message.content or ""
        
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            for token_data in response.choices[0].logprobs.content:
                top_logprobs = []
                if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                    for alt in token_data.top_logprobs:
                        top_logprobs.append(TopLogProb(
                            token=alt.token,
                            logprob=alt.logprob
                        ))
                logprobs.append(LogProb(
                    token=token_data.token,
                    logprob=token_data.logprob,
                    top_logprobs=top_logprobs
                ))

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        pricing = OPENAI_PRICING.get(self.model_name, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] / 1_000_000) + (output_tokens * pricing["output"] / 1_000_000)
        
        return Response(
            text=text,
            logprobs=logprobs,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            structured_response=structured_response,
        )


# TogetherAI model mappings
TOGETHERAI_MODELS = {
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
}

# TogetherAI pricing per 1M tokens (input, output) - approximate values
TOGETHERAI_PRICING = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": {"input": 0.18, "output": 0.18},
}


class TogetherAIClient(Client):
    max_logprobs: int = 5

    def __init__(self, api_key: str, model: str, max_retries: int = 5,
                 base_delay: float = 1.0, max_delay: float = 60.0,
                 exponential_base: float = 2.0, jitter: bool = True):
        self.client = together.AsyncTogether(api_key=api_key)
        if model not in TOGETHERAI_MODELS:
            raise ValueError(f"Invalid model: {model}. Available models: {list(TOGETHERAI_MODELS.keys())}")
        self.model_name = TOGETHERAI_MODELS[model]
        # Llama 3.2 doesn't support alternatives, so set max to 1
        if "11B" in self.model_name:
            self.max_logprobs = 1
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    async def generate(self, messages: list[dict[str, str]], response_model: Optional[Type[T]] = None, **kwargs) -> Response:
        """Async generation with optional structured output."""
        structured_response = None
        logprobs = []

        # TogetherAI uses 'logprobs' as an integer, not 'top_logprobs'
        # Cap at model's maximum
        kwargs["logprobs"] = min(self.max_logprobs, kwargs.get("logprobs", self.max_logprobs))
        
        if response_model is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "schema": response_model.model_json_schema(),
            }

        # Wrap API call in retry logic
        async def _make_api_call():
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
        
        # Execute with retry
        try:
            response = await retry_with_exponential_backoff(
                _make_api_call,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
                exponential_base=self.exponential_base,
                jitter=self.jitter
            )
        except Exception as e:
            # Handle grammar compilation errors by falling back to non-structured output
            if "grammar is not valid" in str(e) and response_model is not None:
                print(f"Warning: Grammar compilation failed for structured output, falling back to text parsing: {e}")
                # Remove structured output and retry
                fallback_kwargs = kwargs.copy()
                fallback_kwargs.pop("response_format", None)
                
                async def _make_fallback_api_call():
                    return await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **fallback_kwargs
                    )
                
                response = await retry_with_exponential_backoff(
                    _make_fallback_api_call,
                    max_retries=self.max_retries,
                    base_delay=self.base_delay,
                    max_delay=self.max_delay,
                    exponential_base=self.exponential_base,
                    jitter=self.jitter
                )
            else:
                raise
        
        if len(response.choices) == 0:
            raise Exception(f"No response choices from model: {response}")

        response_choice: ChatCompletionChoicesData = response.choices[0]
        text = response_choice.message.content or ""
        if text == "":
            raise Exception(f"No text in response choice: {response_choice}")

        # Parse structured response
        if response_model is not None:
            try:
                structured_response = response_model.model_validate_json(text)
            except ValidationError as e:
                # If structured parsing fails, we still have the text response
                # The classifier will handle text parsing as fallback
                print(f"Warning: Failed to parse structured response, will use text parsing fallback: {e}")
                structured_response = None

        # Extract logprobs if available
        if response_choice.logprobs:
            logprobs_data: LogprobsPart = response_choice.logprobs
            tokens = logprobs_data.tokens if hasattr(logprobs_data, 'tokens') and logprobs_data.tokens else []
            token_logprobs = logprobs_data.token_logprobs if hasattr(logprobs_data, 'token_logprobs') and logprobs_data.token_logprobs else []
            top_logprobs_data = logprobs_data.top_logprobs if hasattr(logprobs_data, 'top_logprobs') and logprobs_data.top_logprobs else []
            
            # Build LogProb entries
            for i, token in enumerate(tokens):
                top_logprobs = []
                if top_logprobs_data and i < len(top_logprobs_data) and top_logprobs_data[i]:
                    for alt_token, alt_logprob in top_logprobs_data[i].items():
                        top_logprobs.append(TopLogProb(
                            token=alt_token,
                            logprob=alt_logprob
                        ))
                elif token_logprobs and i < len(token_logprobs):
                    # If no alternatives, just include the selected token
                    top_logprobs.append(TopLogProb(
                        token=token,
                        logprob=token_logprobs[i] if token_logprobs[i] is not None else 0.0
                    ))
                
                logprobs.append(LogProb(
                    token=token,
                    logprob=token_logprobs[i] if token_logprobs and i < len(token_logprobs) and token_logprobs[i] is not None else 0.0,
                    top_logprobs=top_logprobs
                ))
        
        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Calculate cost (pricing is per 1M tokens)
        pricing = TOGETHERAI_PRICING.get(self.model_name, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] / 1_000_000) + (output_tokens * pricing["output"] / 1_000_000)
        
        return Response(
            text=text,
            logprobs=logprobs,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            structured_response=structured_response,
        )
