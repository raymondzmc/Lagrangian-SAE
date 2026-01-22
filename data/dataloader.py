from typing import Any
import math
import time
import random

import einops
import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from data.config import DataConfig


def load_dataset_with_retry(
    dataset_name: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs
) -> Dataset | IterableDataset:
    """Load a HuggingFace dataset with exponential backoff retry.
    
    Args:
        dataset_name: Name of the dataset to load
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        **kwargs: Additional arguments to pass to load_dataset
        
    Returns:
        The loaded dataset
        
    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return load_dataset(dataset_name, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
                total_delay = delay + jitter
                
                print(f"Dataset loading failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"Retrying in {total_delay:.1f}s...")
                time.sleep(total_delay)
            else:
                print(f"Dataset loading failed after {max_retries + 1} attempts")
    
    raise last_exception


def create_eval_dataset_with_progress(dataset: IterableDataset, n_train_samples: int, n_eval_samples: int) -> IterableDataset:
    """Create evaluation dataset by skipping training samples with progress bar.
    
    Args:
        dataset: The original IterableDataset
        n_train_samples: Number of training samples to skip
        n_eval_samples: Number of evaluation samples to take after skipping
        
    Returns:
        IterableDataset containing only evaluation samples
    """
    print(f"Creating evaluation dataset by skipping {n_train_samples:,} training samples...")
    
    def eval_generator():
        iterator = iter(dataset)
        
        # Skip training samples with progress bar
        with tqdm(total=n_train_samples, desc="Skipping training samples", unit="samples") as pbar:
            for _ in range(n_train_samples):
                try:
                    next(iterator)
                    pbar.update(1)
                except StopIteration:
                    print("Warning: Reached end of dataset before finishing skip")
                    return
        
        # Take evaluation samples
        eval_count = 0
        for sample in iterator:
            if eval_count >= n_eval_samples:
                break
            yield sample
            eval_count += 1
    
    # Import here to avoid circular imports
    from datasets import IterableDataset as HFIterableDataset
    
    # Create new IterableDataset from our generator
    # We need to preserve the original features
    return HFIterableDataset.from_generator(
        eval_generator,
        features=dataset.features if hasattr(dataset, 'features') else None
    )


class StreamingDataLoader(DataLoader):
    """DataLoader that supports __len__ for streaming datasets with known sample count."""
    
    def __init__(self, *args, expected_length: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_length = expected_length
    
    def __len__(self) -> int:
        if self._expected_length is not None:
            return self._expected_length
        else:
            # Fall back to default behavior for non-streaming datasets
            return super().__len__()


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = False,
    num_proc: int = 10,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to
    tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of
    shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if
    parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with
    padding, then remove padding at the end.

    NOTE: Adapted from
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/utils.py#L267
    to handle IterableDataset.

    TODO: Fix typing of tokenizer

    This tokenization is useful for training language models, as it allows us to efficiently train
    on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding).
    Further, for models with absolute positional encodings, this avoids privileging early tokens
    (eg, news articles often begin with CNN, and models may learn to use early positional
    encodings to predict these)

    Args:
        dataset: The dataset to tokenize, assumed to be a HuggingFace text dataset. Can be a regular
            Dataset or an IterableDataset.
        tokenizer: The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        max_length: The length of the context window of the sequence. Defaults to 1024.
        column_name: The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token: Add BOS token at the beginning of each sequence. Defaults to False as this
            is not done during training.

    Returns:
        Dataset or IterableDataset: Returns the tokenized dataset, as a dataset of tensors, with a
        single column called "input_ids".

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it
    just outputs nothing. I'm not super sure why
    """

    # Remove all columns apart from the column_name
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)

    if tokenizer.pad_token is None:  # pyright: ignore[reportAttributeAccessIssue]
        # We add a padding token, purely to implement the tokenizer. This will be removed before
        # inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})  # pyright: ignore[reportAttributeAccessIssue]
    # Define the length to chop things up into - leaving space for a bos_token if required
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(
        examples: dict[str, list[str]],
    ) -> dict[
        str,
        NDArray[np.signedinteger[Any]],
    ]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)  # pyright: ignore[reportAttributeAccessIssue]
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        # Tokenize the chunks in parallel. Uses no because HF map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()  # type: ignore
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]  # pyright: ignore[reportAttributeAccessIssue]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)  # pyright: ignore[reportAttributeAccessIssue]
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"input_ids": tokens}

    if isinstance(dataset, IterableDataset):
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name]
        )
    else:
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name], num_proc=num_proc
        )

    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def create_dataloaders(
    data_config: DataConfig,
    global_seed: int = 0,
    buffer_size: int = 1000,
    quick_eval: bool = False,
    mini_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and eval DataLoaders with separate splits from simplified config.
    
    Args:
        data_config: The data configuration
        global_seed: Global seed for reproducibility
        buffer_size: Buffer size for streaming datasets
        quick_eval: If True, use different random seed for eval instead of skipping train samples
        mini_batch_size: If provided, use this as the batch size for training instead of train_batch_size.
                        This is used for gradient accumulation where mini_batch_size < train_batch_size.
        
    Returns:
        Tuple of (train_loader, eval_loader)
        eval_loader is None if n_eval_samples is None
    """
    # Use mini_batch_size if provided, otherwise use train_batch_size
    actual_train_batch_size = mini_batch_size if mini_batch_size is not None else data_config.train_batch_size
    # Load and prepare dataset with exponential retry for robustness
    dataset = load_dataset_with_retry(
        data_config.dataset_name,
        streaming=data_config.streaming,
        split=data_config.split
    )
    seed = data_config.seed if data_config.seed is not None else global_seed
    
    # Create train and eval splits
    if data_config.streaming:
        # For streaming datasets, handle train and eval differently based on quick_eval
        if quick_eval and data_config.n_eval_samples is not None:
            # Quick eval: use different seeds for train and eval datasets
            train_dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size).take(data_config.n_train_samples)
            
            # Use a different seed for eval dataset (seed + 42 for reproducibility)
            eval_seed = seed + 42
            eval_dataset = dataset.shuffle(seed=eval_seed, buffer_size=buffer_size).take(data_config.n_eval_samples)
        else:
            # Traditional approach: shuffle once, then split sequentially
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
            train_dataset = dataset.take(data_config.n_train_samples)
            
            if data_config.n_eval_samples is not None:
                # Skip train samples and take eval samples with progress bar
                eval_dataset = create_eval_dataset_with_progress(
                    dataset, 
                    data_config.n_train_samples, 
                    data_config.n_eval_samples
                )
            else:
                eval_dataset = None
    else:
        # For non-streaming datasets, shuffle and create proper splits
        dataset = dataset.shuffle(seed=seed)
        total_needed = data_config.n_train_samples + (data_config.n_eval_samples or 0)
        
        # Take only the samples we need to avoid loading the entire dataset
        if hasattr(dataset, '__len__') and len(dataset) > total_needed:
            dataset = dataset.select(range(total_needed))
        
        # Split into train and eval
        train_dataset = dataset.select(range(data_config.n_train_samples))
        
        if data_config.n_eval_samples is not None:
            eval_start = data_config.n_train_samples
            eval_end = eval_start + data_config.n_eval_samples
            eval_dataset = dataset.select(range(eval_start, eval_end))
        else:
            eval_dataset = None
    
    # Process datasets (tokenization if needed)
    if data_config.is_tokenized:
        train_torch_dataset = train_dataset.with_format("torch")
        # Validate tokenization
        sample = next(iter(train_torch_dataset))[data_config.column_name]
        assert isinstance(sample, torch.Tensor) and sample.ndim == 1, "Expected tokenized dataset"
        assert len(sample) == data_config.context_length, f"Expected length {data_config.context_length}, got {len(sample)}"
        
        eval_torch_dataset = eval_dataset.with_format("torch") if eval_dataset is not None else None
    else:
        tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
        train_torch_dataset = tokenize_and_concatenate(
            train_dataset,
            tokenizer,
            max_length=data_config.context_length,
            column_name=data_config.column_name,
            add_bos_token=True,
        )
        
        eval_torch_dataset = None
        if eval_dataset is not None:
            eval_torch_dataset = tokenize_and_concatenate(
                eval_dataset,
                tokenizer,
                max_length=data_config.context_length,
                column_name=data_config.column_name,
                add_bos_token=True,
            )
    
    # Use StreamingDataLoader for streaming datasets, regular DataLoader for others
    if data_config.streaming:
        # Calculate expected number of batches for streaming datasets
        expected_train_batches = math.ceil(data_config.n_train_samples / actual_train_batch_size)
        train_loader = StreamingDataLoader(
            train_torch_dataset,
            batch_size=actual_train_batch_size,
            shuffle=False,  # Already shuffled the base dataset
            expected_length=expected_train_batches,
        )
        
        eval_loader = None
        if eval_torch_dataset is not None and data_config.n_eval_samples is not None:
            eval_bs = data_config.effective_eval_batch_size
            expected_eval_batches = math.ceil(data_config.n_eval_samples / eval_bs)
            eval_loader = StreamingDataLoader(
                eval_torch_dataset,
                batch_size=eval_bs,
                shuffle=False,
                expected_length=expected_eval_batches,
            )
    else:
        # Use regular DataLoader for non-streaming datasets
        train_loader = DataLoader(
            train_torch_dataset,
            batch_size=actual_train_batch_size,
            shuffle=False,  # Already shuffled the base dataset
        )

        eval_loader = None
        if eval_torch_dataset is not None:
            eval_loader = DataLoader(
                eval_torch_dataset,
                batch_size=data_config.effective_eval_batch_size,
                shuffle=False,
            )
    
    return train_loader, eval_loader
