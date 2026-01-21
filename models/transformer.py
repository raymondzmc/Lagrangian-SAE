import torch
import tqdm
import yaml
import wandb
import inspect
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict
from wandb.apis.public import Run
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import LocallyOverridenDefaults, sample_logits

from models.saes import (
    SAEConfig,
    SAELoss,
    SAEOutput,
    create_sae_config,
    AllSAEConfigs,
    SAE_TYPE_TO_CLS,
)
from models.loader import load_tlens_model
from utils.constants import CONFIG_FILE, WANDB_CACHE_DIR
from utils.models import get_hook_shapes


class SAETransformerOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    activations: dict[str, Float[torch.Tensor, "batch pos dim"]]
    sae_outputs: dict[str, SAEOutput]
    loss_outputs: dict[str, SAELoss]


class SAETransformer(torch.nn.Module):
    """A transformer model with SAEs at various positions.

    Args:
        tlens_model: TransformerLens model.
        sae_config: Configurations for the SAEs.
    """

    def __init__(self, tlens_model: HookedTransformer, sae_config: SAEConfig, device: torch.device | None = None):
        super().__init__()
        # Ensure the model is moved to the correct device before using it
        self.tlens_model = tlens_model.eval()
        self.sae_config: AllSAEConfigs = sae_config
        self.device = device
        if device is not None:
            self.tlens_model = self.tlens_model.to(device)

        self.raw_sae_positions = sae_config.sae_positions
        self.hook_shapes: dict[str, list[int]] = get_hook_shapes(
            self.tlens_model, self.raw_sae_positions
        )
        # ModuleDict keys can't have periods in them, so we replace them with hyphens
        self.all_sae_positions = [name.replace(".", "-") for name in self.raw_sae_positions]
        self.saes = torch.nn.ModuleDict()
        self.__create_sae_modules()
    
    def __create_sae_modules(self):
        """Create SAE modules with proper device context."""
        sae_config = self.sae_config
        device = self.device or self.tlens_model.cfg.device
        # Get dtype from the model to ensure SAEs use the same precision
        dtype = self.tlens_model.cfg.dtype
        
        # Get the SAE class from the mapping
        sae_cls = SAE_TYPE_TO_CLS.get(sae_config.sae_type)
        if sae_cls is None:
            raise ValueError(f"Unsupported SAE type: {sae_config.sae_type}")
        
        for i in range(len(self.all_sae_positions)):
            input_size = self.hook_shapes[self.raw_sae_positions[i]][-1]
            
            # Get all config parameters as a dict
            config_dict = sae_config.model_dump()
            
            # Add the required parameters first
            config_dict['input_size'] = input_size
            # Use explicit n_dict_components if set, otherwise compute from ratio
            if sae_config.n_dict_components is not None:
                config_dict['n_dict_components'] = sae_config.n_dict_components
            elif sae_config.dict_size_to_input_ratio is not None:
                config_dict['n_dict_components'] = int(sae_config.dict_size_to_input_ratio * input_size)
            else:
                raise ValueError("Either n_dict_components or dict_size_to_input_ratio must be set")
            
            # Get the constructor signature to filter parameters
            constructor_sig = inspect.signature(sae_cls.__init__)
            constructor_params = set(constructor_sig.parameters.keys()) - {'self'}  # Exclude 'self'
            
            # Filter config_dict to only include parameters that the constructor accepts
            filtered_config_dict = {
                key: value for key, value in config_dict.items() 
                if key in constructor_params
            }
            
            # Create SAE and move to device with same dtype as model
            sae = sae_cls(**filtered_config_dict)
            if dtype is not None:
                sae = sae.to(device=device, dtype=dtype)
            else:
                sae = sae.to(device=device)
            self.saes[self.all_sae_positions[i]] = sae

    def forward(
        self,
        tokens: Int[torch.Tensor, "batch pos"],
        sae_positions: list[str],
        cache_positions: list[str] | None = None,
        modify_output: bool = False,
        stop_at_layer: int | None = None,
        compute_loss: bool = True,
    ) -> SAETransformerOutput:
        """Unified forward pass with clear behavior for modify_output.
        TODO: Add support for end-to-end training.
        
        Args:
            tokens: The input tokens.
            sae_positions: The positions where SAEs should be applied.
            cache_positions: Additional positions to cache activations at.
            modify_output: 
                - False: Return original logits/activations. Later layers use unmodified representations.
                - True: Return modified logits/activations. Later layers use SAE-modified representations.
            stop_at_layer: Layer to stop at.

        Returns:
            SAETransformerOutput containing logits, activations, and sae_outputs
        """
        all_hook_names = list(set(sae_positions + (cache_positions or [])))
        activation_cache: dict[str, torch.Tensor] = {}
        sae_outputs: dict[str, SAEOutput] = {}
        loss_outputs: dict[str, SAELoss] = {}
        
        def activation_cache_hook(position: str):
            """Single hook that handles both SAE processing and caching."""
            def hook_fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
                activation_cache[position] = x.detach().clone()

                if position in sae_positions:
                    sae = self.saes[position.replace(".", "-")]
                    
                    # Auto-correct device mismatches
                    input_device = x.device
                    sae_device = next(sae.parameters()).device
                    if input_device != sae_device:
                        # Silently move SAE to input device
                        sae.to(input_device)
                    
                    sae_output = sae(x.detach().clone())
                    sae_outputs[position] = sae_output
                    if compute_loss:
                        loss_output = sae.compute_loss(sae_output)
                        loss_outputs[position] = loss_output

                    if modify_output:
                        # Use SAE reconstructed activations
                        activation_cache[position] = sae_output.output.detach().clone()
                        return sae_output.output
                return x
            return hook_fn

        logits = self.tlens_model.run_with_hooks(
            tokens,
            fwd_hooks=[(pos, activation_cache_hook(pos)) for pos in all_hook_names],
            stop_at_layer=stop_at_layer,
        )
        assert isinstance(logits, torch.Tensor)

        return SAETransformerOutput(
            logits=logits,
            activations=activation_cache,
            sae_outputs=sae_outputs,
            loss_outputs=loss_outputs,
        )

    def to(self, *args: Any, **kwargs: Any) -> "SAETransformer":
        """Move the model to the specified device/dtype.
        """
        # Move base class first
        super().to(*args, **kwargs)
        self.saes.to(*args, **kwargs)
        
        # Determine target device
        target_device = None
        if args and len(args) >= 1:
            target_device = args[0]
        elif 'device' in kwargs:
            target_device = kwargs['device']
        
        # Force move tlens_model and SAEs to exact device
        if target_device is not None:
            # Use assignment to ensure the reference is updated
            self.tlens_model = self.tlens_model.to(target_device)
            self.saes = self.saes.to(target_device)
        else:
            # Fallback for dtype-only moves
            self.tlens_model.to(*args, **kwargs)
            self.saes.to(*args, **kwargs)
        return self

    @torch.inference_mode()
    def generate(
        self,
        input: str | Float[torch.Tensor, "batch pos"] = "",
        sae_positions: list[str] | None | Literal["all"] = "all",
        use_sae_modified: bool = True,
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: int | None = None,
        do_sample: bool = True,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        prepend_bos: bool | None = None,
        padding_side: Literal["left", "right"] | None = None,
        return_type: str | None = "input",
        verbose: bool = True,
    ) -> Int[torch.Tensor, "batch pos_plus_new_tokens"] | str:
        """Sample Tokens from the model.

        Adapted from transformer_lens.HookedTransformer.generate()

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish
        (by producing an EOT token), we keep running the model on the entire batch, but throw away
        the output for a finished sequence and just keep adding EOTs to pad.

        This supports entering a single string, but not a list of strings - if the strings don't
        tokenize to exactly the same length, this gets messy. If that functionality is needed,
        convert them to a batch of tokens and input that instead.

        Args:
            input (Union[str, Int[torch.Tensor, "batch pos"])]): Either a batch of tokens ([batch,
                pos]) or a text string (this will be converted to a batch of tokens with batch size
                1).
            sae_positions (list[str] | None | Literal["all"]): The positions where SAEs should be applied.
                If None, no SAEs are used. If "all", all configured SAE positions are used.
            use_sae_modified (bool): Whether to use SAE-modified representations for generation.
                If True (default), later layers see SAE reconstructions. If False, later layers see
                original activations (SAEs are still applied for analysis but don't affect generation).
            max_new_tokens (int): Maximum number of tokens to generate.
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token.
            eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end
                of sentence. If None, use the tokenizer's eos_token_id - required if using
                stop_at_eos. It's also possible to provide a list of token IDs (not just the
                eos_token_id), in which case the generation will stop when any of them are output
                (useful e.g. for stable_lm).
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use
                greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens.
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0,
                we take the top tokens with cumulative probability >= top_p.
            temperature (float): Temperature for sampling. Higher values will make the model more
                random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is
                sampling from a uniform distribution).
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous
                tokens. Higher values will make the model more random.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (applicable when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos (default is True unless specified
                otherwise). Pass True or False to override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
            return_type (Optional[str]): The type of the output to return - either a string (str),
                a tensor of tokens (tensor) or whatever the format of the input was (input).
            verbose (bool): If True, show tqdm progress bars for generation.

        Returns:
            generated sequence of new tokens, or completed prompt string (by default returns same
                type as input).
        """

        with LocallyOverridenDefaults(
            self.tlens_model, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if isinstance(input, str):
                # If text, convert to tokens (batch_size=1)
                assert (
                    self.tlens_model.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                tokens = self.tlens_model.to_tokens(
                    input, prepend_bos=prepend_bos, padding_side=padding_side
                )
            else:
                tokens = input

            if return_type == "input":
                return_type = "str" if isinstance(input, str) else "tensor"

            assert isinstance(tokens, torch.Tensor)
            batch_size = tokens.shape[0]
            # Use the device the model is currently on instead of defaulting to cuda:0
            device = next(self.tlens_model.parameters()).device
            tokens = tokens.to(device)

            stop_tokens = []
            eos_token_for_padding = 0
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tlens_model.tokenizer is not None
                    and self.tlens_model.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert tokenizer_has_eos_token, (
                        "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or "
                        "has no eos_token_id"
                    )
                    assert self.tlens_model.tokenizer is not None
                    eos_token_id = self.tlens_model.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    assert eos_token_id is not None
                    stop_tokens = eos_token_id
                    eos_token_for_padding = eos_token_id[0]

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(
                batch_size, dtype=torch.bool, device=self.tlens_model.cfg.device
            )

            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.eval()
            for _ in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
                # While generating, we keep generating logits, throw away all but the final logits,
                # and then use those logits to sample from the distribution We keep adding the
                # sampled tokens to the end of tokens.
                # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                # the cache.
                if sae_positions is None:
                    # No SAEs - use empty list for sae_positions
                    result = self.forward(tokens, sae_positions=[], modify_output=False)
                else:
                    if sae_positions == "all":
                        sae_positions = self.raw_sae_positions
                    result = self.forward(tokens, sae_positions=sae_positions, modify_output=use_sae_modified)
                
                logits = result.logits
                assert logits is not None
                final_logits = logits[:, -1, :]

                if do_sample:
                    sampled_tokens = sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(device)
                else:
                    sampled_tokens = final_logits.argmax(-1).to(device)

                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    if isinstance(eos_token_for_padding, int):
                        sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(sampled_tokens, torch.tensor(stop_tokens).to(device))
                    )

                tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

                if stop_at_eos and finished_sequences.all():
                    break

            if return_type == "str":
                assert self.tlens_model.tokenizer is not None
                if self.tlens_model.cfg.default_prepend_bos:
                    # If we prepended a BOS token, remove it when returning output.
                    return self.tlens_model.tokenizer.decode(tokens[0, 1:])
                else:
                    return self.tlens_model.tokenizer.decode(tokens[0])

            else:
                return tokens

    @classmethod
    def from_wandb(cls, wandb_project_run_id: str) -> "SAETransformer":
        """Instantiate an SAETransformer using the latest checkpoint from a wandb run.

        Args:
            wandb_project_run_id: The wandb project name and run ID separated by a forward slash.
                E.g. "gpt2/2lzle2f0"

        Returns:
            An instance of the SAETransformer class loaded from the specified wandb run.
        """
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        model_cache_dir = Path(WANDB_CACHE_DIR) / wandb_project_run_id

        train_config_files_remote = [file for file in run.files() if file.name.endswith(CONFIG_FILE)]
        assert len(train_config_files_remote) > 0, f"Cannot find config file for wandb run {wandb_project_run_id}."
        train_config_file_remote = train_config_files_remote[0]

        train_config_file = train_config_file_remote.download(
            exist_ok=True, replace=True, root=model_cache_dir
        ).name

        checkpoints = [file for file in run.files() if file.name.endswith(".pt")]
        assert len(checkpoints) > 0, f"Cannot find any checkpoints for wandb run {wandb_project_run_id}."
        latest_checkpoint_remote = sorted(
            checkpoints, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1])
        )[-1]
        latest_checkpoint_file = latest_checkpoint_remote.download(
            exist_ok=True, replace=True, root=model_cache_dir
        ).name
        assert latest_checkpoint_file is not None, "Failed to download the latest checkpoint."
        return cls.from_local_path(
            checkpoint_file=latest_checkpoint_file, config_file=train_config_file
        )

    @classmethod
    def from_local_path(
        cls,
        checkpoint_dir: str | Path | None = None,
        checkpoint_file: str | Path | None = None,
        config_file: str | Path | None = None,
    ) -> "SAETransformer":
        """Instantiate an SAETransformer using a checkpoint from a specified directory.

        NOTE: the current implementation restricts us from using the
        e2e_sae/scripts/train_tlens_saes/run_train_tlens_saes.py.Config class for type
        validation due to circular imports. Would need to move the Config class to a separate file
        to use it here.

        Args:
            checkpoint_dir: The directory containing one or more checkpoint files and
                `CONFIG_FILE`. If multiple checkpoints are present, load the one with the
                highest n_samples number (i.e. the latest checkpoint).
            checkpoint_file: The specific checkpoint file to load. If specified, `checkpoint_dir`
                is ignored and config_file must also be specified.
            config_file: The config file to load. If specified, `checkpoint_dir` is ignored and
                checkpoint_file must also be specified.

        Returns:
            An instance of the SAETransformer class loaded from the specified checkpoint.
        """
        if checkpoint_file is not None:
            checkpoint_file = Path(checkpoint_file)
            assert config_file is not None
            config_file = Path(config_file)
        else:
            assert checkpoint_dir is not None
            checkpoint_dir = Path(checkpoint_dir)
            assert config_file is None
            config_file = checkpoint_dir / CONFIG_FILE

            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            checkpoint_file = sorted(
                checkpoint_files, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1])
            )[-1]

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        tlens_model = load_tlens_model(
            tlens_model_name=config["tlens_model_name"]['value'],
            tlens_model_path=config["tlens_model_path"]['value'],
            device="cpu"  # Load to CPU first, then to target device in SAETransformer.__init__
        )
        
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        sae_config_dict = config["saes"]["value"].copy()
        model = cls(tlens_model=tlens_model, sae_config=create_sae_config(sae_config_dict))
        model.saes.load_state_dict(checkpoint)
        return model
