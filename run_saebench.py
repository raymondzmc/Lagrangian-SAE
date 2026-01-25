#!/usr/bin/env python3
"""
Run SAEBench evaluations on SAEs trained with this codebase.

This script:
1. Loads a trained SAE from a wandb run
2. Wraps it for SAEBench compatibility
3. Runs selected SAEBench evaluations
4. Uploads results as artifacts to the same wandb run

Usage:
    python run_saebench.py \
        --wandb_run "entity/project/run_id" \
        --eval_types core sparse_probing scr tpp \
        --device cuda:0 \
        --output_path ./saebench_results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path for local imports
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

import torch
import wandb
from tqdm import tqdm

from models.transformer import SAETransformer
from config import Config
from utils.io import load_config
from sae_bench_wrapper import create_saebench_saes_from_transformer
from settings import settings

# SAEBench imports
import sae_bench.evals.core.main as core
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.sae_bench_utils.general_utils as general_utils

# Optional imports for additional evaluations
try:
    import sae_bench.evals.absorption.main as absorption
    HAS_ABSORPTION = True
except ImportError:
    HAS_ABSORPTION = False

try:
    import sae_bench.evals.autointerp.main as autointerp
    HAS_AUTOINTERP = True
except ImportError:
    HAS_AUTOINTERP = False

try:
    import sae_bench.evals.ravel.main as ravel
    HAS_RAVEL = True
except ImportError:
    HAS_RAVEL = False

try:
    import sae_bench.evals.sparse_probing_sae_probes.main as sparse_probing_sae_probes
    HAS_SPARSE_PROBING_SAE_PROBES = True
except ImportError:
    HAS_SPARSE_PROBING_SAE_PROBES = False


# Default configurations for different models
MODEL_CONFIGS = {
    "gpt2": {
        "batch_size": 256,
        "dtype": "float32",
        "d_model": 768,
    },
    "gpt2-small": {
        "batch_size": 256,
        "dtype": "float32",
        "d_model": 768,
    },
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "d_model": 512,
    },
    "pythia-160m-deduped": {
        "batch_size": 256,
        "dtype": "float32",
        "d_model": 768,
    },
    "pythia-410m-deduped": {
        "batch_size": 128,
        "dtype": "float32",
        "d_model": 1024,
    },
    "gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "d_model": 2304,
    },
    "TinyStories-1M": {
        "batch_size": 256,
        "dtype": "float32",
        "d_model": 64,
    },
}

# Output folder mapping for each evaluation type
OUTPUT_FOLDERS = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "sparse_probing_sae_probes": "eval_results/sparse_probing_sae_probes",
    "ravel": "eval_results/ravel",
}


def get_model_config(model_name: str) -> dict:
    """Get default configuration for a model, with fallback for unknown models."""
    # Try exact match first
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Try partial match (e.g., "roneneldan/TinyStories-1M" -> "TinyStories-1M")
    for key in MODEL_CONFIGS:
        if key in model_name:
            return MODEL_CONFIGS[key]
    
    # Fallback for unknown models
    print(f"Warning: Unknown model '{model_name}', using default config")
    return {
        "batch_size": 128,
        "dtype": "float32",
        "d_model": 768,
    }


def find_run_by_name(project: str, run_name: str, entity: str | None = None) -> str:
    """Find wandb run ID by run name.
    
    Args:
        project: Wandb project name
        run_name: Display name of the run
        entity: Wandb entity (uses default from settings if None)
    
    Returns:
        Full run path: entity/project/run_id
    """
    api = wandb.Api()
    if entity is None:
        entity = settings.wandb_entity
    
    if entity is None:
        raise ValueError("Wandb entity not specified. Set WANDB_ENTITY in .env or pass --entity")
    
    # Query runs by name
    print(f"Searching for run '{run_name}' in project '{entity}/{project}'...")
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    runs_list = list(runs)
    
    if len(runs_list) == 0:
        raise ValueError(f"No run found with name '{run_name}' in project '{entity}/{project}'")
    if len(runs_list) > 1:
        print(f"Warning: Found {len(runs_list)} runs with name '{run_name}', using most recent")
    
    run = runs_list[0]
    full_path = f"{entity}/{project}/{run.id}"
    print(f"Found run: {full_path}")
    return full_path


def load_sae_from_wandb(wandb_run: str, device: torch.device) -> tuple[SAETransformer, Config, str]:
    """
    Load an SAE from a wandb run.
    
    Args:
        wandb_run: The wandb run path (e.g., "entity/project/run_id")
        device: The device to load the model on
        
    Returns:
        Tuple of (SAETransformer model, Config, run_id)
    """
    print(f"Loading SAE from wandb run: {wandb_run}")
    
    # Parse wandb path
    parts = wandb_run.split("/")
    if len(parts) == 3:
        entity, project, run_id = parts
        project_run = f"{entity}/{project}/{run_id}"
    elif len(parts) == 2:
        # Assume entity is default
        project, run_id = parts
        project_run = f"{project}/{run_id}"
    else:
        raise ValueError(f"Invalid wandb_run format: {wandb_run}. Expected 'entity/project/run_id' or 'project/run_id'")
    
    # Load the model from wandb
    sae_transformer = SAETransformer.from_wandb(project_run).to(device)
    sae_transformer.saes.eval()
    
    # Get run config
    api = wandb.Api()
    run = api.run(project_run)
    run_config = run.config
    
    # Parse config (ignore extra fields from older configs)
    config = load_config(run_config, Config, ignore_extra=True)
    
    return sae_transformer, config, run_id


def run_evaluations(
    selected_saes: list[tuple[str, torch.nn.Module]],
    model_name: str,
    eval_types: list[str],
    device: str,
    output_path: str,
    llm_batch_size: int | None = None,
    llm_dtype: str | None = None,
    api_key: str | None = None,
    force_rerun: bool = False,
    save_activations: bool = True,
    random_seed: int = 42,
) -> dict[str, dict]:
    """
    Run SAEBench evaluations on the provided SAEs.
    
    Args:
        selected_saes: List of (sae_id, sae) tuples
        model_name: The TransformerLens model name
        eval_types: List of evaluation types to run
        device: Device string (e.g., "cuda:0")
        output_path: Base path for saving results
        llm_batch_size: Batch size for LLM forward passes (auto-detected if None)
        llm_dtype: Data type string (auto-detected if None)
        api_key: OpenAI API key for autointerp evaluation
        force_rerun: Whether to force re-running evaluations
        save_activations: Whether to save activations for reuse (can use ~10-50GB disk per eval)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping eval_type -> results
    """
    # Get model config
    model_config = get_model_config(model_name)
    
    if llm_batch_size is None:
        llm_batch_size = model_config["batch_size"]
    if llm_dtype is None:
        llm_dtype = model_config["dtype"]
    
    print(f"\nRunning evaluations for model: {model_name}")
    print(f"  Batch size: {llm_batch_size}")
    print(f"  Dtype: {llm_dtype}")
    print(f"  Device: {device}")
    print(f"  Eval types: {eval_types}")
    print(f"  Number of SAEs: {len(selected_saes)}")
    
    all_results = {}
    
    # Define evaluation runners
    def run_core():
        print("\n" + "="*60)
        print("Running CORE evaluation (L0 / Loss Recovered)")
        print("="*60)
        output_folder = os.path.join(output_path, "core")
        os.makedirs(output_folder, exist_ok=True)
        
        return core.multiple_evals(
            selected_saes=selected_saes,
            n_eval_reconstruction_batches=200,
            n_eval_sparsity_variance_batches=2000,
            eval_batch_size_prompts=16,
            compute_featurewise_density_statistics=True,
            compute_featurewise_weight_based_metrics=True,
            exclude_special_tokens_from_reconstruction=True,
            dataset="Skylion007/openwebtext",
            context_size=128,
            output_folder=output_folder,
            verbose=True,
            dtype=llm_dtype,
            device=device,
        )
    
    def run_sparse_probing():
        print("\n" + "="*60)
        print("Running SPARSE PROBING evaluation")
        print("="*60)
        output_folder = os.path.join(output_path, "sparse_probing")
        os.makedirs(output_folder, exist_ok=True)
        
        # Enable lower_vram_usage for large models (>1B params)
        is_large_model = any(x in model_name.lower() for x in ["gemma", "llama", "mistral", "2b", "7b", "13b"])
        
        config = sparse_probing.SparseProbingEvalConfig(
            model_name=model_name,
            random_seed=random_seed,
            llm_batch_size=llm_batch_size,
            llm_dtype=llm_dtype,
            lower_vram_usage=is_large_model,
            sae_batch_size=8,  # Small batch to avoid OOM with large SAEs
        )
        
        return sparse_probing.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=output_folder,
            force_rerun=force_rerun,
            clean_up_activations=not save_activations,
            save_activations=save_activations,
        )
    
    def run_scr():
        print("\n" + "="*60)
        print("Running SCR (Spurious Correlation Removal) evaluation")
        print("="*60)
        output_folder = output_path
        os.makedirs(os.path.join(output_folder, "scr"), exist_ok=True)
        
        # Enable lower_vram_usage for large models (>1B params)
        is_large_model = any(x in model_name.lower() for x in ["gemma", "llama", "mistral", "2b", "7b", "13b"])
        
        # Use smaller batch sizes for large SAEs (65K+) to avoid OOM
        scr_llm_batch_size = max(8, llm_batch_size // 2) if is_large_model else llm_batch_size
        scr_sae_batch_size = 2 if is_large_model else 8  # Reduced from 8 to 2 for 65K SAEs
        
        config = scr_and_tpp.ScrAndTppEvalConfig(
            model_name=model_name,
            random_seed=random_seed,
            perform_scr=True,
            llm_batch_size=scr_llm_batch_size,
            llm_dtype=llm_dtype,
            lower_vram_usage=is_large_model,
            sae_batch_size=scr_sae_batch_size,
        )
        
        return scr_and_tpp.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=output_folder,
            force_rerun=force_rerun,
            clean_up_activations=not save_activations,
            save_activations=save_activations,
        )
    
    def run_tpp():
        print("\n" + "="*60)
        print("Running TPP (Targeted Probe Perturbation) evaluation")
        print("="*60)
        output_folder = output_path
        os.makedirs(os.path.join(output_folder, "tpp"), exist_ok=True)
        
        # Enable lower_vram_usage for large models (>1B params)
        is_large_model = any(x in model_name.lower() for x in ["gemma", "llama", "mistral", "2b", "7b", "13b"])
        
        # Use smaller batch sizes for large SAEs (65K+) to avoid OOM
        tpp_llm_batch_size = max(8, llm_batch_size // 2) if is_large_model else llm_batch_size
        tpp_sae_batch_size = 2 if is_large_model else 8  # Reduced from 8 to 2 for 65K SAEs
        
        config = scr_and_tpp.ScrAndTppEvalConfig(
            model_name=model_name,
            random_seed=random_seed,
            perform_scr=False,
            llm_batch_size=tpp_llm_batch_size,
            llm_dtype=llm_dtype,
            lower_vram_usage=is_large_model,
            sae_batch_size=tpp_sae_batch_size,
        )
        
        return scr_and_tpp.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=output_folder,
            force_rerun=force_rerun,
            clean_up_activations=not save_activations,
            save_activations=save_activations,
        )
    
    def run_absorption():
        if not HAS_ABSORPTION:
            print("Warning: Absorption evaluation not available")
            return None
        
        print("\n" + "="*60)
        print("Running ABSORPTION evaluation")
        print("="*60)
        output_folder = os.path.join(output_path, "absorption")
        os.makedirs(output_folder, exist_ok=True)
        
        config = absorption.AbsorptionEvalConfig(
            model_name=model_name,
            random_seed=random_seed,
            llm_batch_size=llm_batch_size,
            llm_dtype=llm_dtype,
        )
        
        return absorption.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=output_folder,
            force_rerun=force_rerun,
        )
    
    def run_autointerp():
        if not HAS_AUTOINTERP:
            print("Warning: AutoInterp evaluation not available")
            return None
        
        if api_key is None:
            print("Warning: Skipping AutoInterp - no API key provided")
            return None
        
        print("\n" + "="*60)
        print("Running AUTOINTERP evaluation")
        print("="*60)
        output_folder = os.path.join(output_path, "autointerp")
        os.makedirs(output_folder, exist_ok=True)
        
        # Reduce batch size for large SAEs to avoid OOM, but keep 2M tokens
        is_large_model = any(x in model_name.lower() for x in ["gemma", "llama", "mistral", "2b", "7b", "13b"])
        autointerp_batch_size = max(4, llm_batch_size // 4) if is_large_model else llm_batch_size
        
        config = autointerp.AutoInterpEvalConfig(
            model_name=model_name,
            random_seed=random_seed,
            llm_batch_size=autointerp_batch_size,
            llm_dtype=llm_dtype,
            total_tokens=2_000_000,  # Keep at 2M tokens
            n_latents=500,  # Reduce from 1000 to 500 for faster eval with large SAEs
        )
        
        return autointerp.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            api_key=api_key,
            output_path=output_folder,
            force_rerun=force_rerun,
        )
    
    def run_ravel():
        if not HAS_RAVEL:
            print("Warning: RAVEL evaluation not available")
            return None
        
        print("\n" + "="*60)
        print("Running RAVEL (Disentanglement) evaluation")
        print("="*60)
        output_folder = os.path.join(output_path, "ravel")
        os.makedirs(output_folder, exist_ok=True)
        
        # RAVEL expects short model names (e.g., "gemma-2-2b" not "google/gemma-2-2b")
        # Convert full HuggingFace path to short name
        ravel_model_name = model_name
        if "/" in model_name:
            ravel_model_name = model_name.split("/")[-1]
        
        # Use very small batch for large SAEs to avoid OOM
        is_large_model = any(x in model_name.lower() for x in ["gemma", "llama", "mistral", "2b", "7b", "13b"])
        ravel_batch_size = 4 if is_large_model else max(1, llm_batch_size // 4)
        
        config = ravel.RAVELEvalConfig(
            model_name=ravel_model_name,
            random_seed=random_seed,
            llm_batch_size=ravel_batch_size,
            llm_dtype=llm_dtype,
        )
        
        return ravel.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=output_folder,
            force_rerun=force_rerun,
        )
    
    def run_sparse_probing_sae_probes():
        if not HAS_SPARSE_PROBING_SAE_PROBES:
            print("Warning: sparse_probing_sae_probes evaluation not available")
            return None
        
        print("\n" + "="*60)
        print("Running SPARSE PROBING (SAE Probes) evaluation - 140+ datasets")
        print("="*60)
        output_folder = os.path.join(output_path, "sparse_probing_sae_probes")
        os.makedirs(output_folder, exist_ok=True)
        
        # Get the hook name from the first SAE
        hook_name = selected_saes[0][1].cfg.hook_name if selected_saes else None
        
        config = sparse_probing_sae_probes.SparseProbingSaeProbesEvalConfig(
            model_name=model_name,
            ks=[1, 2, 5, 10],  # k values for sparse probing
            include_llm_baseline=True,  # Compare against LLM residual stream baseline
            results_path=os.path.join(output_path, "artifacts/sparse_probing_sae_probes"),
            model_cache_path=os.path.join(output_path, "artifacts/sparse_probing_sae_probes--model_acts_cache"),
        )
        
        return sparse_probing_sae_probes.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=output_folder,
            force_rerun=force_rerun,
        )
    
    # Mapping of eval types to runner functions
    eval_runners = {
        "core": run_core,
        "sparse_probing": run_sparse_probing,
        "sparse_probing_sae_probes": run_sparse_probing_sae_probes,
        "scr": run_scr,
        "tpp": run_tpp,
        "absorption": run_absorption,
        "autointerp": run_autointerp,
        "ravel": run_ravel,
    }
    
    # Run selected evaluations
    for eval_type in tqdm(eval_types, desc="Running evaluations"):
        if eval_type not in eval_runners:
            print(f"Warning: Unknown evaluation type '{eval_type}', skipping")
            continue
        
        try:
            result = eval_runners[eval_type]()
            if result is not None:
                all_results[eval_type] = result
        except Exception as e:
            print(f"Error running {eval_type} evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results


def upload_results_to_wandb(
    results: dict[str, dict],
    wandb_run: str,
    output_path: str,
    run_id: str,
) -> None:
    """
    Upload evaluation results to the original wandb run as an artifact.
    
    Args:
        results: Dictionary of evaluation results
        wandb_run: The wandb run path
        output_path: Path where results are saved
        run_id: The run ID
    """
    print("\n" + "="*60)
    print("Uploading results to Wandb")
    print("="*60)
    
    # Parse wandb path
    parts = wandb_run.split("/")
    if len(parts) == 3:
        entity, project, _ = parts
    else:
        project = parts[0]
        entity = None
    
    # Initialize wandb run (resume the original run)
    wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="allow",
        reinit=True,
    )
    
    # Create artifact
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"saebench_results_{timestamp}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="evaluation_results",
        description="SAEBench evaluation results",
    )
    
    # Add result files to artifact
    for eval_type, result in results.items():
        # Save result as JSON
        result_file = os.path.join(output_path, f"{eval_type}_results.json")
        
        # Handle different result types
        if isinstance(result, dict):
            # Convert tensors to lists for JSON serialization
            serializable_result = {}
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    serializable_result[k] = v.tolist()
                elif hasattr(v, '__dict__'):
                    serializable_result[k] = str(v)
                else:
                    serializable_result[k] = v
            
            with open(result_file, 'w') as f:
                json.dump(serializable_result, f, indent=2, default=str)
        else:
            with open(result_file, 'w') as f:
                json.dump({"result": str(result)}, f, indent=2)
        
        artifact.add_file(result_file)
    
    # Also add any result files from eval_results folder
    eval_results_path = Path(output_path)
    for eval_folder in eval_results_path.glob("*"):
        if eval_folder.is_dir():
            for result_file in eval_folder.glob("*.json"):
                artifact.add_file(str(result_file), name=f"{eval_folder.name}/{result_file.name}")
    
    # Log artifact
    wandb.log_artifact(artifact)
    
    # Also log summary metrics to the run
    summary_metrics = {}
    for eval_type, result in results.items():
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    summary_metrics[f"saebench/{eval_type}/{k}"] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    summary_metrics[f"saebench/{eval_type}/{k}"] = v.item()
    
    if summary_metrics:
        wandb.log(summary_metrics)
    
    wandb.finish()
    print(f"Results uploaded as artifact: {artifact_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SAEBench evaluations on SAEs trained with this codebase"
    )
    
    # Run identification - either wandb_run OR (project + run_name)
    parser.add_argument(
        "--wandb_run",
        type=str,
        default=None,
        help="Wandb run path: 'entity/project/run_id' or 'project/run_id'"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Wandb project name (required if using --run_name)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Wandb run display name (alternative to --wandb_run)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Wandb entity/username (uses WANDB_ENTITY from .env if not specified)"
    )
    
    # Evaluation selection
    parser.add_argument(
        "--eval_types",
        type=str,
        nargs="+",
        default=["core", "sparse_probing"],
        choices=["core", "sparse_probing", "sparse_probing_sae_probes", "scr", "tpp", "absorption", "autointerp", "ravel"],
        help="Evaluation types to run (default: core sparse_probing)"
    )
    
    # Device and output
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda:0 if available)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./saebench_results",
        help="Path for saving results (default: ./saebench_results)"
    )
    
    # Model parameters
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM forward passes (auto-detected if not specified)"
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Data type for LLM (auto-detected if not specified)"
    )
    
    # API key for autointerp
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key for autointerp evaluation"
    )
    parser.add_argument(
        "--api_key_file",
        type=str,
        default="openai_api_key.txt",
        help="File containing OpenAI API key (default: openai_api_key.txt)"
    )
    
    # Other options
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force re-running evaluations even if results exist"
    )
    parser.add_argument(
        "--save_activations",
        action="store_true",
        default=True,
        help="Save activations for reuse across evaluations (default: True, uses ~10-50GB disk per eval)"
    )
    parser.add_argument(
        "--no_save_activations",
        action="store_true",
        help="Do not save activations (clean up after each eval to save disk space)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Skip uploading results to wandb"
    )
    parser.add_argument(
        "--training_tokens",
        type=int,
        default=-1,
        help="Number of training tokens (for plotting metadata, -1 if unknown)"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup wandb
    if settings.wandb_api_key:
        wandb.login(key=settings.wandb_api_key)
    else:
        wandb.login()
    
    # Load API key for autointerp if needed
    api_key = args.api_key
    if api_key is None and "autointerp" in args.eval_types:
        # First try to load from settings.py (environment variable OPENAI_API_KEY)
        if settings.openai_api_key:
            api_key = settings.openai_api_key
            print("Loaded OpenAI API key from settings (OPENAI_API_KEY environment variable)")
        # Fall back to file if settings doesn't have it
        elif os.path.exists(args.api_key_file):
            with open(args.api_key_file) as f:
                api_key = f.read().strip()
            print(f"Loaded OpenAI API key from file: {args.api_key_file}")
    
    # Determine wandb run path - either from --wandb_run or --run_name
    if args.run_name:
        if not args.project:
            raise ValueError("--project is required when using --run_name")
        entity = args.entity  # Will use settings.wandb_entity if None
        wandb_run = find_run_by_name(args.project, args.run_name, entity)
    elif args.wandb_run:
        wandb_run = args.wandb_run
    else:
        raise ValueError("Either --wandb_run or (--project and --run_name) must be specified")
    
    # Load SAE from wandb
    sae_transformer, config, run_id = load_sae_from_wandb(wandb_run, device)
    
    # Get model name - handle different config formats
    model_name = config.tlens_model_name
    if model_name is None and config.tlens_model_path is not None:
        # Extract model name from path if needed
        model_name = str(config.tlens_model_path).split("/")[-1].replace(".pt", "")
    
    print(f"\nLoaded SAE from run: {run_id}")
    print(f"Model: {model_name}")
    print(f"SAE type: {config.saes.sae_type}")
    print(f"SAE positions: {sae_transformer.raw_sae_positions}")
    
    # Determine training tokens from config if not specified
    training_tokens = args.training_tokens
    if training_tokens < 0 and config.data.n_train_samples is not None:
        # Estimate training tokens from samples and context length
        context_length = getattr(config.data, 'context_length', 128)
        training_tokens = config.data.n_train_samples * context_length
    
    # Determine dtype for wrapper
    dtype_str = args.llm_dtype or get_model_config(model_name)["dtype"]
    dtype = general_utils.str_to_dtype(dtype_str)
    
    # Create SAEBench wrappers
    print("\nCreating SAEBench wrappers...")
    selected_saes = create_saebench_saes_from_transformer(
        sae_transformer=sae_transformer,
        config=config,
        device=device,
        dtype=dtype,
        training_tokens=training_tokens,
    )
    
    print(f"Created {len(selected_saes)} SAE wrappers:")
    for sae_id, sae in selected_saes:
        print(f"  - {sae_id}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    
    # Run evaluations
    # Determine whether to save activations (default True, unless --no_save_activations)
    save_activations = not args.no_save_activations
    
    results = run_evaluations(
        selected_saes=selected_saes,
        model_name=model_name,
        eval_types=args.eval_types,
        device=args.device,
        output_path=str(output_path),
        llm_batch_size=args.llm_batch_size,
        llm_dtype=args.llm_dtype,
        api_key=api_key,
        force_rerun=args.force_rerun,
        save_activations=save_activations,
        random_seed=args.random_seed,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for eval_type, result in results.items():
        print(f"\n{eval_type}:")
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    print(f"  {k}: {v.item():.4f}")
    
    # Upload results to wandb
    if not args.skip_upload:
        upload_results_to_wandb(
            results=results,
            wandb_run=wandb_run,
            output_path=str(output_path),
            run_id=run_id,
        )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
