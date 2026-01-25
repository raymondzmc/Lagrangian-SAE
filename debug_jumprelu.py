#!/usr/bin/env python3
"""
Debug script for JumpReLU SAE - investigate norm_factor issues.

This script:
1. Loads the JumpReLU SAE checkpoint from wandb
2. Inspects the state dict buffers (running_norm_factor, norm_factor_initialized)
3. Tests forward pass behavior
4. Compares output vs output_raw
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

import torch
import wandb
import yaml

from models.saes.jumprelu_sae import JumpReLUSAE
from settings import settings
from utils.constants import CONFIG_FILE, WANDB_CACHE_DIR


def main():
    # Configuration
    project = "gemma2-2b-65K"
    run_name = "jumprelu_target_l0_32_n_dict_components_65536"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("JumpReLU SAE Debug Analysis")
    print("=" * 60)
    
    # Login to wandb
    if settings.wandb_api_key:
        wandb.login(key=settings.wandb_api_key)
    else:
        wandb.login()
    
    # Find the run
    api = wandb.Api()
    entity = settings.wandb_entity
    print(f"\nSearching for run '{run_name}' in project '{entity}/{project}'...")
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    runs_list = list(runs)
    
    if len(runs_list) == 0:
        raise ValueError(f"No run found with name '{run_name}'")
    
    run = runs_list[0]
    wandb_run = f"{entity}/{project}/{run.id}"
    print(f"Found run: {wandb_run}")
    
    # Download checkpoint and config directly (without loading full model)
    print(f"\nDownloading checkpoint from wandb...")
    model_cache_dir = Path(WANDB_CACHE_DIR) / wandb_run
    
    # Get config file
    train_config_files_remote = [file for file in run.files() if file.name.endswith(CONFIG_FILE)]
    assert len(train_config_files_remote) > 0, f"Cannot find config file"
    train_config_file_remote = train_config_files_remote[0]
    train_config_file = train_config_file_remote.download(exist_ok=True, replace=True, root=model_cache_dir).name
    
    # Get latest checkpoint
    checkpoints = [file for file in run.files() if file.name.endswith(".pt")]
    assert len(checkpoints) > 0, f"Cannot find any checkpoints"
    latest_checkpoint_remote = sorted(checkpoints, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1]))[-1]
    latest_checkpoint_file = latest_checkpoint_remote.download(exist_ok=True, replace=True, root=model_cache_dir).name
    
    print(f"Config file: {train_config_file}")
    print(f"Checkpoint file: {latest_checkpoint_file}")
    
    # Load config
    with open(train_config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract SAE config
    sae_config = config["saes"]["value"]
    print(f"\nSAE Config from wandb:")
    for key, value in sae_config.items():
        print(f"  {key}: {value}")
    
    # Load checkpoint (just the state dict)
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(latest_checkpoint_file, map_location="cpu")
    
    # Print all keys in checkpoint
    print(f"\nCheckpoint keys:")
    for key in sorted(checkpoint.keys()):
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Look for norm_factor in checkpoint
    print("\n" + "=" * 60)
    print("Looking for norm_factor in checkpoint")
    print("=" * 60)
    
    for key in checkpoint.keys():
        if "norm" in key.lower():
            val = checkpoint[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.item() if val.numel() == 1 else val.shape}")
            else:
                print(f"  {key}: {val}")
        if "initialized" in key.lower():
            val = checkpoint[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.item() if val.numel() == 1 else val.shape}")
            else:
                print(f"  {key}: {val}")
    
    # Create JumpReLU SAE with config and load checkpoint
    print("\n" + "=" * 60)
    print("Creating JumpReLU SAE and loading checkpoint")
    print("=" * 60)
    
    # Get input size from config (Gemma-2-2b d_model is 2304)
    input_size = 2304
    
    # Create SAE (without orthogonal init to avoid CUDA issues)
    sae = JumpReLUSAE(
        input_size=input_size,
        n_dict_components=sae_config.get("n_dict_components", 65536),
        target_l0=sae_config.get("target_l0", 32.0),
        bandwidth=sae_config.get("bandwidth", 0.001),
        initial_threshold=sae_config.get("initial_threshold", 0.001),
        use_pre_enc_bias=sae_config.get("use_pre_enc_bias", False),
        normalize_activations=sae_config.get("normalize_activations", True),
        sparsity_warmup_steps=sae_config.get("sparsity_warmup_steps", 5000),
        init_decoder_orthogonal=False,  # Skip to avoid CUDA issues
        tied_encoder_init=False,  # Skip to avoid issues
    )
    
    # Get the sae key from checkpoint (usually "blocks-12-hook_resid_pre" or similar)
    sae_key_prefix = None
    for key in checkpoint.keys():
        if "encoder.weight" in key:
            # Extract prefix like "blocks-12-hook_resid_pre"
            sae_key_prefix = key.replace(".encoder.weight", "")
            break
    
    print(f"SAE key prefix in checkpoint: {sae_key_prefix}")
    
    # Load state dict with proper key mapping
    if sae_key_prefix:
        # Extract just this SAE's state dict
        sae_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith(sae_key_prefix + "."):
                new_key = key[len(sae_key_prefix) + 1:]  # Remove prefix
                sae_state_dict[new_key] = value
        
        print(f"\nExtracted SAE state dict keys:")
        for key in sorted(sae_state_dict.keys()):
            if isinstance(sae_state_dict[key], torch.Tensor):
                print(f"  {key}: {sae_state_dict[key].shape if sae_state_dict[key].numel() > 1 else sae_state_dict[key].item()}")
        
        sae.load_state_dict(sae_state_dict)
    else:
        print("Could not find SAE key prefix, trying to load directly...")
        sae.load_state_dict(checkpoint)
    
    sae = sae.to(device)
    
    print(f"\nSAE type: {type(sae).__name__}")
    print(f"Input size: {sae.input_size}")
    print(f"Dict components: {sae.n_dict_components}")
    
    # Check JumpReLU-specific attributes
    print("\n" + "=" * 60)
    print("JumpReLU Configuration")
    print("=" * 60)
    
    print(f"normalize_activations: {sae.normalize_activations}")
    print(f"target_l0: {sae.target_l0}")
    print(f"bandwidth: {sae.bandwidth}")
    print(f"initial_threshold: {sae.initial_threshold}")
    
    # Inspect buffers
    print("\n" + "=" * 60)
    print("Buffer Inspection")
    print("=" * 60)
    
    print(f"running_norm_factor: {sae.running_norm_factor.item()}")
    print(f"norm_factor_initialized: {sae.norm_factor_initialized.item()}")
    print(f"sparsity_step: {sae.sparsity_step.item()}")
    
    # Check the threshold values
    threshold = sae.jumprelu.threshold
    print(f"\nThreshold stats:")
    print(f"  Shape: {threshold.shape}")
    print(f"  Min: {threshold.min().item():.6f}")
    print(f"  Max: {threshold.max().item():.6f}")
    print(f"  Mean: {threshold.mean().item():.6f}")
    print(f"  Std: {threshold.std().item():.6f}")
    
    # Test forward pass with random data
    print("\n" + "=" * 60)
    print("Forward Pass Test")
    print("=" * 60)
    
    # Create test input with magnitude matching training data
    # The norm_factor of 159.0 tells us the training data had mean squared norm of 159^2 = 25281
    # So sqrt(25281 / 2304) ≈ 3.3 is the expected std per dimension
    batch_size = 16
    seq_len = 128
    
    # Generate random input scaled to match training distribution
    # norm_factor = sqrt(mean_squared_norm) = 159
    # mean_squared_norm = sum(x^2) / batch_size = norm_factor^2
    # For standard normal with our d_model, we need to scale appropriately
    torch.manual_seed(42)
    expected_norm_factor = sae.running_norm_factor.item()
    # Standard normal has mean squared norm of d_model, so scale to get desired norm_factor
    scale = expected_norm_factor / (sae.input_size ** 0.5)
    x = torch.randn(batch_size, seq_len, sae.input_size, device=device) * scale
    
    print(f"\nInput stats:")
    print(f"  Shape: {x.shape}")
    print(f"  Mean: {x.mean().item():.4f}")
    print(f"  Std: {x.std().item():.4f}")
    print(f"  Squared norm mean (per sample): {(x**2).sum(dim=-1).mean().sqrt().item():.4f}")
    
    # Run forward pass
    sae.eval()
    with torch.no_grad():
        output = sae(x)
    
    print(f"\nOutput fields:")
    print(f"  input: {output.input.shape}")
    print(f"  c (features): {output.c.shape}")
    print(f"  output (normalized): {output.output.shape}")
    print(f"  output_raw (denormalized): {output.output_raw.shape}")
    print(f"  l0: {output.l0.shape}")
    
    # Compare outputs
    print("\n" + "=" * 60)
    print("Output Comparison")
    print("=" * 60)
    
    print(f"\nInput magnitude:")
    print(f"  Original input mean squared norm: {(x**2).sum(dim=-1).mean().sqrt().item():.4f}")
    print(f"  Normalized input mean squared norm: {(output.input**2).sum(dim=-1).mean().sqrt().item():.4f}")
    
    print(f"\nOutput magnitude:")
    print(f"  Normalized output mean squared norm: {(output.output**2).sum(dim=-1).mean().sqrt().item():.4f}")
    print(f"  Raw output mean squared norm: {(output.output_raw**2).sum(dim=-1).mean().sqrt().item():.4f}")
    
    # If norm_factor is 1.0, output and output_raw should be the same
    print(f"\nOutput vs Output_raw difference:")
    diff = (output.output - output.output_raw).abs().mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")
    
    if diff < 1e-6:
        print("  WARNING: output and output_raw are essentially the same!")
        print("  This suggests norm_factor is 1.0 (uninitialized)")
    
    # Compute reconstruction quality
    print("\n" + "=" * 60)
    print("Reconstruction Quality")
    print("=" * 60)
    
    # MSE with raw output (what SAEBench uses)
    mse_raw = ((x - output.output_raw) ** 2).mean().item()
    
    # MSE with normalized output
    mse_normalized = ((output.input - output.output) ** 2).mean().item()
    
    # Variance of input
    var_input = x.var().item()
    var_input_normalized = output.input.var().item()
    
    # Explained variance
    explained_var_raw = 1 - mse_raw / var_input
    explained_var_normalized = 1 - mse_normalized / var_input_normalized
    
    print(f"\nUsing raw output (for external evaluation):")
    print(f"  MSE: {mse_raw:.4f}")
    print(f"  Input variance: {var_input:.4f}")
    print(f"  Explained variance: {explained_var_raw:.4%}")
    
    print(f"\nUsing normalized output (internal loss):")
    print(f"  MSE: {mse_normalized:.4f}")
    print(f"  Input variance: {var_input_normalized:.4f}")
    print(f"  Explained variance: {explained_var_normalized:.4%}")
    
    # L0 stats
    print(f"\nL0 stats:")
    print(f"  Mean: {output.l0.mean().item():.2f}")
    print(f"  Std: {output.l0.std().item():.2f}")
    print(f"  Target: {sae.target_l0}")
    
    # Test what happens if we manually set norm_factor
    print("\n" + "=" * 60)
    print("Test with Corrected Norm Factor")
    print("=" * 60)
    
    # Compute what the norm factor should be
    correct_norm_factor = (x**2).sum(dim=-1).mean().sqrt().item()
    print(f"\nCorrect norm factor for this data: {correct_norm_factor:.4f}")
    print(f"Current norm factor: {sae.running_norm_factor.item():.4f}")
    
    # Temporarily set the correct norm factor and test
    old_norm_factor = sae.running_norm_factor.item()
    old_initialized = sae.norm_factor_initialized.item()
    
    sae.running_norm_factor.fill_(correct_norm_factor)
    sae.norm_factor_initialized.fill_(True)
    
    with torch.no_grad():
        output_corrected = sae(x)
    
    # Compute metrics with corrected norm factor
    mse_corrected = ((x - output_corrected.output_raw) ** 2).mean().item()
    explained_var_corrected = 1 - mse_corrected / var_input
    
    print(f"\nWith corrected norm factor:")
    print(f"  MSE: {mse_corrected:.4f}")
    print(f"  Explained variance: {explained_var_corrected:.4%}")
    print(f"  L0 mean: {output_corrected.l0.mean().item():.2f}")
    
    # Restore old values
    sae.running_norm_factor.fill_(old_norm_factor)
    sae.norm_factor_initialized.fill_(old_initialized)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if sae.running_norm_factor.item() == 1.0 and not sae.norm_factor_initialized.item():
        print("\n❌ PROBLEM IDENTIFIED: norm_factor is 1.0 and not initialized!")
        print("   The model was saved before norm_factor was computed.")
        print("\n   FIX OPTIONS:")
        print("   1. Re-train with proper norm_factor initialization")
        print("   2. Compute norm_factor from data at evaluation time")
        print("   3. Set normalize_activations=False in config")
    elif sae.running_norm_factor.item() != 1.0:
        print(f"\n✓ norm_factor is set to {sae.running_norm_factor.item():.4f}")
        print("  The norm_factor appears to be initialized.")
        if explained_var_raw < 0.5:
            print("\n⚠️  But reconstruction quality is still poor!")
            print("   Need to investigate further...")
    
    return sae, output


if __name__ == "__main__":
    main()
