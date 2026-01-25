#!/usr/bin/env python3
"""Debug the SAEBench wrapper behavior for JumpReLU - verify decode() fix."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

import torch
import wandb
import yaml

from models.saes.jumprelu_sae import JumpReLUSAE
from sae_bench_wrapper import SAEBenchWrapper
from settings import settings
from utils.constants import CONFIG_FILE, WANDB_CACHE_DIR
from utils.enums import SAEType


def main():
    project = "gemma2-2b-65K"
    run_name = "jumprelu_target_l0_32_n_dict_components_65536"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("JumpReLU Wrapper Debug - Testing decode() Fix")
    print("=" * 60)
    
    if settings.wandb_api_key:
        wandb.login(key=settings.wandb_api_key)
    
    api = wandb.Api()
    entity = settings.wandb_entity
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    run = list(runs)[0]
    wandb_run = f"{entity}/{project}/{run.id}"
    
    model_cache_dir = Path(WANDB_CACHE_DIR) / wandb_run
    
    train_config_files = [f for f in run.files() if f.name.endswith(CONFIG_FILE)]
    train_config_file = train_config_files[0].download(exist_ok=True, replace=True, root=model_cache_dir).name
    
    checkpoints = [f for f in run.files() if f.name.endswith(".pt")]
    latest_ckpt = sorted(checkpoints, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1]))[-1]
    latest_ckpt_file = latest_ckpt.download(exist_ok=True, replace=True, root=model_cache_dir).name
    
    with open(train_config_file, "r") as f:
        config = yaml.safe_load(f)
    sae_config = config["saes"]["value"]
    
    checkpoint = torch.load(latest_ckpt_file, map_location="cpu")
    
    input_size = 2304
    sae = JumpReLUSAE(
        input_size=input_size,
        n_dict_components=sae_config.get("n_dict_components", 65536),
        target_l0=sae_config.get("target_l0", 32.0),
        bandwidth=sae_config.get("bandwidth", 0.001),
        initial_threshold=sae_config.get("initial_threshold", 0.001),
        use_pre_enc_bias=sae_config.get("use_pre_enc_bias", False),
        normalize_activations=sae_config.get("normalize_activations", True),
        sparsity_warmup_steps=sae_config.get("sparsity_warmup_steps", 5000),
        init_decoder_orthogonal=False,
        tied_encoder_init=False,
    )
    
    sae_key_prefix = "blocks-12-hook_resid_pre"
    sae_state_dict = {k[len(sae_key_prefix)+1:]: v for k, v in checkpoint.items() if k.startswith(sae_key_prefix + ".")}
    sae.load_state_dict(sae_state_dict)
    sae = sae.to(device).eval()
    
    print(f"\nSAE loaded successfully")
    print(f"  normalize_activations: {sae.normalize_activations}")
    print(f"  running_norm_factor: {sae.running_norm_factor.item():.4f}")
    print(f"  norm_factor_initialized: {sae.norm_factor_initialized.item()}")
    
    print("\n" + "=" * 60)
    print("Creating Wrapper")
    print("=" * 60)
    
    wrapper = SAEBenchWrapper(
        sae=sae, sae_type=SAEType.JUMP_RELU, model_name="google/gemma-2-2b",
        hook_layer=12, hook_name="blocks.12.hook_resid_pre", device=device, dtype=torch.float32,
    )
    print("Wrapper created successfully")
    
    print("\n" + "=" * 60)
    print("Forward Pass Comparison")
    print("=" * 60)
    
    # Create test input matching training distribution
    torch.manual_seed(42)
    norm_factor = sae.running_norm_factor.item()
    scale = norm_factor / (sae.input_size ** 0.5)
    x = torch.randn(16, 128, sae.input_size, device=device) * scale
    
    print(f"\nInput stats:")
    print(f"  Mean squared norm: {(x**2).sum(dim=-1).mean().sqrt():.4f}")
    print(f"  Expected (norm_factor): {norm_factor:.4f}")
    
    with torch.no_grad():
        # Original SAE
        sae_out = sae(x)
        original_recon = sae_out.output_raw
        original_feats = sae_out.c
        
        # Wrapper methods
        wrapper_forward = wrapper.forward(x)
        wrapper_feats = wrapper.encode(x)
        wrapper_decode = wrapper.decode(wrapper_feats)
    
    print(f"\n" + "-" * 40)
    print("Reconstruction Magnitudes:")
    print("-" * 40)
    print(f"  Original output_raw: {(original_recon**2).sum(dim=-1).mean().sqrt():.4f}")
    print(f"  Wrapper forward():   {(wrapper_forward**2).sum(dim=-1).mean().sqrt():.4f}")
    print(f"  Wrapper decode():    {(wrapper_decode**2).sum(dim=-1).mean().sqrt():.4f}")
    
    print(f"\n" + "-" * 40)
    print("Differences (should be near zero after fix):")
    print("-" * 40)
    diff_forward = (wrapper_forward - original_recon).abs().mean().item()
    diff_decode = (wrapper_decode - original_recon).abs().mean().item()
    print(f"  wrapper.forward() vs sae.output_raw: {diff_forward:.6f}")
    print(f"  wrapper.decode() vs sae.output_raw:  {diff_decode:.6f}")
    
    print(f"\n" + "-" * 40)
    print("Explained Variance:")
    print("-" * 40)
    var_x = x.var().item()
    mse_orig = ((x - original_recon)**2).mean().item()
    mse_fwd = ((x - wrapper_forward)**2).mean().item()
    mse_dec = ((x - wrapper_decode)**2).mean().item()
    
    ev_orig = (1 - mse_orig/var_x) * 100
    ev_fwd = (1 - mse_fwd/var_x) * 100
    ev_dec = (1 - mse_dec/var_x) * 100
    
    print(f"  Original SAE:      {ev_orig:.2f}%")
    print(f"  Wrapper forward(): {ev_fwd:.2f}%")
    print(f"  Wrapper decode():  {ev_dec:.2f}%")
    
    print(f"\n" + "-" * 40)
    print("L0 (Feature Sparsity):")
    print("-" * 40)
    orig_l0 = (original_feats != 0).float().sum(dim=-1).mean().item()
    wrap_l0 = (wrapper_feats != 0).float().sum(dim=-1).mean().item()
    print(f"  Original: {orig_l0:.1f}")
    print(f"  Wrapper:  {wrap_l0:.1f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if diff_decode < 0.01:
        print("\n✓ SUCCESS: wrapper.decode() output matches sae.output_raw!")
        print("  The decode() fix is working correctly.")
    else:
        print(f"\n✗ ISSUE: wrapper.decode() differs from sae.output_raw by {diff_decode:.6f}")
        print("  The fix may need further investigation.")
    
    if abs(ev_dec - ev_orig) < 1.0:
        print(f"\n✓ SUCCESS: Explained variance matches ({ev_dec:.2f}% vs {ev_orig:.2f}%)")
    else:
        print(f"\n✗ ISSUE: Explained variance mismatch ({ev_dec:.2f}% vs {ev_orig:.2f}%)")


if __name__ == "__main__":
    main()
