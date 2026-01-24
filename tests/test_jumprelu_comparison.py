"""
Side-by-side comparison of dictionary_learning JumpReLU and our JumpReLUSAE implementation.

This script runs both implementations with identical settings on the same data
to identify the source of any divergence in L0 convergence.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python tests/test_jumprelu_comparison.py

Key differences being tested:
1. Norm factor computation: dictionary_learning computes once at start over 100 batches
2. Sparsity warmup: Both should use linear warmup over 5000 steps
3. LR schedule: dictionary_learning uses linear warmup + constant + linear decay
4. Training loop mechanics: optimizer step order, gradient clipping, etc.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Callable
from dataclasses import dataclass
from collections import namedtuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our implementation
from models.saes.jumprelu_sae import JumpReLUSAE
from models.saes.activations import JumpReLUFunction, StepFunction


# ============================================================================
# Copied from dictionary_learning to avoid dependency issues
# ============================================================================

def get_lr_schedule(
    total_steps: int,
    warmup_steps: int,
    decay_start: Optional[int] = None,
    resample_steps: Optional[int] = None,
    sparsity_warmup_steps: Optional[int] = None,
) -> Callable[[int], float]:
    """LR schedule from dictionary_learning."""
    if decay_start is not None:
        assert resample_steps is None
        assert 0 <= decay_start < total_steps
        assert decay_start > warmup_steps
        if sparsity_warmup_steps is not None:
            assert decay_start > sparsity_warmup_steps

    assert 0 <= warmup_steps < total_steps

    if resample_steps is None:
        def lr_schedule(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            if decay_start is not None and step >= decay_start:
                return (total_steps - step) / (total_steps - decay_start)
            return 1.0
    else:
        def lr_schedule(step: int) -> float:
            return min((step % resample_steps) / warmup_steps, 1.0)

    return lr_schedule


def get_sparsity_warmup_fn(
    total_steps: int, sparsity_warmup_steps: Optional[int] = None
) -> Callable[[int], float]:
    """Sparsity warmup function from dictionary_learning."""
    if sparsity_warmup_steps is not None:
        assert 0 <= sparsity_warmup_steps < total_steps

    def sparsity_warmup(step: int) -> float:
        if not sparsity_warmup_steps:
            return 1.0
        return min(step / sparsity_warmup_steps, 1.0)

    return sparsity_warmup


class JumpReluAutoEncoder(nn.Module):
    """JumpReLU autoencoder copied from dictionary_learning for comparison.
    
    Note: dictionary_learning has shapes:
    - W_enc: (activation_dim, dict_size)
    - W_dec: (dict_size, activation_dim)
    
    encode: x @ W_enc + b_enc -> JumpReLU -> f
    decode: f @ W_dec + b_dec -> x_hat
    """
    
    def __init__(self, activation_dim: int, dict_size: int, device: str = "cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        
        # Parameters (matching dictionary_learning shapes)
        # W_enc: (activation_dim, dict_size)
        # W_dec: (dict_size, activation_dim)
        self.W_enc = nn.Parameter(torch.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(torch.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(torch.empty(dict_size, activation_dim, device=device))
        self.b_dec = nn.Parameter(torch.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(torch.ones(dict_size, device=device) * 0.001)
        
        # Initialize weights (matching dictionary_learning)
        nn.init.kaiming_uniform_(self.W_dec)
        # Normalize decoder rows to unit norm
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)
        # Encoder = Decoder.T
        self.W_enc.data = self.W_dec.data.clone().T
    
    def decode(self, f):
        # f @ W_dec: (batch, dict_size) @ (dict_size, activation_dim) -> (batch, activation_dim)
        return f @ self.W_dec + self.b_dec


# ============================================================================


@dataclass
class ComparisonConfig:
    """Configuration for the comparison test."""
    # Model dimensions (Gemma-2-2B hidden size)
    activation_dim: int = 2304
    dict_size: int = 16384  # 2^14
    
    # Training hyperparameters (matching dictionary_learning)
    target_l0: float = 20.0
    lr: float = 7e-5
    bandwidth: float = 0.001
    initial_threshold: float = 0.001
    sparsity_penalty: float = 1.0
    
    # Schedule parameters
    warmup_steps: int = 1000
    sparsity_warmup_steps: int = 5000
    total_steps: int = 10000
    decay_start_fraction: float = 0.8
    
    # Batch settings
    batch_size: int = 2048  # dictionary_learning default for Gemma-2-2b
    
    # Normalization
    norm_factor_estimation_steps: int = 100
    
    # Logging
    log_every: int = 100
    
    # Device
    device: str = "cuda:0"


class DictionaryLearningTrainer:
    """Wrapper around dictionary_learning's JumpReluTrainer for comparison."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.device = config.device
        
        # Create the autoencoder
        self.ae = JumpReluAutoEncoder(
            activation_dim=config.activation_dim,
            dict_size=config.dict_size,
            device=config.device,
        ).to(config.device)
        
        # Set threshold
        self.ae.threshold.data.fill_(config.initial_threshold)
        
        # Create optimizer (matching JumpReluTrainer)
        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), 
            lr=config.lr, 
            betas=(0.0, 0.999), 
            eps=1e-8
        )
        
        # Create LR schedule
        decay_start = int(config.total_steps * config.decay_start_fraction)
        lr_fn = get_lr_schedule(
            config.total_steps,
            config.warmup_steps,
            decay_start,
            resample_steps=None,
            sparsity_warmup_steps=config.sparsity_warmup_steps,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        # Create sparsity warmup function
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            config.total_steps, 
            config.sparsity_warmup_steps
        )
        
        self.bandwidth = config.bandwidth
        self.target_l0 = config.target_l0
        self.sparsity_coefficient = config.sparsity_penalty
        
    def loss(self, x: torch.Tensor, step: int):
        """Compute loss exactly as dictionary_learning does."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        x = x.to(self.ae.W_enc.dtype)
        
        # Forward pass (matching dictionary_learning/trainers/jumprelu.py)
        # x @ W_enc: (batch, activation_dim) @ (activation_dim, dict_size) -> (batch, dict_size)
        pre_jump = x @ self.ae.W_enc + self.ae.b_enc
        f = JumpReLUFunction.apply(pre_jump, self.ae.threshold, self.bandwidth)
        recon = self.ae.decode(f)
        
        # Losses
        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        l0 = StepFunction.apply(f, self.ae.threshold, self.bandwidth).sum(dim=-1).mean()
        sparsity_loss = self.sparsity_coefficient * ((l0 / self.target_l0) - 1).pow(2) * sparsity_scale
        
        total_loss = recon_loss + sparsity_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'l0': l0.item(),
            'sparsity_scale': sparsity_scale,
        }
    
    def update(self, step: int, x: torch.Tensor):
        """Single training step exactly as dictionary_learning does."""
        x = x.to(self.device)
        
        # Compute loss
        loss_info = self.loss(x, step)
        loss = loss_info['loss']
        loss.backward()
        
        # Remove gradient parallel to decoder directions
        # dictionary_learning/trainers/jumprelu.py does this with transposed W_dec
        # because their W_dec is (dict_size, activation_dim), they transpose to get (activation_dim, dict_size)
        # then normalize along dim=0 (activation_dim) so each feature column has unit norm
        # 
        # W_dec shape: (dict_size, activation_dim)
        # W_dec.T shape: (activation_dim, dict_size) - this is what they use for gradient manipulation
        W_dec_T = self.ae.W_dec.data.T  # (activation_dim, dict_size)
        W_dec_grad_T = self.ae.W_dec.grad.T  # (activation_dim, dict_size)
        
        # Normalize along dim=0 (activation_dim) to get unit norm feature columns
        normed_W_dec_T = W_dec_T / (torch.norm(W_dec_T, dim=0, keepdim=True) + 1e-6)
        parallel_component = (W_dec_grad_T * normed_W_dec_T).sum(dim=0, keepdim=True)
        modified_grad_T = W_dec_grad_T - parallel_component * normed_W_dec_T
        
        # Set gradient back (transpose back)
        self.ae.W_dec.grad = modified_grad_T.T
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Normalize decoder to unit norm
        # dictionary_learning normalizes W_dec.T along dim=0 (activation_dim)
        # which is the same as normalizing W_dec along dim=1 (activation_dim)
        eps = torch.finfo(self.ae.W_dec.dtype).eps
        norm = torch.norm(self.ae.W_dec.data, dim=1, keepdim=True)
        self.ae.W_dec.data /= norm + eps
        
        loss_info['lr'] = self.optimizer.param_groups[0]['lr']
        loss_info['threshold_mean'] = self.ae.threshold.mean().item()
        
        return loss_info


class OurTrainer:
    """Wrapper around our JumpReLUSAE for comparison."""
    
    def __init__(self, config: ComparisonConfig, norm_factor: float = 1.0, use_internal_normalization: bool = False):
        self.config = config
        self.device = config.device
        self.norm_factor = norm_factor
        self.use_internal_normalization = use_internal_normalization
        
        # Create our SAE
        self.sae = JumpReLUSAE(
            input_size=config.activation_dim,
            n_dict_components=config.dict_size,
            target_l0=config.target_l0,
            bandwidth=config.bandwidth,
            initial_threshold=config.initial_threshold,
            use_pre_enc_bias=False,
            normalize_activations=use_internal_normalization,  # Can test both modes
            sparsity_warmup_steps=config.sparsity_warmup_steps,
            sparsity_coeff=config.sparsity_penalty,
            mse_coeff=1.0,
            init_decoder_orthogonal=True,
            tied_encoder_init=True,
        ).to(config.device)
        
        # If using internal normalization, pre-set the norm factor
        if use_internal_normalization:
            self.sae.set_norm_factor(norm_factor)
        
        # Create optimizer
        optimizer_kwargs = self.sae.get_optimizer_kwargs()
        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=config.lr,
            **optimizer_kwargs
        )
        
        # Create LR schedule (matching dictionary_learning)
        decay_start = int(config.total_steps * config.decay_start_fraction)
        lr_fn = get_lr_schedule(
            config.total_steps,
            config.warmup_steps,
            decay_start,
            resample_steps=None,
            sparsity_warmup_steps=config.sparsity_warmup_steps,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
    def update(self, step: int, x: torch.Tensor):
        """Single training step matching dictionary_learning order."""
        x = x.to(self.device)
        
        # Normalize input (externally if not using internal normalization)
        if self.use_internal_normalization:
            x_input = x  # SAE will normalize internally
        else:
            x_input = x / self.norm_factor  # External normalization
        
        # Forward pass
        self.sae.train()
        output = self.sae(x_input)
        
        # Compute loss
        loss_output = self.sae.compute_loss(output)
        loss = loss_output.loss
        loss.backward()
        
        # Pre-optimizer hooks (gradient manipulation)
        self.sae.on_before_optimizer_step()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()  # Match dictionary_learning: scheduler after optimizer
        self.optimizer.zero_grad()
        
        # Post-optimizer hooks (decoder normalization, step counter)
        self.sae.on_after_optimizer_step()
        
        return {
            'loss': loss.item(),
            'recon_loss': loss_output.loss_dict['recon_loss'].item(),
            'sparsity_loss': loss_output.loss_dict['sparsity_loss'].item(),
            'l0': loss_output.loss_dict['l0_norm'].item(),
            'sparsity_scale': loss_output.loss_dict['sparsity_warmup_factor'].item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'threshold_mean': self.sae.jumprelu.threshold.mean().item(),
        }


def generate_synthetic_activations(
    batch_size: int, 
    activation_dim: int, 
    device: str,
    scale: float = 50.0,  # Approximate Gemma-2 activation magnitude
) -> torch.Tensor:
    """Generate synthetic activations with realistic magnitude."""
    # Generate random activations with magnitude similar to Gemma-2-2b
    # Gemma-2-2b activations typically have sqrt(E[||x||²]) ~ 50-100
    return torch.randn(batch_size, activation_dim, device=device) * (scale / (activation_dim ** 0.5))


def compute_norm_factor(
    batch_size: int,
    activation_dim: int,
    device: str,
    num_batches: int = 100,
    activation_scale: float = 50.0,
) -> float:
    """Compute norm factor over multiple batches (matching dictionary_learning)."""
    total_mean_squared_norm = 0.0
    
    for _ in range(num_batches):
        act = generate_synthetic_activations(batch_size, activation_dim, device, activation_scale)
        mean_squared_norm = torch.mean(torch.sum(act ** 2, dim=1))
        total_mean_squared_norm += mean_squared_norm.item()
    
    average_mean_squared_norm = total_mean_squared_norm / num_batches
    norm_factor = average_mean_squared_norm ** 0.5
    
    print(f"Computed norm factor over {num_batches} batches: {norm_factor:.4f}")
    return norm_factor


def run_comparison(config: ComparisonConfig):
    """Run side-by-side comparison of both implementations."""
    
    print("=" * 80)
    print("JumpReLU Implementation Comparison Test")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  activation_dim: {config.activation_dim}")
    print(f"  dict_size: {config.dict_size}")
    print(f"  target_l0: {config.target_l0}")
    print(f"  lr: {config.lr}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  warmup_steps: {config.warmup_steps}")
    print(f"  sparsity_warmup_steps: {config.sparsity_warmup_steps}")
    print(f"  total_steps: {config.total_steps}")
    print(f"  device: {config.device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Compute norm factor (matching dictionary_learning)
    print(f"\nComputing norm factor over {config.norm_factor_estimation_steps} batches...")
    norm_factor = compute_norm_factor(
        config.batch_size,
        config.activation_dim,
        config.device,
        config.norm_factor_estimation_steps,
    )
    
    # Reset seed for training
    torch.manual_seed(42)
    
    # Create trainers
    print("\nInitializing trainers...")
    dict_trainer = DictionaryLearningTrainer(config)
    our_trainer = OurTrainer(config, norm_factor)
    
    # Sync initial weights for fair comparison
    print("Syncing initial weights...")
    with torch.no_grad():
        # dictionary_learning shapes:
        #   W_enc: (activation_dim, dict_size)
        #   W_dec: (dict_size, activation_dim)
        # Our implementation (nn.Linear) shapes:
        #   encoder.weight: (dict_size, activation_dim)  [transposed for F.linear]
        #   decoder.weight: (activation_dim, dict_size)  [transposed for F.linear]
        
        # Normalize dictionary_learning decoder first (along dim=1, activation_dim)
        eps = torch.finfo(dict_trainer.ae.W_dec.dtype).eps
        norm = torch.norm(dict_trainer.ae.W_dec.data, dim=1, keepdim=True)
        dict_trainer.ae.W_dec.data /= norm + eps
        
        # Copy weights (note the transposes!)
        # dictionary_learning W_enc (activation_dim, dict_size) -> our encoder.weight (dict_size, activation_dim)
        our_trainer.sae.encoder.weight.data.copy_(dict_trainer.ae.W_enc.data.T)
        # dictionary_learning W_dec (dict_size, activation_dim) -> our decoder.weight (activation_dim, dict_size)
        our_trainer.sae.decoder.weight.data.copy_(dict_trainer.ae.W_dec.data.T)
        our_trainer.sae.encoder_bias.data.copy_(dict_trainer.ae.b_enc.data)
        our_trainer.sae.decoder_bias.data.copy_(dict_trainer.ae.b_dec.data)
        our_trainer.sae.jumprelu.threshold.data.copy_(dict_trainer.ae.threshold.data)
    
    # Training loop
    print("\nStarting training comparison...")
    print("-" * 80)
    print(f"{'Step':>6} | {'DL L0':>8} {'DL Recon':>10} {'DL Spars':>10} {'DL Thresh':>10} | "
          f"{'Ours L0':>8} {'Ours Recon':>10} {'Ours Spars':>10} {'Ours Thresh':>10}")
    print("-" * 80)
    
    results = {
        'dict_learning': [],
        'ours': [],
    }
    
    for step in range(config.total_steps):
        # Generate same batch for both
        torch.manual_seed(step)  # Ensure same data
        raw_act = generate_synthetic_activations(
            config.batch_size, 
            config.activation_dim, 
            config.device
        )
        
        # Normalize (same as dictionary_learning does externally)
        act_normalized = raw_act / norm_factor
        
        # Train dictionary_learning
        dl_info = dict_trainer.update(step, act_normalized)
        
        # Train ours (note: we pass raw activations, trainer normalizes internally)
        our_info = our_trainer.update(step, raw_act)
        
        # Store results
        results['dict_learning'].append(dl_info)
        results['ours'].append(our_info)
        
        # Log
        if step % config.log_every == 0 or step == config.total_steps - 1:
            print(f"{step:>6} | "
                  f"{dl_info['l0']:>8.2f} {dl_info['recon_loss']:>10.4f} {dl_info['sparsity_loss']:>10.4f} {dl_info['threshold_mean']:>10.6f} | "
                  f"{our_info['l0']:>8.2f} {our_info['recon_loss']:>10.4f} {our_info['sparsity_loss']:>10.4f} {our_info['threshold_mean']:>10.6f}")
    
    print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Final metrics
    dl_final = results['dict_learning'][-1]
    our_final = results['ours'][-1]
    
    print(f"\nFinal L0:")
    print(f"  dictionary_learning: {dl_final['l0']:.2f}")
    print(f"  Ours:                {our_final['l0']:.2f}")
    print(f"  Target:              {config.target_l0}")
    
    print(f"\nFinal threshold mean:")
    print(f"  dictionary_learning: {dl_final['threshold_mean']:.6f}")
    print(f"  Ours:                {our_final['threshold_mean']:.6f}")
    
    print(f"\nFinal reconstruction loss:")
    print(f"  dictionary_learning: {dl_final['recon_loss']:.4f}")
    print(f"  Ours:                {our_final['recon_loss']:.4f}")
    
    # Check L0 at key points
    key_steps = [0, 100, 500, 1000, 2000, 5000, config.total_steps - 1]
    print(f"\nL0 at key steps:")
    print(f"{'Step':>6} | {'DL L0':>10} | {'Ours L0':>10} | {'Diff':>10}")
    print("-" * 45)
    for s in key_steps:
        if s < len(results['dict_learning']):
            dl_l0 = results['dict_learning'][s]['l0']
            our_l0 = results['ours'][s]['l0']
            diff = our_l0 - dl_l0
            print(f"{s:>6} | {dl_l0:>10.2f} | {our_l0:>10.2f} | {diff:>+10.2f}")
    
    return results


def run_quick_diagnostic(config: ComparisonConfig):
    """Run a quick diagnostic to check if basic components match."""
    
    print("\n" + "=" * 80)
    print("QUICK DIAGNOSTIC")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    # Create small test
    batch_size = 64
    x = torch.randn(batch_size, config.activation_dim, device=config.device)
    
    # Create both autoencoders
    dl_ae = JumpReluAutoEncoder(
        activation_dim=config.activation_dim,
        dict_size=config.dict_size,
        device=config.device,
    ).to(config.device)
    
    our_sae = JumpReLUSAE(
        input_size=config.activation_dim,
        n_dict_components=config.dict_size,
        target_l0=config.target_l0,
        normalize_activations=False,
        sparsity_warmup_steps=0,  # No warmup for diagnostic
    ).to(config.device)
    
    # Copy weights (note the transpose between shapes!)
    # dictionary_learning: W_enc (activation_dim, dict_size), W_dec (dict_size, activation_dim)
    # Our nn.Linear:       encoder.weight (dict_size, activation_dim), decoder.weight (activation_dim, dict_size)
    with torch.no_grad():
        # Normalize dl decoder (along dim=1, activation_dim)
        eps = torch.finfo(dl_ae.W_dec.dtype).eps
        norm = torch.norm(dl_ae.W_dec.data, dim=1, keepdim=True)
        dl_ae.W_dec.data /= norm + eps
        
        # W_enc (activation_dim, dict_size) -> encoder.weight (dict_size, activation_dim)
        our_sae.encoder.weight.data.copy_(dl_ae.W_enc.data.T)
        # W_dec (dict_size, activation_dim) -> decoder.weight (activation_dim, dict_size)
        our_sae.decoder.weight.data.copy_(dl_ae.W_dec.data.T)
        our_sae.encoder_bias.data.copy_(dl_ae.b_enc.data)
        our_sae.decoder_bias.data.copy_(dl_ae.b_dec.data)
        our_sae.jumprelu.threshold.data.copy_(dl_ae.threshold.data)
    
    # Forward pass - dictionary_learning
    # x @ W_enc: (batch, activation_dim) @ (activation_dim, dict_size) -> (batch, dict_size)
    with torch.no_grad():
        dl_pre_jump = x @ dl_ae.W_enc + dl_ae.b_enc
        dl_f = JumpReLUFunction.apply(dl_pre_jump, dl_ae.threshold, config.bandwidth)
        dl_recon = dl_ae.decode(dl_f)
        dl_l0 = StepFunction.apply(dl_f, dl_ae.threshold, config.bandwidth).sum(dim=-1).mean()
    
    # Forward pass - ours
    with torch.no_grad():
        our_output = our_sae(x)
        our_pre_jump = our_output.pre_activations
        our_f = our_output.c
        our_recon = our_output.output
        our_l0 = our_output.l0.mean()
    
    print(f"\nPre-activations match: {torch.allclose(dl_pre_jump, our_pre_jump, atol=1e-5)}")
    print(f"  Max diff: {(dl_pre_jump - our_pre_jump).abs().max().item():.2e}")
    
    print(f"\nFeature activations match: {torch.allclose(dl_f, our_f, atol=1e-5)}")
    print(f"  Max diff: {(dl_f - our_f).abs().max().item():.2e}")
    
    print(f"\nReconstruction match: {torch.allclose(dl_recon, our_recon, atol=1e-5)}")
    print(f"  Max diff: {(dl_recon - our_recon).abs().max().item():.2e}")
    
    print(f"\nL0 values:")
    print(f"  dictionary_learning: {dl_l0.item():.2f}")
    print(f"  Ours:                {our_l0.item():.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare JumpReLU implementations")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--target_l0", type=float, default=20.0, help="Target L0")
    parser.add_argument("--dict_size", type=int, default=16384, help="Dictionary size")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnostic only")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    
    args = parser.parse_args()
    
    config = ComparisonConfig(
        device=args.device,
        total_steps=args.steps,
        batch_size=args.batch_size,
        target_l0=args.target_l0,
        dict_size=args.dict_size,
        log_every=args.log_every,
    )
    
    if args.quick:
        run_quick_diagnostic(config)
    else:
        run_quick_diagnostic(config)
        run_comparison(config)
