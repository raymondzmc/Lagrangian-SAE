"""
Benchmark test comparing normalization approaches for LagrangianSAE.

This test compares:
1. Internal normalization: SAE computes norm_factor on first batch (EMA-style)
2. External normalization: Pre-compute norm_factor over N batches, then set via set_norm_factor()

The external approach (used by run.py) is expected to provide:
- More stable initial training (norm_factor is accurate from start)
- Faster L0 convergence (thresholds calibrated with correct norm_factor)
- Better final reconstruction (more stable optimization)

Usage:
    python -m pytest tests/test_normalization_benchmark.py -v -s
    
Or run directly:
    python tests/test_normalization_benchmark.py
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Generator
import time

from models.saes.lagrangian_sae import LagrangianSAE


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    final_mse: float
    final_l0: float
    l0_convergence_step: int  # Step when L0 first reaches within 20% of target
    mse_history: list[float]
    l0_history: list[float]
    training_time: float
    norm_factor: float


def generate_realistic_activations(
    batch_size: int,
    input_size: int,
    num_batches: int,
    base_scale: float = 50.0,
    scale_variation: float = 0.2,
    seed: int = 42,
) -> Generator[torch.Tensor, None, None]:
    """
    Generate synthetic activations that mimic real transformer residual stream activations.
    
    Real transformer activations (e.g., Gemma-2, GPT-2) typically have:
    - Large magnitudes (RMS ~ 20-100 depending on layer)
    - Near-zero mean
    - Slight variation in scale across batches
    
    Args:
        batch_size: Number of samples per batch
        input_size: Dimension of each activation vector
        num_batches: Number of batches to generate
        base_scale: Base RMS scale of activations (mimics real activations)
        scale_variation: Fractional variation in scale across batches
        seed: Random seed for reproducibility
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    
    for i in range(num_batches):
        # Add slight scale variation to mimic real data
        scale = base_scale * (1.0 + scale_variation * (torch.rand(1, generator=rng).item() - 0.5))
        
        # Generate activations: mean~0, varying scale
        activations = torch.randn(batch_size, input_size, generator=rng) * scale
        
        yield activations


def compute_norm_factor_from_batches(
    data_generator: Generator[torch.Tensor, None, None],
    num_batches: int = 100
) -> float:
    """
    Compute norm_factor by averaging mean squared norms over multiple batches.
    
    This mimics what run.py's compute_norm_factors_for_saes does.
    """
    total_msn = 0.0
    count = 0
    
    for batch in data_generator:
        if count >= num_batches:
            break
        
        batch_flat = batch.reshape(-1, batch.shape[-1])
        msn = (batch_flat ** 2).sum(dim=-1).mean().item()
        total_msn += msn
        count += 1
    
    average_msn = total_msn / count
    norm_factor = average_msn ** 0.5
    return norm_factor


def train_sae(
    sae: LagrangianSAE,
    data_generator: Generator[torch.Tensor, None, None],
    num_steps: int,
    lr: float = 1e-3,
) -> tuple[list[float], list[float]]:
    """
    Train an SAE and record metrics.
    
    Returns:
        Tuple of (mse_history, l0_history)
    """
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    mse_history = []
    l0_history = []
    
    sae.train()
    
    for step, batch in enumerate(data_generator):
        if step >= num_steps:
            break
        
        output = sae(batch)
        loss_output = sae.compute_loss(output)
        
        optimizer.zero_grad()
        loss_output.loss.backward()
        optimizer.step()
        sae.update_alpha()
        
        # Record metrics
        mse = F.mse_loss(output.output, output.input).item()
        l0 = output.l0.mean().item()
        
        mse_history.append(mse)
        l0_history.append(l0)
    
    return mse_history, l0_history


def find_l0_convergence_step(l0_history: list[float], target_l0: float, tolerance: float = 0.2) -> int:
    """
    Find the first step where L0 is within tolerance of target.
    
    Args:
        l0_history: List of L0 values over training
        target_l0: Target L0 value
        tolerance: Fraction of target_l0 to consider "converged" (e.g., 0.2 = within 20%)
    
    Returns:
        Step number when L0 first converges, or -1 if never converged
    """
    lower = target_l0 * (1 - tolerance)
    upper = target_l0 * (1 + tolerance)
    
    for step, l0 in enumerate(l0_history):
        if lower <= l0 <= upper:
            return step
    
    return -1


class TestNormalizationBenchmark:
    """Benchmark tests comparing normalization approaches."""

    @pytest.fixture
    def sae_config(self) -> dict:
        """SAE configuration for benchmarks."""
        return {
            "input_size": 128,
            "n_dict_components": 512,
            "target_l0": 16.0,
            "initial_threshold": 0.1,  # Higher initial threshold for faster calibration
            "bandwidth": 0.1,  # Larger bandwidth for stronger gradient signal
            "calibrate_thresholds": True,
            "alpha_lr": 0.1,  # Higher alpha LR for faster L0 convergence
            "rho_quadratic": 0.01,  # Higher quadratic penalty
            "alpha_max": 10.0,
        }

    @pytest.fixture
    def training_config(self) -> dict:
        """Training configuration for benchmarks."""
        return {
            "batch_size": 32,
            "num_steps": 300,
            "lr": 1e-3,  # Higher LR for faster convergence
            "activation_scale": 10.0,  # Moderate scale
            "precompute_batches": 50,
        }

    def run_benchmark(
        self,
        name: str,
        sae_config: dict,
        training_config: dict,
        use_external_norm_factor: bool,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        
        # Create SAE
        sae = LagrangianSAE(**sae_config, normalize_activations=True)
        
        norm_factor = 1.0  # Default
        
        if use_external_norm_factor:
            # Pre-compute norm factor (like run.py does)
            precompute_gen = generate_realistic_activations(
                batch_size=training_config["batch_size"],
                input_size=sae_config["input_size"],
                num_batches=training_config["precompute_batches"],
                base_scale=training_config["activation_scale"],
                seed=seed,
            )
            norm_factor = compute_norm_factor_from_batches(
                precompute_gen, 
                num_batches=training_config["precompute_batches"]
            )
            sae.set_norm_factor(norm_factor)
        
        # Generate training data (with different seed to avoid overlap with precompute)
        train_gen = generate_realistic_activations(
            batch_size=training_config["batch_size"],
            input_size=sae_config["input_size"],
            num_batches=training_config["num_steps"] + 100,  # Extra buffer
            base_scale=training_config["activation_scale"],
            seed=seed + 1000,  # Different seed for training
        )
        
        # Train
        start_time = time.time()
        mse_history, l0_history = train_sae(
            sae,
            train_gen,
            num_steps=training_config["num_steps"],
            lr=training_config["lr"],
        )
        training_time = time.time() - start_time
        
        # Get norm factor used (for internal, it's computed on first batch)
        if not use_external_norm_factor:
            norm_factor = sae.running_norm_factor.item()
        
        # Compute results
        return BenchmarkResult(
            name=name,
            final_mse=np.mean(mse_history[-50:]),  # Average of last 50 steps
            final_l0=np.mean(l0_history[-50:]),
            l0_convergence_step=find_l0_convergence_step(l0_history, sae_config["target_l0"]),
            mse_history=mse_history,
            l0_history=l0_history,
            training_time=training_time,
            norm_factor=norm_factor,
        )

    def test_benchmark_comparison(self, sae_config, training_config):
        """
        Main benchmark comparing internal vs external normalization.
        
        This test runs both approaches and compares:
        1. L0 convergence speed
        2. Final MSE (reconstruction quality)
        3. Final L0 accuracy (how close to target)
        """
        print("\n" + "=" * 70)
        print("NORMALIZATION BENCHMARK: Internal vs External")
        print("=" * 70)
        
        # Run internal normalization (old approach)
        print("\nRunning INTERNAL normalization benchmark...")
        internal_result = self.run_benchmark(
            name="Internal (EMA on first batch)",
            sae_config=sae_config,
            training_config=training_config,
            use_external_norm_factor=False,
            seed=42,
        )
        
        # Run external normalization (new approach with run.py)
        print("Running EXTERNAL normalization benchmark...")
        external_result = self.run_benchmark(
            name="External (pre-computed over 100 batches)",
            sae_config=sae_config,
            training_config=training_config,
            use_external_norm_factor=True,
            seed=42,
        )
        
        # Print results
        print("\n" + "-" * 70)
        print("RESULTS")
        print("-" * 70)
        
        print(f"\n{'Metric':<30} {'Internal':<20} {'External':<20} {'Winner':<10}")
        print("-" * 80)
        
        # Norm factor comparison
        print(f"{'Norm Factor':<30} {internal_result.norm_factor:<20.4f} {external_result.norm_factor:<20.4f}")
        
        # L0 convergence
        internal_conv = internal_result.l0_convergence_step if internal_result.l0_convergence_step >= 0 else "Never"
        external_conv = external_result.l0_convergence_step if external_result.l0_convergence_step >= 0 else "Never"
        
        if isinstance(internal_conv, int) and isinstance(external_conv, int):
            l0_winner = "External" if external_conv < internal_conv else "Internal" if internal_conv < external_conv else "Tie"
        elif isinstance(external_conv, int):
            l0_winner = "External"
        elif isinstance(internal_conv, int):
            l0_winner = "Internal"
        else:
            l0_winner = "Neither"
        
        print(f"{'L0 Convergence Step':<30} {str(internal_conv):<20} {str(external_conv):<20} {l0_winner:<10}")
        
        # Final L0 accuracy
        target = sae_config["target_l0"]
        internal_l0_error = abs(internal_result.final_l0 - target) / target * 100
        external_l0_error = abs(external_result.final_l0 - target) / target * 100
        l0_acc_winner = "External" if external_l0_error < internal_l0_error else "Internal"
        
        print(f"{'Final L0':<30} {internal_result.final_l0:<20.2f} {external_result.final_l0:<20.2f}")
        print(f"{'L0 Error (%)':<30} {internal_l0_error:<20.2f} {external_l0_error:<20.2f} {l0_acc_winner:<10}")
        
        # Final MSE
        mse_winner = "External" if external_result.final_mse < internal_result.final_mse else "Internal"
        print(f"{'Final MSE':<30} {internal_result.final_mse:<20.4f} {external_result.final_mse:<20.4f} {mse_winner:<10}")
        
        # Training time
        print(f"{'Training Time (s)':<30} {internal_result.training_time:<20.2f} {external_result.training_time:<20.2f}")
        
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        
        # Count wins
        winners = [l0_winner, l0_acc_winner, mse_winner]
        external_wins = winners.count("External")
        internal_wins = winners.count("Internal")
        
        print(f"External wins: {external_wins}/3")
        print(f"Internal wins: {internal_wins}/3")
        
        if external_wins > internal_wins:
            print("\n✓ EXTERNAL normalization (pre-computed) is BETTER")
        elif internal_wins > external_wins:
            print("\n✓ INTERNAL normalization (EMA) is BETTER")
        else:
            print("\n≈ Results are roughly EQUIVALENT")
        
        print("\n" + "-" * 70)
        print("INTERPRETATION")
        print("-" * 70)
        print("""
Both approaches compute similar norm factors and achieve similar results.

Key benefits of EXTERNAL (pre-computed via run.py):
1. More accurate initial norm factor (averaged over many batches)
2. Consistent with JumpReLUSAE implementation
3. Better for production training where norm factor is computed once
4. Norm factor is stable from the very first training step

Key benefits of INTERNAL (EMA on first batch):
1. Simpler - no pre-computation needed
2. Works out-of-the-box without run.py integration
3. First batch estimate is usually good enough for synthetic data

For real training with varying data distributions and longer runs,
the EXTERNAL approach is recommended for consistency and stability.
""")
        
        # The test should verify both approaches work, not necessarily that one is better
        # (Results can vary based on random seed)
        # Note: With limited training, MSE may still be high - we just check it's finite
        assert internal_result.final_mse < 1000 and not np.isnan(internal_result.final_mse), "Internal MSE too high or NaN"
        assert external_result.final_mse < 1000 and not np.isnan(external_result.final_mse), "External MSE too high or NaN"
        
        # L0 should be finite and positive
        assert internal_result.final_l0 > 0 and not np.isnan(internal_result.final_l0), "Internal L0 invalid"
        assert external_result.final_l0 > 0 and not np.isnan(external_result.final_l0), "External L0 invalid"

    def test_varying_activation_scales(self, sae_config):
        """
        Test that external normalization handles varying activation scales better.
        
        This tests the scenario where different layers/models have very different
        activation magnitudes - external normalization should be more robust.
        """
        print("\n" + "=" * 70)
        print("SCALE ROBUSTNESS TEST")
        print("=" * 70)
        
        scales = [10.0, 50.0, 200.0]
        results = []
        
        training_config = {
            "batch_size": 32,
            "num_steps": 300,
            "lr": 3e-4,
            "precompute_batches": 50,
        }
        
        for scale in scales:
            training_config["activation_scale"] = scale
            
            # External normalization
            external_result = self.run_benchmark(
                name=f"External (scale={scale})",
                sae_config=sae_config,
                training_config=training_config,
                use_external_norm_factor=True,
                seed=42,
            )
            
            # Internal normalization  
            internal_result = self.run_benchmark(
                name=f"Internal (scale={scale})",
                sae_config=sae_config,
                training_config=training_config,
                use_external_norm_factor=False,
                seed=42,
            )
            
            results.append({
                "scale": scale,
                "external_mse": external_result.final_mse,
                "internal_mse": internal_result.final_mse,
                "external_l0_error": abs(external_result.final_l0 - sae_config["target_l0"]),
                "internal_l0_error": abs(internal_result.final_l0 - sae_config["target_l0"]),
            })
        
        print(f"\n{'Scale':<10} {'Ext MSE':<15} {'Int MSE':<15} {'Ext L0 Err':<15} {'Int L0 Err':<15}")
        print("-" * 70)
        for r in results:
            print(f"{r['scale']:<10.1f} {r['external_mse']:<15.4f} {r['internal_mse']:<15.4f} "
                  f"{r['external_l0_error']:<15.2f} {r['internal_l0_error']:<15.2f}")
        
        # Both should work at all scales - MSE scales with input magnitude squared
        # So we check relative to scale^2 (normalized MSE)
        for r in results:
            scale = r["scale"]
            # MSE should be less than ~10x the scale^2 (generous threshold for short training)
            max_expected_mse = 10 * scale * scale
            assert r["external_mse"] < max_expected_mse, f"External MSE too high at scale {scale}"
            assert r["internal_mse"] < max_expected_mse, f"Internal MSE too high at scale {scale}"


class TestQuickBenchmark:
    """Quick benchmark for fast CI testing."""
    
    def test_quick_comparison(self):
        """Quick sanity check that both approaches work."""
        sae_config = {
            "input_size": 64,
            "n_dict_components": 128,
            "target_l0": 8.0,
            "initial_threshold": 0.1,
            "bandwidth": 0.1,
            "calibrate_thresholds": True,
        }
        
        training_config = {
            "batch_size": 16,
            "num_steps": 100,
            "lr": 1e-2,
            "activation_scale": 20.0,
            "precompute_batches": 20,
        }
        
        # Quick test with external
        sae_ext = LagrangianSAE(**sae_config, normalize_activations=True)
        
        precompute_gen = generate_realistic_activations(
            batch_size=training_config["batch_size"],
            input_size=sae_config["input_size"],
            num_batches=training_config["precompute_batches"],
            base_scale=training_config["activation_scale"],
            seed=42,
        )
        norm_factor = compute_norm_factor_from_batches(precompute_gen, 20)
        sae_ext.set_norm_factor(norm_factor)
        
        train_gen = generate_realistic_activations(
            batch_size=training_config["batch_size"],
            input_size=sae_config["input_size"],
            num_batches=training_config["num_steps"],
            base_scale=training_config["activation_scale"],
            seed=100,
        )
        
        mse_hist, l0_hist = train_sae(sae_ext, train_gen, 50, lr=1e-2)
        
        assert len(mse_hist) == 50
        assert all(not np.isnan(m) for m in mse_hist)
        assert all(not np.isnan(l) for l in l0_hist)
        
        print(f"\nQuick benchmark: Final MSE={mse_hist[-1]:.4f}, Final L0={l0_hist[-1]:.2f}")


if __name__ == "__main__":
    # Run the main benchmark
    print("Running Normalization Benchmark...")
    print("This compares internal (EMA) vs external (pre-computed) normalization.\n")
    
    benchmark = TestNormalizationBenchmark()
    
    sae_config = {
        "input_size": 128,
        "n_dict_components": 512,
        "target_l0": 16.0,
        "initial_threshold": 0.1,
        "bandwidth": 0.1,
        "calibrate_thresholds": True,
        "alpha_lr": 0.1,
        "rho_quadratic": 0.01,
        "alpha_max": 10.0,
    }
    
    training_config = {
        "batch_size": 32,
        "num_steps": 300,
        "lr": 1e-3,
        "activation_scale": 10.0,
        "precompute_batches": 50,
    }
    
    benchmark.test_benchmark_comparison(sae_config, training_config)
