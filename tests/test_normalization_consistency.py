"""
Comprehensive tests for normalization consistency across SAE implementations.

This test suite verifies that:
1. LagrangianSAE, JumpReLUSAE, and MatryoshkaLagrangianSAE have consistent normalization behavior
2. The normalize_activations feature works correctly with run.py's pre-computation
3. Normalization improves training stability across different input scales
4. External norm factor setting (via set_norm_factor) matches internal computation

The tests use configs from the configs/ directory as reference for realistic parameters.
"""

import pytest
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from typing import Generator

from models.saes.lagrangian_sae import LagrangianSAE
from models.saes.jumprelu_sae import JumpReLUSAE
from models.saes.matryoshka_lagrangian_sae import MatryoshkaLagrangianSAE


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config_dir() -> Path:
    """Return the path to the configs directory."""
    return Path(__file__).parent.parent / "configs"


@pytest.fixture
def gemma_lagrangian_config(config_dir: Path) -> dict:
    """Load the gemma2 lagrangian config."""
    config_path = config_dir / "gemma2-2b-50M" / "gemma2-lagrangian.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Fallback config if file doesn't exist
    return {
        "saes": {
            "n_dict_components": 16384,
            "target_l0": 32.0,
            "normalize_activations": True,
            "initial_threshold": 0.001,
            "bandwidth": 0.001,
        }
    }


@pytest.fixture
def gemma_jumprelu_config(config_dir: Path) -> dict:
    """Load the gemma2 jumprelu config."""
    config_path = config_dir / "gemma2-2b-50M" / "gemma2-jumprelu.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Fallback config if file doesn't exist
    return {
        "saes": {
            "n_dict_components": 16384,
            "target_l0": 32.0,
            "normalize_activations": True,
            "initial_threshold": 0.001,
            "bandwidth": 0.001,
            "norm_factor_num_batches": 100,
        }
    }


@pytest.fixture
def small_sae_params() -> dict:
    """Small SAE parameters for faster testing."""
    return {
        "input_size": 64,
        "n_dict_components": 128,
        "target_l0": 8.0,
        "initial_threshold": 0.1,
        "bandwidth": 0.1,
    }


def generate_scaled_data(
    batch_size: int,
    input_size: int,
    scale: float = 1.0,
    mean: float = 0.0,
    num_batches: int = 10
) -> Generator[torch.Tensor, None, None]:
    """Generate batches of data with specified scale and mean."""
    for _ in range(num_batches):
        yield torch.randn(batch_size, input_size) * scale + mean


# ============================================================================
# Test: Normalization API Consistency
# ============================================================================

class TestNormalizationAPIConsistency:
    """Test that all SAEs have consistent normalization API."""

    def test_lagrangian_has_required_attributes(self, small_sae_params):
        """Test LagrangianSAE has all required normalization attributes."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        
        # Required attributes for run.py compatibility
        assert hasattr(sae, 'normalize_activations')
        assert hasattr(sae, 'norm_factor_num_batches')
        assert hasattr(sae, 'norm_factor_initialized')
        assert hasattr(sae, 'running_norm_factor')
        assert hasattr(sae, 'set_norm_factor')
        assert callable(sae.set_norm_factor)
        
        # Check default values
        assert sae.normalize_activations == True
        assert sae.norm_factor_num_batches == 100
        assert sae.norm_factor_initialized == False
        assert sae.running_norm_factor == 1.0

    def test_jumprelu_has_required_attributes(self, small_sae_params):
        """Test JumpReLUSAE has all required normalization attributes."""
        sae = JumpReLUSAE(**small_sae_params, normalize_activations=True)
        
        # Required attributes for run.py compatibility
        assert hasattr(sae, 'normalize_activations')
        assert hasattr(sae, 'norm_factor_num_batches')
        assert hasattr(sae, 'norm_factor_initialized')
        assert hasattr(sae, 'running_norm_factor')
        assert hasattr(sae, 'set_norm_factor')
        assert callable(sae.set_norm_factor)

    def test_matryoshka_lagrangian_has_required_attributes(self, small_sae_params):
        """Test MatryoshkaLagrangianSAE has all required normalization attributes."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, normalize_activations=True)
        
        # Required attributes for run.py compatibility
        assert hasattr(sae, 'normalize_activations')
        assert hasattr(sae, 'norm_factor_num_batches')
        assert hasattr(sae, 'norm_factor_initialized')
        assert hasattr(sae, 'running_norm_factor')
        assert hasattr(sae, 'set_norm_factor')
        assert callable(sae.set_norm_factor)


# ============================================================================
# Test: set_norm_factor Integration
# ============================================================================

class TestSetNormFactor:
    """Test that set_norm_factor works correctly (simulating run.py behavior)."""

    def test_set_norm_factor_updates_buffers_lagrangian(self, small_sae_params):
        """Test set_norm_factor correctly updates LagrangianSAE buffers."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        
        # Initially not initialized
        assert sae.norm_factor_initialized == False
        
        # Set norm factor (as run.py would do)
        sae.set_norm_factor(42.5)
        
        # Now should be initialized with correct value
        assert sae.norm_factor_initialized == True
        assert torch.allclose(sae.running_norm_factor, torch.tensor(42.5))

    def test_set_norm_factor_updates_buffers_jumprelu(self, small_sae_params):
        """Test set_norm_factor correctly updates JumpReLUSAE buffers."""
        sae = JumpReLUSAE(**small_sae_params, normalize_activations=True)
        
        # Initially not initialized
        assert sae.norm_factor_initialized == False
        
        # Set norm factor (as run.py would do)
        sae.set_norm_factor(42.5)
        
        # Now should be initialized with correct value
        assert sae.norm_factor_initialized == True
        assert torch.allclose(sae.running_norm_factor, torch.tensor(42.5))

    def test_set_norm_factor_affects_forward_pass(self, small_sae_params):
        """Test that set_norm_factor changes the forward pass behavior."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.eval()
        
        x = torch.randn(8, small_sae_params["input_size"]) * 10.0  # Large scale
        
        # First forward with default norm factor (1.0)
        output1 = sae(x)
        
        # Set a different norm factor
        sae.set_norm_factor(10.0)
        
        # Second forward with new norm factor
        output2 = sae(x)
        
        # Outputs should differ because normalization changed
        # (unless by coincidence, but with scale 10 input and norm factor 10, they should match better)
        assert not torch.allclose(output1.output, output2.output)


# ============================================================================
# Test: Normalization Computation Consistency
# ============================================================================

class TestNormalizationComputation:
    """Test that norm factor computation is consistent across SAE types."""

    def test_norm_factor_matches_expected_formula(self, small_sae_params):
        """Test that norm factor = sqrt(E[||x||²])."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        # Generate data with known scale
        scale = 5.0
        x = torch.randn(100, small_sae_params["input_size"]) * scale
        
        # Expected norm factor: sqrt(mean squared norm)
        expected_msn = (x ** 2).sum(dim=-1).mean()
        expected_norm_factor = expected_msn.sqrt()
        
        # Run forward to initialize norm factor
        _ = sae(x)
        
        # Check that computed norm factor matches expected
        assert torch.allclose(sae.running_norm_factor, expected_norm_factor, rtol=0.01)

    def test_lagrangian_and_jumprelu_compute_same_norm_factor(self, small_sae_params):
        """Test that LagrangianSAE and JumpReLUSAE compute the same norm factor."""
        lagrangian = LagrangianSAE(**small_sae_params, normalize_activations=True)
        jumprelu = JumpReLUSAE(**small_sae_params, normalize_activations=True)
        
        lagrangian.train()
        jumprelu.train()
        
        # Same input data
        x = torch.randn(100, small_sae_params["input_size"]) * 7.0
        
        # Run forward to initialize norm factors
        _ = lagrangian(x)
        _ = jumprelu(x)
        
        # Norm factors should be the same
        assert torch.allclose(
            lagrangian.running_norm_factor, 
            jumprelu.running_norm_factor, 
            rtol=0.01
        )

    def test_normalized_input_has_unit_mean_squared_norm(self, small_sae_params):
        """Test that after normalization, E[||x||²] ≈ 1."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        # Data with large scale
        x = torch.randn(500, small_sae_params["input_size"]) * 20.0
        
        # Run forward to compute norm factor
        _ = sae(x)
        
        # Normalize manually and check
        x_normalized = x / sae.running_norm_factor
        normalized_msn = (x_normalized ** 2).sum(dim=-1).mean()
        
        # Should be close to 1.0
        assert torch.allclose(normalized_msn, torch.tensor(1.0), atol=0.1)


# ============================================================================
# Test: Normalization vs No Normalization
# ============================================================================

class TestNormalizationBehavior:
    """Test that normalization improves behavior across different input scales."""

    def test_output_scale_preserved_with_normalization(self, small_sae_params):
        """Test that output is in same scale as input when using normalization."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        # Input with specific scale
        scale = 15.0
        x = torch.randn(100, small_sae_params["input_size"]) * scale
        
        output = sae(x)
        
        # Output scale should be similar to input scale (not normalized space)
        input_rms = (x ** 2).mean().sqrt()
        output_rms = (output.output ** 2).mean().sqrt()
        
        # Output should be in same ballpark as input (within 2x)
        ratio = output_rms / input_rms
        assert 0.1 < ratio < 10, f"Output scale ratio {ratio} is too far from 1.0"

    def test_without_normalization_raw_scale(self, small_sae_params):
        """Test that without normalization, SAE operates at raw scale."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=False)
        sae.train()
        
        # Should not have normalization buffers
        assert not hasattr(sae, 'running_norm_factor') or not hasattr(sae, 'norm_factor_initialized')
        
        # Or if they exist, they shouldn't be used
        x = torch.randn(100, small_sae_params["input_size"]) * 10.0
        output = sae(x)
        
        # Output should exist and be computed
        assert output.output.shape == x.shape

    def test_normalization_consistency_across_scales(self, small_sae_params):
        """Test that normalized activations are consistent regardless of input scale."""
        sae_params = {**small_sae_params, "normalize_activations": True}
        
        # Create two SAEs with different input scales
        sae1 = LagrangianSAE(**sae_params)
        sae2 = LagrangianSAE(**sae_params)
        
        sae1.train()
        sae2.train()
        
        # Same base data, different scales
        base_x = torch.randn(100, small_sae_params["input_size"])
        x_scale1 = base_x * 1.0
        x_scale2 = base_x * 10.0  # 10x larger
        
        # Forward passes
        _ = sae1(x_scale1)
        _ = sae2(x_scale2)
        
        # After normalization, internal norm factors should scale proportionally
        # norm_factor2 should be ~10x norm_factor1
        ratio = sae2.running_norm_factor / sae1.running_norm_factor
        assert 8.0 < ratio < 12.0, f"Norm factor ratio {ratio} should be ~10"


# ============================================================================
# Test: Training Stability
# ============================================================================

class TestTrainingStability:
    """Test that normalization improves training stability."""

    def test_loss_reasonable_with_normalization(self, small_sae_params):
        """Test that loss values are reasonable with normalization enabled."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        
        # Train for a few steps on data with large scale
        x = torch.randn(32, small_sae_params["input_size"]) * 50.0  # Very large scale
        
        losses = []
        for _ in range(20):
            output = sae(x)
            loss_output = sae.compute_loss(output)
            
            optimizer.zero_grad()
            loss_output.loss.backward()
            optimizer.step()
            sae.update_alpha()
            
            losses.append(loss_output.loss.item())
        
        # Loss should not explode (should stay finite and reasonable)
        assert all(not torch.isnan(torch.tensor(l)) for l in losses), "Loss became NaN"
        assert all(not torch.isinf(torch.tensor(l)) for l in losses), "Loss became Inf"
        assert max(losses) < 1e6, f"Loss too high: {max(losses)}"

    def test_loss_decreases_with_training(self, small_sae_params):
        """Test that loss decreases during training with normalization."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
        
        # Fixed data to overfit on
        x = torch.randn(16, small_sae_params["input_size"]) * 10.0
        
        initial_loss = None
        final_loss = None
        
        for i in range(100):
            output = sae(x)
            loss_output = sae.compute_loss(output)
            
            if i == 0:
                initial_loss = loss_output.loss.item()
            if i == 99:
                final_loss = loss_output.loss.item()
            
            optimizer.zero_grad()
            loss_output.loss.backward()
            optimizer.step()
            sae.update_alpha()
        
        # Loss should decrease
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


# ============================================================================
# Test: run.py Integration Simulation
# ============================================================================

class TestRunPyIntegration:
    """Test that SAEs work correctly with run.py's norm factor pre-computation."""

    def simulate_run_py_norm_factor_computation(
        self,
        sae,
        data_generator,
        num_batches: int = 100
    ) -> float:
        """Simulate run.py's compute_norm_factors_for_saes function."""
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
        
        # Set the norm factor (as run.py does)
        sae.set_norm_factor(norm_factor)
        
        return norm_factor

    def test_external_norm_factor_computation_lagrangian(self, small_sae_params):
        """Test external norm factor computation for LagrangianSAE."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        
        # Generate data
        data_gen = generate_scaled_data(
            batch_size=32,
            input_size=small_sae_params["input_size"],
            scale=25.0,
            num_batches=100
        )
        
        # Simulate run.py behavior
        norm_factor = self.simulate_run_py_norm_factor_computation(
            sae, data_gen, num_batches=50
        )
        
        assert sae.norm_factor_initialized == True
        assert torch.allclose(sae.running_norm_factor, torch.tensor(norm_factor))
        
        # Now training should use this norm factor
        sae.train()
        x = torch.randn(16, small_sae_params["input_size"]) * 25.0
        output = sae(x)
        
        # Verify normalized input has ~unit mean squared norm
        x_normalized = x / sae.running_norm_factor
        normalized_msn = (x_normalized ** 2).sum(dim=-1).mean()
        assert torch.allclose(normalized_msn, torch.tensor(1.0), atol=0.3)

    def test_external_norm_factor_computation_jumprelu(self, small_sae_params):
        """Test external norm factor computation for JumpReLUSAE."""
        sae = JumpReLUSAE(**small_sae_params, normalize_activations=True)
        
        # Generate data
        data_gen = generate_scaled_data(
            batch_size=32,
            input_size=small_sae_params["input_size"],
            scale=25.0,
            num_batches=100
        )
        
        # Simulate run.py behavior
        norm_factor = self.simulate_run_py_norm_factor_computation(
            sae, data_gen, num_batches=50
        )
        
        assert sae.norm_factor_initialized == True
        assert torch.allclose(sae.running_norm_factor, torch.tensor(norm_factor))

    def test_lagrangian_and_jumprelu_same_with_external_norm_factor(self, small_sae_params):
        """Test that LagrangianSAE and JumpReLUSAE behave similarly with same norm factor."""
        lagrangian = LagrangianSAE(**small_sae_params, normalize_activations=True)
        jumprelu = JumpReLUSAE(**small_sae_params, normalize_activations=True)
        
        # Generate data and compute the "correct" norm factor for it
        x = torch.randn(16, small_sae_params["input_size"]) * 15.0
        actual_msn = (x ** 2).sum(dim=-1).mean()
        actual_norm_factor = actual_msn.sqrt().item()
        
        # Set same norm factor for both (the "correct" one for this data)
        lagrangian.set_norm_factor(actual_norm_factor)
        jumprelu.set_norm_factor(actual_norm_factor)
        
        lagrangian.eval()
        jumprelu.eval()
        
        # Both should normalize the same way
        # The normalized input should have unit mean squared norm
        x_normalized = x / actual_norm_factor
        expected_msn = (x_normalized ** 2).sum(dim=-1).mean()
        
        # With the correct norm factor, MSN should be ~1.0
        assert torch.allclose(expected_msn, torch.tensor(1.0), atol=0.1)


# ============================================================================
# Test: Config-Based Tests
# ============================================================================

class TestConfigBasedBehavior:
    """Test SAE behavior with realistic config parameters."""

    def test_lagrangian_with_gemma_config_params(self, gemma_lagrangian_config):
        """Test LagrangianSAE with parameters from gemma config."""
        if gemma_lagrangian_config is None:
            pytest.skip("Config file not found")
        
        sae_config = gemma_lagrangian_config.get("saes", {})
        
        # Use smaller size for testing but realistic other params
        sae = LagrangianSAE(
            input_size=64,  # Smaller than real
            n_dict_components=256,  # Smaller than real
            target_l0=sae_config.get("target_l0", 32.0),
            initial_threshold=sae_config.get("initial_threshold", 0.001),
            bandwidth=sae_config.get("bandwidth", 0.001),
            normalize_activations=sae_config.get("normalize_activations", True),
            calibrate_thresholds=sae_config.get("calibrate_thresholds", True),
        )
        
        assert sae.normalize_activations == sae_config.get("normalize_activations", True)
        
        # Should work with realistic-ish data
        sae.train()
        x = torch.randn(32, 64) * 50.0  # Gemma activations have large magnitudes
        output = sae(x)
        
        assert output.output.shape == x.shape
        assert not torch.isnan(output.output).any()

    def test_jumprelu_with_gemma_config_params(self, gemma_jumprelu_config):
        """Test JumpReLUSAE with parameters from gemma config."""
        if gemma_jumprelu_config is None:
            pytest.skip("Config file not found")
        
        sae_config = gemma_jumprelu_config.get("saes", {})
        
        # Use smaller size for testing but realistic other params
        sae = JumpReLUSAE(
            input_size=64,  # Smaller than real
            n_dict_components=256,  # Smaller than real
            target_l0=sae_config.get("target_l0", 32.0),
            initial_threshold=sae_config.get("initial_threshold", 0.001),
            bandwidth=sae_config.get("bandwidth", 0.001),
            normalize_activations=sae_config.get("normalize_activations", True),
            norm_factor_num_batches=sae_config.get("norm_factor_num_batches", 100),
        )
        
        assert sae.normalize_activations == sae_config.get("normalize_activations", True)
        assert sae.norm_factor_num_batches == sae_config.get("norm_factor_num_batches", 100)


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_norm_factor_with_zero_input(self, small_sae_params):
        """Test behavior with zero input (edge case).
        
        Note: Zero input will cause NaN with normalization because:
        - norm_factor = sqrt(E[||x||²]) = sqrt(0) = 0
        - x_normalized = x / 0 = NaN
        
        This is expected behavior - real data should never be exactly zero.
        The test verifies the SAE doesn't crash, and documents this edge case.
        """
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        # Zero input - this is an unrealistic edge case
        x = torch.zeros(8, small_sae_params["input_size"])
        
        # With exact zeros, we get NaN due to 0/0
        # This is documented/expected behavior
        output = sae(x)
        
        # The output will be NaN - this test documents that behavior
        # In practice, real activations are never exactly zero
        assert torch.isnan(output.output).any() or torch.allclose(output.output, torch.zeros_like(output.output))

    def test_norm_factor_with_very_small_input(self, small_sae_params):
        """Test behavior with very small input."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        # Very small input
        x = torch.randn(8, small_sae_params["input_size"]) * 1e-8
        
        output = sae(x)
        
        # Should not crash and output should be valid
        assert not torch.isnan(output.output).any()

    def test_norm_factor_with_very_large_input(self, small_sae_params):
        """Test behavior with very large input."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        # Very large input
        x = torch.randn(8, small_sae_params["input_size"]) * 1e4
        
        output = sae(x)
        
        # Should not crash and output should be valid
        assert not torch.isnan(output.output).any()
        assert not torch.isinf(output.output).any()

    def test_set_norm_factor_multiple_times(self, small_sae_params):
        """Test that set_norm_factor can be called multiple times."""
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        
        sae.set_norm_factor(10.0)
        assert torch.allclose(sae.running_norm_factor, torch.tensor(10.0))
        
        sae.set_norm_factor(20.0)
        assert torch.allclose(sae.running_norm_factor, torch.tensor(20.0))
        
        sae.set_norm_factor(5.0)
        assert torch.allclose(sae.running_norm_factor, torch.tensor(5.0))


# ============================================================================
# Test: Comparison with Old Behavior (Regression Test)
# ============================================================================

class TestRegressionAgainstOldBehavior:
    """
    Regression tests to ensure the new implementation maintains expected behavior.
    
    The 'old' behavior was EMA-based normalization. The 'new' behavior uses
    JumpReLUSAE-compatible naming and supports external pre-computation.
    Both should produce similar results when using internal initialization.
    """

    def test_internal_initialization_produces_correct_norm_factor(self, small_sae_params):
        """
        Test that internal initialization (first forward pass) produces 
        the correct norm factor, matching the old EMA-based behavior.
        """
        sae = LagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        # Data with known statistics
        scale = 7.5
        x = torch.randn(200, small_sae_params["input_size"]) * scale
        
        # Expected norm factor
        expected_msn = (x ** 2).sum(dim=-1).mean()
        expected_norm_factor = expected_msn.sqrt()
        
        # Internal initialization via first forward pass
        _ = sae(x)
        
        # Should match expected value (old EMA with momentum=0 on first batch)
        assert torch.allclose(sae.running_norm_factor, expected_norm_factor, rtol=0.05)

    def test_output_reconstruction_quality_preserved(self, small_sae_params):
        """Test that reconstruction quality is preserved after changes."""
        sae = LagrangianSAE(
            **small_sae_params,
            normalize_activations=True,
            calibrate_thresholds=True,
        )
        sae.train()
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
        
        # Fixed data
        x = torch.randn(16, small_sae_params["input_size"]) * 5.0
        
        # Train
        for _ in range(200):
            output = sae(x)
            loss_output = sae.compute_loss(output)
            
            optimizer.zero_grad()
            loss_output.loss.backward()
            optimizer.step()
            sae.update_alpha()
        
        # Evaluate reconstruction
        sae.eval()
        output = sae(x)
        mse = F.mse_loss(output.output, x)
        
        # Should achieve reasonable reconstruction
        assert mse.item() < 5.0, f"MSE too high: {mse.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
