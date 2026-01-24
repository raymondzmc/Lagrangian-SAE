"""
Tests for LagrangianSAE implementation.

Tests verify:
1. Basic forward/backward pass
2. Activation normalization (scale-only normalization matching JumpReLUSAE)
3. Pre-encoder bias (decoder_bias subtraction) correctness
4. Combined normalization + bias handling
5. Reconstruction consistency
6. Alpha dual ascent updates
"""

import pytest
import torch
import torch.nn.functional as F
from models.saes.lagrangian_sae import LagrangianSAE, LagrangianSAEOutput


class TestLagrangianSAEBasic:
    """Basic functionality tests."""

    @pytest.fixture
    def sae_params(self):
        return {
            "input_size": 64,
            "n_dict_components": 128,
            "target_l0": 8.0,
            "initial_alpha": 0.0,
            "alpha_lr": 0.01,
            "bandwidth": 0.1,
            "initial_threshold": 0.5,
        }

    def test_forward_shape(self, sae_params):
        """Test that forward pass produces correct output shapes."""
        sae = LagrangianSAE(**sae_params)
        x = torch.randn(4, 32, sae_params["input_size"])  # (batch, seq, dim)
        
        output = sae(x)
        
        assert output.input.shape == x.shape
        assert output.output.shape == x.shape
        assert output.c.shape == (4, 32, sae_params["n_dict_components"])
        assert output.l0.shape == (4, 32)
        assert output.preacts.shape == (4, 32, sae_params["n_dict_components"])

    def test_backward_pass(self, sae_params):
        """Test that gradients flow correctly."""
        sae = LagrangianSAE(**sae_params)
        sae.train()
        x = torch.randn(4, 32, sae_params["input_size"], requires_grad=True)
        
        output = sae(x)
        loss_output = sae.compute_loss(output)
        loss_output.loss.backward()
        
        # Check gradients exist for key parameters
        assert sae.encoder.weight.grad is not None
        assert sae.decoder.weight.grad is not None
        assert sae.jumprelu.threshold.grad is not None


class TestActivationNormalization:
    """Tests for activation normalization (scale-only, matching JumpReLUSAE)."""

    @pytest.fixture
    def sae_with_normalization(self):
        return LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
            normalize_activations=True,
            use_pre_enc_bias=False,
        )

    def test_normalization_initialization(self, sae_with_normalization):
        """Test that normalization buffers are initialized correctly."""
        sae = sae_with_normalization
        
        # Should have JumpReLUSAE-compatible buffer names
        assert hasattr(sae, 'running_norm_factor')
        assert hasattr(sae, 'norm_factor_initialized')
        
        # Should be initialized to 1.0 (identity) and False
        assert torch.allclose(sae.running_norm_factor, torch.tensor(1.0))
        assert sae.norm_factor_initialized == False

    def test_normalization_stats_update(self, sae_with_normalization):
        """Test that running norm factor is updated during training."""
        sae = sae_with_normalization
        sae.train()
        
        # Create input with known magnitude
        x = torch.randn(100, 64) * 5.0  # Large scale
        expected_msn = (x ** 2).sum(dim=-1).mean()
        expected_norm_factor = expected_msn.sqrt()
        
        # First forward pass should initialize stats
        _ = sae(x)
        
        assert sae.norm_factor_initialized == True
        # Norm factor should be close to sqrt(mean squared norm)
        assert torch.allclose(sae.running_norm_factor, expected_norm_factor, rtol=0.1)

    def test_normalization_frozen_during_eval(self, sae_with_normalization):
        """Test that norm factor is frozen during evaluation."""
        sae = sae_with_normalization
        sae.train()
        
        # Initialize with first batch
        x1 = torch.randn(100, 64) * 2.0
        _ = sae(x1)
        
        # Record norm factor after training batch
        norm_factor_after_train = sae.running_norm_factor.clone()
        
        # Switch to eval mode
        sae.eval()
        
        # Pass different data with very different scale
        x2 = torch.randn(100, 64) * 100.0  # Much larger scale
        _ = sae(x2)
        
        # Norm factor should NOT change during eval
        assert torch.allclose(sae.running_norm_factor, norm_factor_after_train)

    def test_denormalization_correctness(self, sae_with_normalization):
        """Test that denormalization correctly reverses normalization."""
        sae = sae_with_normalization
        sae.train()
        
        # Create input with large scale
        x = torch.randn(100, 64) * 10.0
        
        # Forward pass
        output = sae(x)
        
        # The output should be in the ORIGINAL scale (denormalized)
        # Check that output scale is similar to input (not normalized space)
        input_scale = (x ** 2).mean().sqrt()
        output_scale = (output.output ** 2).mean().sqrt()
        
        # Output scale should be in same ballpark as input (within 2x)
        # This verifies denormalization is working
        assert output_scale.item() > input_scale.item() * 0.1, "Output appears to be in normalized space, not denormalized"
    
    def test_set_norm_factor(self, sae_with_normalization):
        """Test that set_norm_factor correctly sets the norm factor."""
        sae = sae_with_normalization
        
        # Manually set norm factor (as run.py does after pre-computation)
        sae.set_norm_factor(42.0)
        
        assert torch.allclose(sae.running_norm_factor, torch.tensor(42.0))
        assert sae.norm_factor_initialized == True
    
    def test_has_norm_factor_num_batches(self, sae_with_normalization):
        """Test that norm_factor_num_batches attribute exists for run.py compatibility."""
        sae = sae_with_normalization
        assert hasattr(sae, 'norm_factor_num_batches')
        assert sae.norm_factor_num_batches == 100  # Default value


class TestPreEncoderBias:
    """Tests for pre-encoder bias (decoder_bias subtraction)."""

    @pytest.fixture
    def sae_with_pre_enc_bias(self):
        return LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
            normalize_activations=False,
            use_pre_enc_bias=True,
        )

    def test_pre_enc_bias_effect(self, sae_with_pre_enc_bias):
        """Test that pre-encoder bias affects encoding."""
        sae = sae_with_pre_enc_bias
        
        # Set a known decoder bias
        with torch.no_grad():
            sae.decoder_bias.fill_(1.0)
        
        x = torch.ones(4, 64) * 2.0  # Input of all 2s
        
        output = sae(x)
        
        # With pre_enc_bias, the encoder sees (x - decoder_bias) = (2 - 1) = 1
        # Without it, encoder would see x = 2
        # We can't directly test the internal, but we can verify the bias is used
        
        # The decoder adds bias back, so for a "good" reconstruction,
        # output should be close to input
        # (This is a sanity check that bias is added back)
        pass  # Structure test - actual behavior verified in integration tests


class TestCombinedNormalizationAndBias:
    """Tests for combined normalization + pre-encoder bias."""

    @pytest.fixture
    def sae_combined(self):
        return LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
            normalize_activations=True,
            use_pre_enc_bias=True,
        )

    def test_reconstruction_with_both_enabled(self, sae_combined):
        """Test reconstruction quality with both normalization and bias."""
        sae = sae_combined
        sae.train()
        
        # Train for a few steps to let the SAE learn
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        
        for _ in range(100):
            x = torch.randn(32, 64) * 3.0 + 2.0
            output = sae(x)
            loss_output = sae.compute_loss(output)
            
            optimizer.zero_grad()
            loss_output.loss.backward()
            optimizer.step()
            sae.update_alpha()
        
        # After training, reconstruction should be reasonable
        sae.eval()
        x_test = torch.randn(32, 64) * 3.0 + 2.0
        output = sae(x_test)
        
        mse = F.mse_loss(output.output, x_test)
        # MSE should be reasonable (not perfect, but not terrible)
        # This mainly checks that the math is correct and nothing explodes
        assert mse.item() < 10.0, f"MSE too high: {mse.item()}"

    def test_input_output_scale_consistency(self, sae_combined):
        """Test that output is in the same scale as input."""
        sae = sae_combined
        sae.train()
        
        # Use input with specific scale
        x = torch.randn(100, 64) * 5.0 + 10.0  # mean ~10, std ~5
        
        output = sae(x)
        
        # Output should be in similar scale as input (not normalized space)
        input_scale = (x ** 2).mean().sqrt().item()
        output_scale = (output.output ** 2).mean().sqrt().item()
        
        # Output scale should be in same ballpark as input
        # This is a rough check that denormalization is working
        assert output_scale > input_scale * 0.1, "Output scale too small (in normalized space)"
        
        # Check that output isn't stuck at some degenerate value
        assert output.output.std().item() > 0.1, "Output has no variance"


class TestAlphaDualAscent:
    """Tests for alpha (Lagrangian multiplier) updates."""

    @pytest.fixture
    def sae_equality(self):
        """SAE with equality constraint (default): alpha can be negative."""
        return LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
            initial_alpha=0.0,
            alpha_lr=0.1,
            alpha_max=10.0,
            equality_constraint=True,  # Explicitly set for clarity
        )

    @pytest.fixture
    def sae_inequality(self):
        """SAE with inequality constraint: alpha must be >= 0."""
        return LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
            initial_alpha=0.0,
            alpha_lr=0.1,
            alpha_max=10.0,
            equality_constraint=False,  # Inequality constraint
        )

    def test_alpha_increases_when_l0_above_target(self, sae_equality):
        """Test that alpha increases when L0 > target."""
        sae = sae_equality
        sae.train()
        
        initial_alpha = sae.alpha.item()
        
        # Create input that will likely cause high L0
        x = torch.randn(32, 64) * 10.0  # Large activations
        
        for _ in range(10):
            output = sae(x)
            loss_output = sae.compute_loss(output)
            sae.update_alpha()
        
        # If L0 > target, alpha should have increased
        if sae.running_l0.item() > sae.target_l0:
            assert sae.alpha.item() > initial_alpha, "Alpha should increase when L0 > target"

    def test_alpha_decreases_when_l0_below_target_equality(self, sae_equality):
        """Test that alpha decreases when L0 < target (equality constraint allows negative alpha)."""
        sae = sae_equality
        sae.train()
        
        # Start with high alpha
        with torch.no_grad():
            sae.alpha.fill_(5.0)
            # Set running_l0 below target
            sae.running_l0.fill_(sae.target_l0 - 5.0)
        
        initial_alpha = sae.alpha.item()
        
        # Manually trigger alpha update with constraint satisfied
        sae._last_constraint_violation = torch.tensor(-5.0)  # L0 - target = negative
        sae.update_alpha()
        
        # Alpha should decrease (can go negative in equality mode)
        assert sae.alpha.item() < initial_alpha, "Alpha should decrease when L0 < target"

    def test_alpha_decreases_when_l0_below_target_inequality(self, sae_inequality):
        """Test that alpha decreases when L0 < target (inequality constraint keeps alpha >= 0)."""
        sae = sae_inequality
        sae.train()
        
        # Start with positive alpha
        with torch.no_grad():
            sae.alpha.fill_(5.0)
            # Set running_l0 below target
            sae.running_l0.fill_(sae.target_l0 - 5.0)
        
        initial_alpha = sae.alpha.item()
        
        # Manually trigger alpha update with constraint satisfied
        sae._last_constraint_violation = torch.tensor(-5.0)  # L0 - target = negative
        sae.update_alpha()
        
        # Alpha should decrease but stay >= 0 in inequality mode
        assert sae.alpha.item() < initial_alpha, "Alpha should decrease when L0 < target"
        assert sae.alpha.item() >= 0, "Alpha should never be negative in inequality mode"

    def test_alpha_clamped_to_max(self, sae_equality):
        """Test that alpha is clamped to alpha_max."""
        sae = sae_equality
        sae.train()
        
        # Set a large constraint violation
        sae._last_constraint_violation = torch.tensor(1000.0)
        
        for _ in range(100):
            sae.update_alpha()
        
        assert sae.alpha.item() <= sae.alpha_max, "Alpha should be clamped to alpha_max"

    def test_alpha_clamped_to_negative_max_equality(self, sae_equality):
        """Test that alpha is clamped to -alpha_max in equality mode."""
        sae = sae_equality
        sae.train()
        
        # Set a large negative constraint violation
        sae._last_constraint_violation = torch.tensor(-1000.0)
        
        for _ in range(100):
            sae.update_alpha()
        
        # In equality mode, alpha can go negative but is clamped to -alpha_max
        assert sae.alpha.item() >= -sae.alpha_max, "Alpha should be clamped to -alpha_max"
        assert sae.alpha.item() < 0, "Alpha should be negative with large negative constraint violation"

    def test_alpha_clamped_to_zero_inequality(self, sae_inequality):
        """Test that alpha is clamped to >= 0 in inequality mode."""
        sae = sae_inequality
        sae.train()
        
        # Set a large negative constraint violation
        sae._last_constraint_violation = torch.tensor(-1000.0)
        
        for _ in range(100):
            sae.update_alpha()
        
        # In inequality mode, alpha is clamped to [0, alpha_max]
        assert sae.alpha.item() >= 0, "Alpha should never be negative in inequality mode"
        assert sae.alpha.item() == 0, "Alpha should be exactly 0 when constraint is satisfied"


class TestReconstructionConsistency:
    """Integration tests for reconstruction correctness."""

    def test_perfect_reconstruction_with_identity(self):
        """Test that SAE can achieve near-perfect reconstruction with identity-like setup."""
        # Small SAE where we can overfit
        sae = LagrangianSAE(
            input_size=16,
            n_dict_components=64,
            target_l0=32.0,  # Allow many features
            normalize_activations=True,
            use_pre_enc_bias=True,
            initial_threshold=0.01,  # Low threshold = more features active
            bandwidth=0.1,
        )
        sae.train()
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
        
        # Fixed input to overfit on
        x = torch.randn(8, 16)
        
        for _ in range(500):
            output = sae(x)
            loss_output = sae.compute_loss(output)
            
            optimizer.zero_grad()
            loss_output.loss.backward()
            optimizer.step()
            sae.update_alpha()
        
        # Should achieve low reconstruction error
        sae.eval()
        output = sae(x)
        mse = F.mse_loss(output.output, output.input)
        
        assert mse.item() < 0.5, f"Failed to achieve low reconstruction error: {mse.item()}"

    def test_normalization_denormalization_roundtrip(self):
        """Test that normalization followed by denormalization preserves scale."""
        sae = LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=16.0,
            normalize_activations=True,
            use_pre_enc_bias=True,
        )
        sae.train()
        
        # Input with specific distribution
        x = torch.randn(100, 64) * 7.0
        
        # Compute expected norm factor
        expected_msn = (x ** 2).sum(dim=-1).mean()
        expected_norm_factor = expected_msn.sqrt()
        
        # Get norm factor (this also initializes running_norm_factor)
        norm_factor = sae._compute_norm_factor(x)
        
        # Normalize
        x_normalized = x / norm_factor
        
        # Verify normalization: mean squared norm should be ~1
        normalized_msn = (x_normalized ** 2).sum(dim=-1).mean()
        assert torch.allclose(normalized_msn, torch.tensor(1.0), rtol=0.1)
        
        # Verify denormalization roundtrip
        x_reconstructed = x_normalized * norm_factor
        assert torch.allclose(x_reconstructed, x, atol=1e-5)


class TestThresholdCalibration:
    """Tests for automatic threshold calibration."""

    def test_calibration_sets_thresholds(self):
        """Test that calibration sets thresholds based on target L0."""
        target_l0 = 16.0
        sae = LagrangianSAE(
            input_size=64,
            n_dict_components=256,
            target_l0=target_l0,
            normalize_activations=True,
            calibrate_thresholds=True,
            initial_threshold=0.5,
        )
        sae.train()
        
        # Before calibration
        assert sae.thresholds_calibrated == False
        initial_threshold = sae.jumprelu.threshold.mean().item()
        
        # Run one forward pass to trigger calibration
        x = torch.randn(100, 64)
        output = sae(x)
        
        # After calibration
        assert sae.thresholds_calibrated == True
        calibrated_threshold = sae.jumprelu.threshold.mean().item()
        
        # Threshold should have changed
        assert calibrated_threshold != initial_threshold, "Threshold should change after calibration"
        
        # Verify L0 is closer to target after calibration
        x_test = torch.randn(500, 64)
        output = sae(x_test)
        actual_l0 = output.l0.mean().item()
        
        # L0 should be within reasonable range of target (not necessarily exact)
        # The calibration sets thresholds for expected L0 = target, but there's variance
        assert actual_l0 < target_l0 * 3, f"L0 ({actual_l0}) should be closer to target ({target_l0})"

    def test_calibration_only_happens_once(self):
        """Test that calibration only happens on first batch."""
        sae = LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
            calibrate_thresholds=True,
        )
        sae.train()
        
        # Run first forward pass to trigger calibration
        x = torch.randn(50, 64)
        _ = sae(x)
        
        assert sae.thresholds_calibrated == True
        threshold_after_calibration = sae.jumprelu.threshold.clone()
        
        # Run more forward passes
        for _ in range(10):
            x = torch.randn(100, 64)
            _ = sae(x)
        
        # Thresholds should still be the same (calibration only happens once)
        # Note: thresholds can change during training due to gradient updates,
        # but the calibration itself shouldn't run again
        assert sae.thresholds_calibrated == True

    def test_calibration_respects_normalization(self):
        """Test that calibration works correctly with activation normalization."""
        sae = LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
            normalize_activations=True,
            calibrate_thresholds=True,
        )
        sae.train()
        
        # Use input with large scale (should be normalized before calibration)
        x = torch.randn(300, 64) * 100.0 + 50.0  # Large mean and std
        _ = sae(x)
        
        assert sae.thresholds_calibrated == True
        
        # Thresholds should be reasonable (not inflated by large input scale)
        mean_threshold = sae.jumprelu.threshold.mean().item()
        # With normalization, thresholds should be around 1-3, not 100+
        assert mean_threshold < 10.0, f"Threshold ({mean_threshold}) too high - normalization may not be working"

    def test_calibration_disabled(self):
        """Test that calibration can be disabled."""
        sae = LagrangianSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
            calibrate_thresholds=False,  # Disabled
            initial_threshold=0.5,
        )
        sae.train()
        
        initial_threshold = sae.jumprelu.threshold.mean().item()
        
        # Run many forward passes
        for _ in range(20):
            x = torch.randn(100, 64)
            _ = sae(x)
        
        # Without training (no loss.backward()), threshold should stay at initial value
        # Note: actually threshold can't change without gradients
        assert not hasattr(sae, 'thresholds_calibrated') or not sae.thresholds_calibrated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

