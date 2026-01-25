"""
Comprehensive tests for MatryoshkaLagrangianSAE implementation.

This test suite verifies that MatryoshkaLagrangianSAE:
1. Satisfies all MatryoshkaSAE structural requirements (groups, progressive loss)
2. Follows LagrangianSAE training procedure (alpha, L0, thresholds, normalization)
3. Has consistent API with both parent implementations
4. Trains correctly with combined loss function

Test Categories:
- MatryoshkaSAE Requirements: Group structure, progressive loss, group weights
- LagrangianSAE Requirements: Alpha dynamics, L0 tracking, threshold calibration, normalization
- API Consistency: Required attributes and methods
- Integration: End-to-end training behavior
- Caveats Documentation: Tests that document known differences from parent implementations
"""

import pytest
import torch
import torch.nn.functional as F
from math import isclose

from models.saes.matryoshka_lagrangian_sae import (
    MatryoshkaLagrangianSAE,
    MatryoshkaLagrangianSAEConfig,
    MatryoshkaLagrangianSAEOutput,
)
from models.saes.matryoshka_sae import MatryoshkaSAE
from models.saes.lagrangian_sae import LagrangianSAE


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_sae_params():
    """Small SAE parameters for fast testing."""
    return {
        "input_size": 64,
        "n_dict_components": 128,
        "target_l0": 8.0,
        "initial_threshold": 0.1,
        "bandwidth": 0.1,
        "init_decoder_orthogonal": False,  # Avoid CUDA for CPU tests
    }


@pytest.fixture
def matryoshka_lagrangian_sae(small_sae_params):
    """Create a MatryoshkaLagrangianSAE instance."""
    return MatryoshkaLagrangianSAE(**small_sae_params)


@pytest.fixture
def matryoshka_sae():
    """Create a MatryoshkaSAE instance for comparison."""
    return MatryoshkaSAE(
        input_size=64,
        n_dict_components=128,
        k=8,
        init_decoder_orthogonal=False,  # Avoid CUDA for CPU tests
    )


@pytest.fixture
def lagrangian_sae(small_sae_params):
    """Create a LagrangianSAE instance for comparison."""
    return LagrangianSAE(**small_sae_params)


# ============================================================================
# Test: MatryoshkaSAE Structural Requirements
# ============================================================================

class TestMatryoshkaStructuralRequirements:
    """Tests that MatryoshkaLagrangianSAE satisfies MatryoshkaSAE structural requirements."""

    def test_has_group_structure_attributes(self, matryoshka_lagrangian_sae):
        """Test that all group structure attributes exist."""
        sae = matryoshka_lagrangian_sae
        
        assert hasattr(sae, 'num_groups')
        assert hasattr(sae, 'group_sizes')
        assert hasattr(sae, 'group_boundaries')
        assert hasattr(sae, 'group_weights')
        
    def test_group_fractions_default(self, small_sae_params):
        """Test default group fractions are set correctly."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params)
        
        # Default: [1/32, 1/16, 1/8, 1/4, 1/2 + 1/32]
        assert sae.num_groups == 5
        
    def test_custom_group_fractions(self, small_sae_params):
        """Test custom group fractions work correctly."""
        custom_fractions = [0.25, 0.25, 0.5]  # 3 groups
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            group_fractions=custom_fractions,
        )
        
        assert sae.num_groups == 3
        # Check that group sizes sum to n_dict_components
        total_size = sum(sae.group_sizes.tolist())
        assert total_size == small_sae_params["n_dict_components"]
        
    def test_group_fractions_must_sum_to_one(self, small_sae_params):
        """Test that group fractions must sum to 1.0."""
        with pytest.raises(AssertionError):
            MatryoshkaLagrangianSAE(
                **small_sae_params,
                group_fractions=[0.3, 0.3, 0.3],  # Sums to 0.9, not 1.0
            )
            
    def test_group_boundaries_computation(self, small_sae_params):
        """Test that group boundaries are computed correctly."""
        fractions = [0.25, 0.25, 0.5]
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            group_fractions=fractions,
        )
        
        n = small_sae_params["n_dict_components"]
        boundaries = sae.group_boundaries.tolist()
        
        # Boundaries should be cumulative: [0, 32, 64, 128]
        assert boundaries[0] == 0
        assert boundaries[-1] == n
        # Should be monotonically increasing
        for i in range(len(boundaries) - 1):
            assert boundaries[i] < boundaries[i + 1]
            
    def test_group_weights_default_uniform(self, small_sae_params):
        """Test that default group weights are uniform."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params)
        
        expected_weight = 1.0 / sae.num_groups
        for w in sae.group_weights.tolist():
            assert isclose(w, expected_weight, rel_tol=1e-5)
            
    def test_custom_group_weights(self, small_sae_params):
        """Test custom group weights work correctly."""
        fractions = [0.25, 0.25, 0.5]
        weights = [0.5, 0.3, 0.2]
        
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            group_fractions=fractions,
            group_weights=weights,
        )
        
        for i, w in enumerate(sae.group_weights.tolist()):
            assert isclose(w, weights[i], rel_tol=1e-5)
            
    def test_group_weights_length_must_match_fractions(self, small_sae_params):
        """Test that group_weights length must match group_fractions."""
        with pytest.raises(AssertionError):
            MatryoshkaLagrangianSAE(
                **small_sae_params,
                group_fractions=[0.5, 0.5],  # 2 groups
                group_weights=[0.25, 0.25, 0.25, 0.25],  # 4 weights
            )


class TestProgressiveLossComputation:
    """Tests for progressive reconstruction loss at each group boundary."""
    
    def test_loss_dict_contains_group_losses(self, matryoshka_lagrangian_sae):
        """Test that loss_dict contains individual group losses."""
        sae = matryoshka_lagrangian_sae
        sae.train()
        
        x = torch.randn(8, 64)
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        # Should have mse_loss_group_i for each group
        for i in range(sae.num_groups):
            assert f"mse_loss_group_{i}" in loss_output.loss_dict
            
    def test_progressive_reconstruction(self, small_sae_params):
        """Test that reconstruction is computed progressively."""
        fractions = [0.5, 0.5]  # Two equal groups
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            group_fractions=fractions,
        )
        sae.eval()
        
        x = torch.randn(8, 64)
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        # Second group (full reconstruction) should have lower or equal MSE
        # than first group (partial reconstruction) - not always true but
        # generally expected with trained models
        mse_group_0 = loss_output.loss_dict["mse_loss_group_0"].item()
        mse_group_1 = loss_output.loss_dict["mse_loss_group_1"].item()
        
        # Both losses should be finite
        assert not torch.isnan(torch.tensor(mse_group_0))
        assert not torch.isnan(torch.tensor(mse_group_1))
        
    def test_progressive_loss_uses_decoder_bias(self, small_sae_params):
        """Test that progressive reconstruction includes decoder bias."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params)
        sae.train()
        
        # Set a non-zero decoder bias
        with torch.no_grad():
            sae.decoder_bias.fill_(1.0)
        
        x = torch.randn(8, 64)
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        # Loss should be finite (bias is being used correctly)
        assert not torch.isnan(loss_output.loss)
        assert loss_output.loss.item() > 0
        
    def test_total_mse_is_weighted_sum(self, small_sae_params):
        """Test that total MSE loss is weighted sum of group losses."""
        fractions = [0.5, 0.5]
        weights = [0.3, 0.7]
        
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            group_fractions=fractions,
            group_weights=weights,
            rho_quadratic=0.0,  # Disable quadratic penalty for cleaner test
        )
        sae.train()
        
        x = torch.randn(8, 64)
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        # Compute expected weighted MSE
        group_0_mse = loss_output.loss_dict["mse_loss_group_0"]
        group_1_mse = loss_output.loss_dict["mse_loss_group_1"]
        expected_mse = weights[0] * group_0_mse + weights[1] * group_1_mse
        
        # Total MSE (before Lagrangian terms) should match
        actual_mse = loss_output.loss_dict["mse_loss"]
        assert torch.allclose(actual_mse, expected_mse, rtol=1e-4)


# ============================================================================
# Test: LagrangianSAE Training Procedure Requirements
# ============================================================================

class TestLagrangianTrainingProcedure:
    """Tests that MatryoshkaLagrangianSAE follows LagrangianSAE training procedure."""
    
    def test_has_alpha_buffer(self, matryoshka_lagrangian_sae):
        """Test that alpha (Lagrangian multiplier) buffer exists."""
        sae = matryoshka_lagrangian_sae
        assert hasattr(sae, 'alpha')
        assert isinstance(sae.alpha, torch.Tensor)
        
    def test_has_running_l0_buffer(self, matryoshka_lagrangian_sae):
        """Test that running L0 EMA buffer exists."""
        sae = matryoshka_lagrangian_sae
        assert hasattr(sae, 'running_l0')
        assert isinstance(sae.running_l0, torch.Tensor)
        
    def test_has_update_alpha_method(self, matryoshka_lagrangian_sae):
        """Test that update_alpha method exists."""
        sae = matryoshka_lagrangian_sae
        assert hasattr(sae, 'update_alpha')
        assert callable(sae.update_alpha)
        
    def test_alpha_dual_ascent_increases_when_l0_above_target(self, small_sae_params):
        """Test that alpha increases when L0 > target."""
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            initial_alpha=0.0,
            alpha_lr=0.1,
        )
        sae.train()
        
        initial_alpha = sae.alpha.item()
        
        # Manually set constraint violation (L0 > target)
        sae._last_constraint_violation = torch.tensor(5.0)  # L0 - target > 0
        sae.update_alpha()
        
        assert sae.alpha.item() > initial_alpha
        
    def test_alpha_dual_ascent_decreases_when_l0_below_target(self, small_sae_params):
        """Test that alpha decreases when L0 < target (equality constraint)."""
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            initial_alpha=5.0,
            alpha_lr=0.1,
            equality_constraint=True,
        )
        sae.train()
        
        initial_alpha = sae.alpha.item()
        
        # Manually set constraint violation (L0 < target)
        sae._last_constraint_violation = torch.tensor(-5.0)  # L0 - target < 0
        sae.update_alpha()
        
        assert sae.alpha.item() < initial_alpha
        
    def test_alpha_clamped_to_bounds(self, small_sae_params):
        """Test that alpha is clamped to [−alpha_min, alpha_max]."""
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            initial_alpha=0.0,
            alpha_lr=1.0,
            alpha_max=10.0,
            alpha_min=5.0,  # Asymmetric bounds
            equality_constraint=True,
        )
        
        # Large positive violation
        sae._last_constraint_violation = torch.tensor(100.0)
        for _ in range(100):
            sae.update_alpha()
        assert sae.alpha.item() <= 10.0
        
        # Large negative violation
        sae._last_constraint_violation = torch.tensor(-100.0)
        for _ in range(100):
            sae.update_alpha()
        assert sae.alpha.item() >= -5.0
        
    def test_inequality_constraint_alpha_non_negative(self, small_sae_params):
        """Test that alpha >= 0 with inequality constraint."""
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            initial_alpha=1.0,
            alpha_lr=1.0,
            equality_constraint=False,  # Inequality constraint
        )
        
        # Large negative violation
        sae._last_constraint_violation = torch.tensor(-100.0)
        for _ in range(100):
            sae.update_alpha()
        assert sae.alpha.item() >= 0
        
    def test_running_l0_ema_update(self, small_sae_params):
        """Test that running L0 is updated via EMA during training."""
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            l0_ema_momentum=0.9,
        )
        sae.train()
        
        initial_running_l0 = sae.running_l0.item()
        
        # Forward + compute_loss updates running_l0
        x = torch.randn(100, 64)
        output = sae(x)
        _ = sae.compute_loss(output)
        
        # running_l0 should have changed
        # (unless by coincidence the L0 exactly equals target)
        assert sae.running_l0.item() != initial_running_l0 or \
               abs(output.l0.mean().item() - initial_running_l0) < 0.01
               

class TestThresholdCalibration:
    """Tests for automatic threshold calibration."""
    
    def test_has_calibrate_thresholds_flag(self, matryoshka_lagrangian_sae):
        """Test that calibrate_thresholds attribute exists."""
        assert hasattr(matryoshka_lagrangian_sae, 'calibrate_thresholds')
        
    def test_thresholds_calibrated_buffer(self, small_sae_params):
        """Test that thresholds_calibrated buffer exists."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, calibrate_thresholds=True)
        assert hasattr(sae, 'thresholds_calibrated')
        assert sae.thresholds_calibrated == False
        
    def test_calibration_runs_on_first_forward(self, small_sae_params):
        """Test that calibration runs on first forward pass in training mode."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, calibrate_thresholds=True)
        sae.train()
        
        assert sae.thresholds_calibrated == False
        
        x = torch.randn(100, 64)
        _ = sae(x)
        
        assert sae.thresholds_calibrated == True
        
    def test_calibration_only_runs_once(self, small_sae_params):
        """Test that calibration only runs once."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, calibrate_thresholds=True)
        sae.train()
        
        # First forward - calibrate
        x = torch.randn(100, 64)
        _ = sae(x)
        
        threshold_after_calibration = sae.jumprelu.threshold.clone()
        
        # Second forward - should not recalibrate
        x2 = torch.randn(100, 64) * 100  # Very different scale
        _ = sae(x2)
        
        # Thresholds unchanged (no gradient applied, just forward)
        assert torch.allclose(sae.jumprelu.threshold, threshold_after_calibration)
        
    def test_calibration_disabled(self, small_sae_params):
        """Test that calibration can be disabled."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, calibrate_thresholds=False)
        sae.train()
        
        initial_threshold = sae.jumprelu.threshold.mean().item()
        
        x = torch.randn(100, 64)
        _ = sae(x)
        
        # Threshold should not change without training
        assert sae.jumprelu.threshold.mean().item() == initial_threshold
        assert not hasattr(sae, 'thresholds_calibrated') or not sae.thresholds_calibrated


class TestJumpReLUActivation:
    """Tests for JumpReLU activation with learned thresholds."""
    
    def test_has_jumprelu_module(self, matryoshka_lagrangian_sae):
        """Test that JumpReLU module exists."""
        assert hasattr(matryoshka_lagrangian_sae, 'jumprelu')
        
    def test_jumprelu_has_learnable_thresholds(self, matryoshka_lagrangian_sae):
        """Test that JumpReLU has learnable threshold parameters."""
        sae = matryoshka_lagrangian_sae
        assert hasattr(sae.jumprelu, 'threshold')
        assert isinstance(sae.jumprelu.threshold, torch.nn.Parameter)
        
    def test_threshold_gradient_flows(self, small_sae_params):
        """Test that gradients flow to threshold parameters."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params)
        sae.train()
        
        x = torch.randn(32, 64)
        output = sae(x)
        loss_output = sae.compute_loss(output)
        loss_output.loss.backward()
        
        assert sae.jumprelu.threshold.grad is not None
        assert not torch.all(sae.jumprelu.threshold.grad == 0)


class TestDifferentiableL0:
    """Tests for differentiable L0 approximation."""
    
    def test_output_has_l0_differentiable(self, matryoshka_lagrangian_sae):
        """Test that output contains l0_differentiable."""
        sae = matryoshka_lagrangian_sae
        x = torch.randn(8, 64)
        output = sae(x)
        
        assert hasattr(output, 'l0_differentiable')
        assert output.l0_differentiable is not None
        
    def test_l0_differentiable_requires_grad(self, matryoshka_lagrangian_sae):
        """Test that l0_differentiable has gradient."""
        sae = matryoshka_lagrangian_sae
        sae.train()
        
        x = torch.randn(8, 64, requires_grad=True)
        output = sae(x)
        
        # l0_differentiable should be part of computation graph
        mean_l0_diff = output.l0_differentiable.mean()
        mean_l0_diff.backward()
        
        # Gradient should flow to thresholds
        assert sae.jumprelu.threshold.grad is not None


# ============================================================================
# Test: Normalization Consistency
# ============================================================================

class TestNormalizationConsistency:
    """Tests for activation normalization consistency with LagrangianSAE."""
    
    def test_has_normalize_activations_flag(self, small_sae_params):
        """Test that normalize_activations attribute exists."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, normalize_activations=True)
        assert hasattr(sae, 'normalize_activations')
        assert sae.normalize_activations == True
        
    def test_has_running_norm_factor(self, small_sae_params):
        """Test that running_norm_factor buffer exists when normalization enabled."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, normalize_activations=True)
        assert hasattr(sae, 'running_norm_factor')
        assert hasattr(sae, 'norm_factor_initialized')
        
    def test_has_set_norm_factor_method(self, small_sae_params):
        """Test that set_norm_factor method exists."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, normalize_activations=True)
        assert hasattr(sae, 'set_norm_factor')
        assert callable(sae.set_norm_factor)
        
    def test_set_norm_factor_updates_buffers(self, small_sae_params):
        """Test that set_norm_factor correctly updates buffers."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, normalize_activations=True)
        
        assert sae.norm_factor_initialized == False
        
        sae.set_norm_factor(42.5)
        
        assert sae.norm_factor_initialized == True
        assert torch.allclose(sae.running_norm_factor, torch.tensor(42.5))
        
    def test_norm_factor_num_batches_exists(self, small_sae_params):
        """Test that norm_factor_num_batches attribute exists (for run.py compatibility)."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, normalize_activations=True)
        assert hasattr(sae, 'norm_factor_num_batches')
        assert sae.norm_factor_num_batches == 100  # Default
        
    def test_normalization_affects_output_scale(self, small_sae_params):
        """Test that normalization affects output scale correctly."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, normalize_activations=True)
        sae.train()
        
        # Large scale input
        x = torch.randn(100, 64) * 10.0
        
        output = sae(x)
        
        # Output should be denormalized (similar scale to input)
        input_scale = (x ** 2).mean().sqrt().item()
        output_scale = (output.output ** 2).mean().sqrt().item()
        
        # Output scale should be in similar ballpark as input (within 10x)
        assert output_scale > input_scale * 0.1


# ============================================================================
# Test: Float32 Control Buffers
# ============================================================================

class TestFloat32ControlBuffers:
    """Tests for float32 precision in control buffers."""
    
    def test_alpha_stays_float32_after_dtype_conversion(self, small_sae_params):
        """Test that alpha buffer stays float32 after .to(dtype)."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params)
        
        # Convert to bfloat16
        sae = sae.to(dtype=torch.bfloat16)
        
        # Alpha should remain float32 for numerical stability
        assert sae.alpha.dtype == torch.float32
        
    def test_running_l0_stays_float32_after_dtype_conversion(self, small_sae_params):
        """Test that running_l0 buffer stays float32 after .to(dtype)."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params)
        
        # Convert to bfloat16
        sae = sae.to(dtype=torch.bfloat16)
        
        # running_l0 should remain float32 for EMA stability
        assert sae.running_l0.dtype == torch.float32


# ============================================================================
# Test: API Consistency
# ============================================================================

class TestAPIConsistency:
    """Tests for API consistency between MatryoshkaLagrangianSAE and parent classes."""
    
    def test_forward_output_type(self, matryoshka_lagrangian_sae):
        """Test that forward returns MatryoshkaLagrangianSAEOutput."""
        sae = matryoshka_lagrangian_sae
        x = torch.randn(8, 64)
        output = sae(x)
        
        assert isinstance(output, MatryoshkaLagrangianSAEOutput)
        
    def test_output_has_required_fields(self, matryoshka_lagrangian_sae):
        """Test that output has all required fields."""
        sae = matryoshka_lagrangian_sae
        x = torch.randn(8, 64)
        output = sae(x)
        
        # Base SAEOutput fields
        assert hasattr(output, 'input')
        assert hasattr(output, 'c')
        assert hasattr(output, 'output')
        assert hasattr(output, 'logits')
        
        # Lagrangian-specific fields
        assert hasattr(output, 'preacts')
        assert hasattr(output, 'l0')
        assert hasattr(output, 'l0_differentiable')
        assert hasattr(output, 'alpha')
        
    def test_forward_shape_consistency(self, matryoshka_lagrangian_sae):
        """Test that forward produces correct output shapes."""
        sae = matryoshka_lagrangian_sae
        
        # Test with 2D input
        x_2d = torch.randn(8, 64)
        output_2d = sae(x_2d)
        assert output_2d.input.shape == x_2d.shape
        assert output_2d.output.shape == x_2d.shape
        assert output_2d.c.shape == (8, 128)
        assert output_2d.l0.shape == (8,)
        
        # Test with 3D input (batch, seq, dim)
        x_3d = torch.randn(4, 16, 64)
        output_3d = sae(x_3d)
        assert output_3d.input.shape == x_3d.shape
        assert output_3d.output.shape == x_3d.shape
        assert output_3d.c.shape == (4, 16, 128)
        assert output_3d.l0.shape == (4, 16)
        
    def test_dict_elements_property(self, matryoshka_lagrangian_sae):
        """Test that dict_elements property exists and is normalized."""
        sae = matryoshka_lagrangian_sae
        
        assert hasattr(sae, 'dict_elements')
        dict_elem = sae.dict_elements
        
        # Should be column-wise unit norm
        norms = torch.norm(dict_elem, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
    def test_device_property(self, matryoshka_lagrangian_sae):
        """Test that device property exists."""
        sae = matryoshka_lagrangian_sae
        assert hasattr(sae, 'device')


# ============================================================================
# Test: Loss Computation
# ============================================================================

class TestLossComputation:
    """Tests for loss computation combining Matryoshka and Lagrangian components."""
    
    def test_loss_dict_has_required_keys(self, matryoshka_lagrangian_sae):
        """Test that loss_dict has all required keys."""
        sae = matryoshka_lagrangian_sae
        sae.train()
        
        x = torch.randn(8, 64)
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        required_keys = [
            'mse_loss',
            'running_l0',
            'alpha',
            'quadratic_penalty',
            'mean_threshold',
        ]
        
        for key in required_keys:
            assert key in loss_output.loss_dict, f"Missing key: {key}"
            
    def test_loss_includes_lagrangian_penalty(self, small_sae_params):
        """Test that loss includes Lagrangian sparsity penalty."""
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            initial_alpha=1.0,  # Non-zero alpha
            rho_quadratic=0.1,
            calibrate_thresholds=False,  # Avoid automatic calibration changing L0
        )
        sae.train()
        
        x = torch.randn(8, 64)
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        # Verify Lagrangian penalty terms are computed
        # quadratic_penalty should be positive when L0 != target
        quadratic_penalty = loss_output.loss_dict['quadratic_penalty']
        
        # The total loss includes mse_loss + sparsity_loss + quadratic_penalty
        # sparsity_loss = alpha * normalized_violation which can be negative
        # quadratic_penalty is always >= 0
        assert quadratic_penalty >= 0
        
        # Loss should be a valid tensor
        assert not torch.isnan(loss_output.loss)
        assert torch.isfinite(loss_output.loss)
        
    def test_quadratic_penalty_in_loss(self, small_sae_params):
        """Test that quadratic penalty is computed correctly."""
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            rho_quadratic=1.0,  # Strong quadratic penalty
        )
        sae.train()
        
        x = torch.randn(8, 64)
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        # Quadratic penalty should be non-negative
        assert loss_output.loss_dict['quadratic_penalty'] >= 0
        

# ============================================================================
# Test: Integration - Training Loop
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for training behavior."""
    
    def test_full_training_step(self, small_sae_params):
        """Test a full training step with optimizer."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params)
        sae.train()
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        
        x = torch.randn(32, 64)
        
        # Forward
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        # Backward
        optimizer.zero_grad()
        loss_output.loss.backward()
        optimizer.step()
        
        # Post-step: update alpha
        sae.update_alpha()
        
        # All should complete without error
        assert True
        
    def test_training_reduces_loss(self, small_sae_params):
        """Test that training reduces loss over multiple steps."""
        sae = MatryoshkaLagrangianSAE(**small_sae_params, calibrate_thresholds=False)
        sae.train()
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
        
        # Fixed input to overfit
        x = torch.randn(32, 64)
        
        initial_loss = None
        final_loss = None
        
        for step in range(100):
            output = sae(x)
            loss_output = sae.compute_loss(output)
            
            if step == 0:
                initial_loss = loss_output.loss.item()
            if step == 99:
                final_loss = loss_output.loss.item()
            
            optimizer.zero_grad()
            loss_output.loss.backward()
            optimizer.step()
            sae.update_alpha()
        
        # Loss should decrease
        assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {final_loss}"
        
    def test_l0_converges_toward_target(self, small_sae_params):
        """Test that L0 converges toward target over training."""
        target_l0 = small_sae_params["target_l0"]
        sae = MatryoshkaLagrangianSAE(
            **small_sae_params,
            alpha_lr=0.1,
            rho_quadratic=0.1,
        )
        sae.train()
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        
        for _ in range(200):
            x = torch.randn(32, 64)
            output = sae(x)
            loss_output = sae.compute_loss(output)
            
            optimizer.zero_grad()
            loss_output.loss.backward()
            optimizer.step()
            sae.update_alpha()
        
        # running_l0 should be closer to target than initial
        final_running_l0 = sae.running_l0.item()
        # Allow generous tolerance - convergence may take more steps
        assert abs(final_running_l0 - target_l0) < target_l0 * 2, \
            f"L0 ({final_running_l0}) far from target ({target_l0})"


# ============================================================================
# Test: Caveats Documentation (Known Differences)
# ============================================================================

class TestCaveatsDocumentation:
    """Tests that document known differences from parent implementations.
    
    These tests don't assert failures but document behavioral differences
    that users should be aware of.
    """
    
    def test_encoder_preact_has_relu_unlike_lagrangian(self, small_sae_params):
        """
        CAVEAT: MatryoshkaLagrangianSAE applies ReLU before JumpReLU,
        while LagrangianSAE passes raw encoder outputs to JumpReLU.
        
        This means threshold values have different semantics between the two.
        """
        matryoshka_lagrangian = MatryoshkaLagrangianSAE(**small_sae_params)
        lagrangian = LagrangianSAE(**small_sae_params)
        
        # Sync weights
        with torch.no_grad():
            lagrangian.encoder.weight.data.copy_(matryoshka_lagrangian.encoder.weight.data)
            lagrangian.encoder_bias.data.copy_(matryoshka_lagrangian.encoder_bias.data)
        
        x = torch.randn(32, 64)
        
        # Get pre-activations from both
        ml_output = matryoshka_lagrangian(x)
        l_output = lagrangian(x)
        
        # MatryoshkaLagrangian preacts should be all non-negative (ReLU applied)
        assert (ml_output.preacts >= 0).all(), "Preacts should be non-negative (ReLU)"
        
        # LagrangianSAE preacts can be negative (no ReLU)
        # Note: This test documents the difference, doesn't assert one is "correct"
        has_negative = (l_output.preacts < 0).any()
        # Usually there will be some negatives
        print(f"LagrangianSAE preacts has negative values: {has_negative}")
        
    def test_normalization_differs_from_matryoshka(self, small_sae_params):
        """
        CAVEAT: MatryoshkaLagrangianSAE uses normalize_activations (scale-only),
        while MatryoshkaSAE uses input_unit_norm (mean+std normalization).
        
        These are semantically different normalizations.
        """
        # MatryoshkaLagrangianSAE uses scale-only normalization
        ml_sae = MatryoshkaLagrangianSAE(**small_sae_params, normalize_activations=True)
        
        # MatryoshkaSAE uses mean+std normalization
        m_sae = MatryoshkaSAE(
            input_size=64, 
            n_dict_components=128, 
            k=8,
            input_unit_norm=True,
            init_decoder_orthogonal=False,  # Avoid CUDA for CPU tests
        )
        
        # Document the difference
        assert hasattr(ml_sae, 'normalize_activations')
        assert hasattr(m_sae, 'input_unit_norm')
        assert not hasattr(ml_sae, 'input_unit_norm')  # Different API
        assert not hasattr(m_sae, 'normalize_activations')  # Different API
        
    def test_sparsity_is_soft_not_hard(self, small_sae_params):
        """
        CAVEAT: MatryoshkaLagrangianSAE has variable L0 (soft sparsity via JumpReLU),
        while MatryoshkaSAE uses BatchTopK which guarantees k * batch_size total
        active features but not exactly k per sample.
        
        Direct L0/MSE comparisons are not meaningful due to different sparsity mechanisms.
        """
        target_l0 = small_sae_params["target_l0"]
        ml_sae = MatryoshkaLagrangianSAE(**small_sae_params)
        m_sae = MatryoshkaSAE(input_size=64, n_dict_components=128, k=8, init_decoder_orthogonal=False)
        
        batch_size = 100
        x = torch.randn(batch_size, 64)
        
        ml_output = ml_sae(x)
        m_output = m_sae(x)
        
        # MatryoshkaSAE uses BatchTopK: total active features = k * batch_size
        # But per-sample L0 varies (some samples get more, some get fewer)
        m_l0 = (m_output.c > 0).sum(dim=-1).float()
        total_active = m_l0.sum().item()
        expected_total = 8.0 * batch_size  # k * batch_size
        
        # Total should be close to k * batch_size (due to batch topk selection)
        assert abs(total_active - expected_total) < expected_total * 0.1, \
            f"MatryoshkaSAE total active features ({total_active}) should be close to k*batch_size ({expected_total})"
        
        # MatryoshkaLagrangianSAE has variable L0 based on learned thresholds
        ml_l0 = ml_output.l0
        ml_l0_std = ml_l0.std()
        # Should have some variance (not exactly target for every sample)
        print(f"MatryoshkaLagrangianSAE L0: mean={ml_l0.mean():.2f}, std={ml_l0_std:.2f}")
        print(f"MatryoshkaSAE L0: mean={m_l0.mean():.2f}, std={m_l0.std():.2f}")
        # We don't assert on the variance value, just document it varies
        
    def test_has_encoder_bias_unlike_matryoshka(self, small_sae_params):
        """
        CAVEAT: MatryoshkaLagrangianSAE has encoder_bias parameter,
        while MatryoshkaSAE does not have a separate encoder bias.
        """
        ml_sae = MatryoshkaLagrangianSAE(**small_sae_params)
        m_sae = MatryoshkaSAE(input_size=64, n_dict_components=128, k=8, init_decoder_orthogonal=False)
        
        assert hasattr(ml_sae, 'encoder_bias'), "MatryoshkaLagrangianSAE should have encoder_bias"
        assert isinstance(ml_sae.encoder_bias, torch.nn.Parameter)
        
        # MatryoshkaSAE encoder has no bias (Linear bias=False and no separate bias)
        assert m_sae.encoder.bias is None, "MatryoshkaSAE encoder should have no bias"


# ============================================================================
# Test: Config Validation
# ============================================================================

class TestConfigValidation:
    """Tests for config validation."""
    
    def test_config_validates_group_fractions(self):
        """Test that config validates group_fractions sum to 1.0."""
        # Valid config
        valid_config = MatryoshkaLagrangianSAEConfig(
            name="test",
            sae_positions=["blocks.0.hook_resid_pre"],
            target_l0=32.0,
            group_fractions=[0.5, 0.5],
        )
        assert isclose(sum(valid_config.group_fractions), 1.0, rel_tol=1e-5)
        
        # Invalid config
        with pytest.raises(ValueError):
            MatryoshkaLagrangianSAEConfig(
                name="test",
                sae_positions=["blocks.0.hook_resid_pre"],
                target_l0=32.0,
                group_fractions=[0.3, 0.3],  # Sum != 1.0
            )
            
    def test_config_validates_group_weights_length(self):
        """Test that config validates group_weights length matches group_fractions."""
        with pytest.raises(ValueError):
            MatryoshkaLagrangianSAEConfig(
                name="test",
                sae_positions=["blocks.0.hook_resid_pre"],
                target_l0=32.0,
                group_fractions=[0.5, 0.5],  # 2 groups
                group_weights=[0.25, 0.25, 0.25, 0.25],  # 4 weights - mismatch!
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
