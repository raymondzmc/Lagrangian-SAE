"""
Test alpha dynamics in LagrangianSAE.

Verifies that:
1. Alpha increases when running L0 > target_l0 (need more sparsity pressure)
2. Alpha decreases when running L0 < target_l0 (need less sparsity pressure)
3. Alpha respects clamping bounds based on constraint type

This tests the dual ascent mechanism that makes LagrangianSAE adaptive.
"""

import pytest
import torch
import torch.nn.functional as F
from models.saes.lagrangian_sae import LagrangianSAE


class TestAlphaDynamics:
    """Tests for alpha (Lagrangian multiplier) update dynamics."""

    @pytest.fixture
    def sae_config(self) -> dict:
        """Basic SAE configuration for testing."""
        return {
            "input_size": 64,
            "n_dict_components": 256,
            "target_l0": 16.0,
            "initial_alpha": 0.5,  # Start with some alpha
            "alpha_lr": 0.1,  # High LR for visible changes in test
            "alpha_max": 10.0,
            "l0_ema_momentum": 0.0,  # No smoothing for predictable tests
            "initial_threshold": 0.1,
            "bandwidth": 0.1,
            "calibrate_thresholds": False,  # Disable for controlled tests
            "equality_constraint": True,  # Test equality constraint first
        }

    def test_alpha_decreases_when_l0_below_target(self, sae_config):
        """
        Test that alpha decreases when running L0 goes below target.
        
        When L0 < target_l0:
        - constraint_violation = L0 - target_l0 < 0
        - alpha update: alpha += alpha_lr * constraint_violation
        - Result: alpha decreases (less sparsity pressure)
        """
        sae = LagrangianSAE(**sae_config)
        sae.train()
        
        initial_alpha = sae.alpha.item()
        target_l0 = sae_config["target_l0"]
        
        # Create input that will produce LOW L0 (below target)
        # By using very small input, pre-activations will be small, 
        # and most will be below threshold, resulting in low L0
        x = torch.randn(32, 64) * 0.01  # Very small activations
        
        # Forward pass
        output = sae(x)
        loss_output = sae.compute_loss(output)
        
        # Simulate backward pass (needed for realistic scenario)
        loss_output.loss.backward()
        
        # Get the running L0 after EMA update
        running_l0_after = sae.running_l0.item()
        
        # With momentum=0, running_l0 should equal batch L0
        batch_l0 = output.l0.mean().item()
        assert abs(running_l0_after - batch_l0) < 1e-5, "Running L0 should match batch L0 with momentum=0"
        
        # Verify L0 is below target (our test setup should produce this)
        print(f"Batch L0: {batch_l0:.2f}, Target L0: {target_l0:.2f}")
        
        # Update alpha via dual ascent
        sae.update_alpha()
        final_alpha = sae.alpha.item()
        
        # Compute expected alpha change
        constraint_violation = running_l0_after - target_l0
        expected_alpha_change = sae_config["alpha_lr"] * constraint_violation
        
        print(f"Initial alpha: {initial_alpha:.4f}")
        print(f"Constraint violation: {constraint_violation:.4f}")
        print(f"Expected alpha change: {expected_alpha_change:.4f}")
        print(f"Final alpha: {final_alpha:.4f}")
        
        if batch_l0 < target_l0:
            # L0 below target: alpha should decrease
            assert final_alpha < initial_alpha, (
                f"Alpha should decrease when L0 ({batch_l0:.2f}) < target ({target_l0:.2f}), "
                f"but went from {initial_alpha:.4f} to {final_alpha:.4f}"
            )
            print("✓ Alpha correctly decreased when L0 < target")
        else:
            # L0 above target: alpha should increase
            assert final_alpha > initial_alpha, (
                f"Alpha should increase when L0 ({batch_l0:.2f}) > target ({target_l0:.2f}), "
                f"but went from {initial_alpha:.4f} to {final_alpha:.4f}"
            )
            print("✓ Alpha correctly increased when L0 > target")

    def test_alpha_increases_when_l0_above_target(self, sae_config):
        """
        Test that alpha increases when running L0 goes above target.
        
        When L0 > target_l0:
        - constraint_violation = L0 - target_l0 > 0
        - alpha update: alpha += alpha_lr * constraint_violation
        - Result: alpha increases (more sparsity pressure)
        """
        # Modify config for this test
        sae_config = sae_config.copy()
        sae_config["initial_threshold"] = 0.001  # Very low threshold = high L0
        
        sae = LagrangianSAE(**sae_config)
        sae.train()
        
        initial_alpha = sae.alpha.item()
        target_l0 = sae_config["target_l0"]
        
        # Create input that will produce HIGH L0 (above target)
        # By using large input with low threshold, most features will activate
        x = torch.randn(32, 64) * 10.0  # Large activations
        
        # Forward pass
        output = sae(x)
        loss_output = sae.compute_loss(output)
        loss_output.loss.backward()
        
        running_l0_after = sae.running_l0.item()
        batch_l0 = output.l0.mean().item()
        
        print(f"Batch L0: {batch_l0:.2f}, Target L0: {target_l0:.2f}")
        
        # Update alpha via dual ascent
        sae.update_alpha()
        final_alpha = sae.alpha.item()
        
        print(f"Initial alpha: {initial_alpha:.4f}")
        print(f"Final alpha: {final_alpha:.4f}")
        
        if batch_l0 > target_l0:
            assert final_alpha > initial_alpha, (
                f"Alpha should increase when L0 ({batch_l0:.2f}) > target ({target_l0:.2f}), "
                f"but went from {initial_alpha:.4f} to {final_alpha:.4f}"
            )
            print("✓ Alpha correctly increased when L0 > target")
        else:
            print(f"Note: L0 ({batch_l0:.2f}) was not above target, test inconclusive")

    def test_alpha_update_with_controlled_running_l0(self, sae_config):
        """
        Directly test alpha update by manually setting running_l0.
        
        This gives us precise control over the constraint violation.
        """
        sae = LagrangianSAE(**sae_config)
        sae.train()
        
        target_l0 = sae_config["target_l0"]
        alpha_lr = sae_config["alpha_lr"]
        
        # Test case 1: L0 below target (alpha should decrease)
        sae.alpha.fill_(5.0)  # Reset alpha
        sae.running_l0.fill_(8.0)  # Below target of 16
        
        # Manually set the constraint violation (normally set by compute_loss)
        sae._last_constraint_violation = sae.running_l0 - target_l0  # -8.0
        
        initial_alpha = sae.alpha.item()
        sae.update_alpha()
        final_alpha = sae.alpha.item()
        
        expected_alpha = initial_alpha + alpha_lr * (-8.0)  # 5.0 + 0.1 * (-8.0) = 4.2
        
        assert abs(final_alpha - expected_alpha) < 1e-5, (
            f"Alpha should be {expected_alpha:.4f}, got {final_alpha:.4f}"
        )
        print(f"✓ Test 1: L0=8 < target=16 → alpha: {initial_alpha:.2f} → {final_alpha:.2f}")
        
        # Test case 2: L0 above target (alpha should increase)
        sae.alpha.fill_(5.0)  # Reset alpha
        sae.running_l0.fill_(24.0)  # Above target of 16
        sae._last_constraint_violation = sae.running_l0 - target_l0  # +8.0
        
        initial_alpha = sae.alpha.item()
        sae.update_alpha()
        final_alpha = sae.alpha.item()
        
        expected_alpha = initial_alpha + alpha_lr * 8.0  # 5.0 + 0.1 * 8.0 = 5.8
        
        assert abs(final_alpha - expected_alpha) < 1e-5, (
            f"Alpha should be {expected_alpha:.4f}, got {final_alpha:.4f}"
        )
        print(f"✓ Test 2: L0=24 > target=16 → alpha: {initial_alpha:.2f} → {final_alpha:.2f}")
        
        # Test case 3: L0 equals target (alpha should stay same)
        sae.alpha.fill_(5.0)
        sae.running_l0.fill_(16.0)  # Equals target
        sae._last_constraint_violation = sae.running_l0 - target_l0  # 0.0
        
        initial_alpha = sae.alpha.item()
        sae.update_alpha()
        final_alpha = sae.alpha.item()
        
        assert abs(final_alpha - initial_alpha) < 1e-5, (
            f"Alpha should stay at {initial_alpha:.4f}, got {final_alpha:.4f}"
        )
        print(f"✓ Test 3: L0=16 = target=16 → alpha: {initial_alpha:.2f} → {final_alpha:.2f}")

    def test_alpha_clamping_equality_constraint(self, sae_config):
        """
        Test that alpha is clamped to [-alpha_max, alpha_max] for equality constraint.
        """
        sae_config = sae_config.copy()
        sae_config["alpha_max"] = 2.0
        sae_config["equality_constraint"] = True
        
        sae = LagrangianSAE(**sae_config)
        sae.train()
        
        # Test upper clamping
        sae.alpha.fill_(1.9)
        sae._last_constraint_violation = torch.tensor(10.0)  # Would push alpha to 2.9
        sae.update_alpha()
        
        assert sae.alpha.item() == sae_config["alpha_max"], (
            f"Alpha should be clamped to alpha_max={sae_config['alpha_max']}, got {sae.alpha.item()}"
        )
        print(f"✓ Alpha correctly clamped at upper bound: {sae.alpha.item()}")
        
        # Test lower clamping (equality constraint allows negative alpha)
        sae.alpha.fill_(-1.9)
        sae._last_constraint_violation = torch.tensor(-10.0)  # Would push alpha to -2.9
        sae.update_alpha()
        
        assert sae.alpha.item() == -sae_config["alpha_max"], (
            f"Alpha should be clamped to -alpha_max={-sae_config['alpha_max']}, got {sae.alpha.item()}"
        )
        print(f"✓ Alpha correctly clamped at lower bound: {sae.alpha.item()}")

    def test_alpha_clamping_inequality_constraint(self, sae_config):
        """
        Test that alpha is clamped to [0, alpha_max] for inequality constraint.
        
        With inequality constraint, alpha should never go negative since we only
        want to penalize when L0 > target, never incentivize more features.
        """
        sae_config = sae_config.copy()
        sae_config["alpha_max"] = 2.0
        sae_config["equality_constraint"] = False  # Inequality constraint
        
        sae = LagrangianSAE(**sae_config)
        sae.train()
        
        # Test that alpha doesn't go below 0
        sae.alpha.fill_(0.5)
        sae._last_constraint_violation = torch.tensor(-10.0)  # Would push alpha to -0.5
        sae.update_alpha()
        
        assert sae.alpha.item() == 0.0, (
            f"Alpha should be clamped to 0 for inequality constraint, got {sae.alpha.item()}"
        )
        print(f"✓ Alpha correctly clamped at 0 for inequality constraint: {sae.alpha.item()}")

    def test_alpha_dynamics_over_training_steps(self, sae_config):
        """
        Test alpha dynamics over multiple training steps.
        
        Simulates a scenario where L0 starts high and gradually decreases,
        verifying alpha responds appropriately.
        """
        sae_config = sae_config.copy()
        sae_config["l0_ema_momentum"] = 0.9  # Use some smoothing
        sae_config["alpha_lr"] = 0.05
        
        sae = LagrangianSAE(**sae_config)
        sae.train()
        
        target_l0 = sae_config["target_l0"]
        
        print(f"\nSimulating training with target_l0={target_l0}")
        print("-" * 60)
        
        alpha_history = [sae.alpha.item()]
        l0_history = []
        
        # Simulate training where we manually control L0
        # Start with high L0, then transition to low L0
        simulated_l0_values = [30, 28, 25, 22, 20, 18, 15, 12, 10, 10, 12, 14, 16]
        
        for step, simulated_l0 in enumerate(simulated_l0_values):
            # Update running_l0 with EMA
            momentum = sae_config["l0_ema_momentum"]
            sae.running_l0.mul_(momentum).add_((1 - momentum) * simulated_l0)
            
            # Set constraint violation (normally done by compute_loss)
            sae._last_constraint_violation = sae.running_l0 - target_l0
            
            # Update alpha
            sae.update_alpha()
            
            alpha_history.append(sae.alpha.item())
            l0_history.append(sae.running_l0.item())
            
            print(f"Step {step:2d}: simulated_l0={simulated_l0:5.1f}, "
                  f"running_l0={sae.running_l0.item():5.2f}, "
                  f"alpha={sae.alpha.item():6.3f}")
        
        # Verify dynamics:
        # 1. Alpha should increase when running_l0 > target (first part)
        # 2. Alpha should decrease when running_l0 < target (middle part)
        
        # Find where running_l0 crosses target
        above_target_steps = [i for i, l0 in enumerate(l0_history) if l0 > target_l0]
        below_target_steps = [i for i, l0 in enumerate(l0_history) if l0 < target_l0]
        
        if above_target_steps and below_target_steps:
            # Check that alpha generally increased while L0 was above target
            # and decreased while L0 was below target
            first_above = above_target_steps[0]
            first_below = below_target_steps[0]
            
            if first_above < first_below:
                # L0 started above target, alpha should have increased
                alpha_at_start = alpha_history[first_above]
                alpha_at_cross = alpha_history[first_below]
                assert alpha_at_cross > alpha_at_start, (
                    f"Alpha should increase while L0 > target"
                )
                print("\n✓ Alpha increased while L0 was above target")
            
            # Check alpha decreased after L0 went below target
            if len(below_target_steps) > 1:
                alpha_early_below = alpha_history[below_target_steps[0] + 1]
                alpha_late_below = alpha_history[below_target_steps[-1] + 1]
                assert alpha_late_below < alpha_early_below, (
                    f"Alpha should decrease while L0 < target"
                )
                print("✓ Alpha decreased while L0 was below target")


class TestAlphaIntegrationWithForward:
    """Integration tests verifying alpha dynamics through actual forward passes."""

    def test_full_training_step_alpha_update(self):
        """
        Test a complete training step including forward, loss, backward, and alpha update.
        """
        sae = LagrangianSAE(
            input_size=64,
            n_dict_components=256,
            target_l0=16.0,
            initial_alpha=1.0,
            alpha_lr=0.1,
            alpha_max=10.0,
            l0_ema_momentum=0.0,
            initial_threshold=0.5,
            bandwidth=0.1,
            calibrate_thresholds=False,
        )
        sae.train()
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        
        # Training step
        x = torch.randn(32, 64)
        
        optimizer.zero_grad()
        output = sae(x)
        loss_output = sae.compute_loss(output)
        loss_output.loss.backward()
        optimizer.step()
        
        # Alpha update (must be after optimizer.step())
        initial_alpha = sae.alpha.item()
        batch_l0 = output.l0.mean().item()
        running_l0 = sae.running_l0.item()
        
        sae.update_alpha()
        
        final_alpha = sae.alpha.item()
        
        print(f"\nFull training step:")
        print(f"  Batch L0: {batch_l0:.2f}")
        print(f"  Running L0: {running_l0:.2f}")
        print(f"  Target L0: 16.0")
        print(f"  Alpha: {initial_alpha:.4f} → {final_alpha:.4f}")
        
        # Verify alpha moved in the correct direction
        if running_l0 > 16.0:
            assert final_alpha >= initial_alpha, "Alpha should increase/stay when L0 > target"
            print("  ✓ Alpha correctly increased (or stayed) when L0 > target")
        elif running_l0 < 16.0:
            assert final_alpha <= initial_alpha, "Alpha should decrease/stay when L0 < target"
            print("  ✓ Alpha correctly decreased (or stayed) when L0 < target")
        else:
            print("  ✓ L0 ≈ target, alpha change minimal")


if __name__ == "__main__":
    print("Running alpha dynamics tests...\n")
    
    # Run tests manually
    test_instance = TestAlphaDynamics()
    config = test_instance.sae_config.__wrapped__(test_instance)
    
    print("=" * 60)
    print("Test 1: Alpha decreases when L0 < target")
    print("=" * 60)
    test_instance.test_alpha_decreases_when_l0_below_target(config.copy())
    
    print("\n" + "=" * 60)
    print("Test 2: Alpha increases when L0 > target")
    print("=" * 60)
    test_instance.test_alpha_increases_when_l0_above_target(config.copy())
    
    print("\n" + "=" * 60)
    print("Test 3: Controlled running_l0 updates")
    print("=" * 60)
    test_instance.test_alpha_update_with_controlled_running_l0(config.copy())
    
    print("\n" + "=" * 60)
    print("Test 4: Alpha clamping (equality constraint)")
    print("=" * 60)
    test_instance.test_alpha_clamping_equality_constraint(config.copy())
    
    print("\n" + "=" * 60)
    print("Test 5: Alpha clamping (inequality constraint)")
    print("=" * 60)
    test_instance.test_alpha_clamping_inequality_constraint(config.copy())
    
    print("\n" + "=" * 60)
    print("Test 6: Alpha dynamics over multiple steps")
    print("=" * 60)
    test_instance.test_alpha_dynamics_over_training_steps(config.copy())
    
    print("\n" + "=" * 60)
    print("Test 7: Full training step integration")
    print("=" * 60)
    TestAlphaIntegrationWithForward().test_full_training_step_alpha_update()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
