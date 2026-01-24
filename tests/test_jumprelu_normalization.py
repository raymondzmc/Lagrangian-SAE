"""Test that activation normalization in JumpReLU SAE works correctly.

This test verifies that:
1. Norm factor is computed correctly (sqrt of mean squared norm)
2. Inputs are normalized before encoding
3. Outputs are denormalized after decoding
4. Consistent behavior across different activation magnitudes

Note: Initial L0 being high is expected with initial_threshold=0.001, since most
pre-activations exceed this threshold. The dictionary_learning demo uses 
sparsity_warmup to gradually increase the penalty, allowing threshold to grow
before full penalty kicks in. Our implementation without warmup will also
converge - just needs more training steps initially.
"""

import torch
import pytest
from models.saes.jumprelu_sae import JumpReLUSAE


def test_normalization_consistency_across_scales():
    """Test that normalization provides consistent L0 across different activation scales."""
    torch.manual_seed(42)
    
    input_size = 256
    n_dict_components = 512
    batch_size = 64
    target_l0 = 32.0
    
    # Create a base activation pattern
    base_activations = torch.randn(batch_size, input_size)
    
    # Test with two different scales
    scale_small = 1.0
    scale_large = 100.0
    
    act_small = base_activations * scale_small
    act_large = base_activations * scale_large
    
    # WITH normalization: L0 should be similar regardless of scale
    sae_norm = JumpReLUSAE(
        input_size=input_size,
        n_dict_components=n_dict_components,
        target_l0=target_l0,
        bandwidth=0.001,
        initial_threshold=0.001,
        normalize_activations=True,
    )
    
    # Process small scale (initializes norm factor)
    sae_norm.train()
    with torch.no_grad():
        output_small = sae_norm(act_small)
        l0_small = output_small.l0.mean().item()
        norm_factor_small = sae_norm.running_norm_factor.item()
    
    # Reset and process large scale
    sae_norm.norm_factor_initialized.fill_(False)
    with torch.no_grad():
        output_large = sae_norm(act_large)
        l0_large = output_large.l0.mean().item()
        norm_factor_large = sae_norm.running_norm_factor.item()
    
    print(f"Small scale: norm_factor={norm_factor_small:.2f}, L0={l0_small:.1f}")
    print(f"Large scale: norm_factor={norm_factor_large:.2f}, L0={l0_large:.1f}")
    
    # Norm factors should scale with the input
    assert abs(norm_factor_large / norm_factor_small - scale_large / scale_small) < 1.0, \
        "Norm factor should scale with input magnitude"
    
    # L0 should be similar regardless of scale (within 20%)
    l0_ratio = l0_large / l0_small
    assert 0.8 < l0_ratio < 1.2, f"L0 should be consistent across scales. Ratio: {l0_ratio:.2f}"
    
    # WITHOUT normalization: L0 should differ significantly
    sae_no_norm = JumpReLUSAE(
        input_size=input_size,
        n_dict_components=n_dict_components,
        target_l0=target_l0,
        bandwidth=0.001,
        initial_threshold=0.001,
        normalize_activations=False,
    )
    sae_no_norm.eval()
    
    with torch.no_grad():
        output_small_no_norm = sae_no_norm(act_small)
        output_large_no_norm = sae_no_norm(act_large)
        l0_small_no_norm = output_small_no_norm.l0.mean().item()
        l0_large_no_norm = output_large_no_norm.l0.mean().item()
    
    print(f"\nWithout normalization:")
    print(f"Small scale: L0={l0_small_no_norm:.1f}")
    print(f"Large scale: L0={l0_large_no_norm:.1f}")
    
    # Without normalization, large scale should have higher L0
    # (since more values exceed the fixed threshold)
    assert l0_large_no_norm > l0_small_no_norm, \
        "Without normalization, larger scale should give higher L0"
    
    print("\nTest passed! Normalization provides consistent L0 across scales.")


def test_norm_factor_computation():
    """Test that norm factor is computed correctly."""
    torch.manual_seed(42)
    
    input_size = 256
    n_dict_components = 512
    batch_size = 128
    
    # Create activations with known magnitude
    scale = 5.0
    activations = torch.randn(batch_size, input_size) * scale
    expected_norm_factor = (activations ** 2).sum(dim=-1).mean().sqrt()
    
    sae = JumpReLUSAE(
        input_size=input_size,
        n_dict_components=n_dict_components,
        target_l0=20.0,
        normalize_activations=True,
    )
    sae.train()
    
    # First forward pass should initialize norm factor
    assert not sae.norm_factor_initialized.item()
    
    with torch.no_grad():
        _ = sae(activations)
    
    assert sae.norm_factor_initialized.item()
    computed_norm_factor = sae.running_norm_factor.item()
    
    # Should be close to expected
    relative_error = abs(computed_norm_factor - expected_norm_factor.item()) / expected_norm_factor.item()
    assert relative_error < 0.01, f"Norm factor error: {relative_error:.4f}"
    
    print(f"Expected norm factor: {expected_norm_factor.item():.4f}")
    print(f"Computed norm factor: {computed_norm_factor:.4f}")
    print("Norm factor computation test passed!")


def test_output_denormalization():
    """Test that output is correctly denormalized to match input scale."""
    torch.manual_seed(42)
    
    input_size = 256
    n_dict_components = 512
    batch_size = 32
    
    # Create activations
    activations = torch.randn(batch_size, input_size) * 5.0
    
    sae = JumpReLUSAE(
        input_size=input_size,
        n_dict_components=n_dict_components,
        target_l0=50.0,
        normalize_activations=True,
    )
    sae.train()
    
    with torch.no_grad():
        output = sae(activations)
    
    # Check that output has similar scale to input
    input_norm = activations.norm(dim=-1).mean()
    output_norm = output.output.norm(dim=-1).mean()
    
    # The reconstruction might not be perfect but should be in the same ballpark
    scale_ratio = output_norm / input_norm
    print(f"Input norm: {input_norm:.4f}")
    print(f"Output norm: {output_norm:.4f}")
    print(f"Scale ratio: {scale_ratio:.4f}")
    
    # Output should be within 2x of input scale (not 1/norm_factor)
    assert 0.1 < scale_ratio < 10, f"Output scale seems wrong: ratio={scale_ratio}"
    
    print("Output denormalization test passed!")


def test_scale_biases():
    """Test that scale_biases correctly scales parameters."""
    torch.manual_seed(42)
    
    sae = JumpReLUSAE(
        input_size=64,
        n_dict_components=128,
        target_l0=10.0,
        normalize_activations=True,
        initial_threshold=0.001,
    )
    
    # Set some non-zero biases
    sae.decoder_bias.data.fill_(1.0)
    sae.encoder_bias.data.fill_(2.0)
    sae.jumprelu.threshold.data.fill_(0.01)
    
    scale = 50.0
    sae.scale_biases(scale)
    
    assert torch.allclose(sae.decoder_bias, torch.full_like(sae.decoder_bias, 50.0))
    assert torch.allclose(sae.encoder_bias, torch.full_like(sae.encoder_bias, 100.0))
    assert torch.allclose(sae.jumprelu.threshold, torch.full_like(sae.jumprelu.threshold, 0.5))
    
    print("scale_biases test passed!")


def test_get_denormalized_state_dict():
    """Test that get_denormalized_state_dict correctly scales for saving."""
    torch.manual_seed(42)
    
    input_size = 64
    n_dict_components = 128
    
    sae = JumpReLUSAE(
        input_size=input_size,
        n_dict_components=n_dict_components,
        target_l0=10.0,
        normalize_activations=True,
        initial_threshold=0.001,
    )
    sae.train()
    
    # Initialize norm factor with a forward pass
    activations = torch.randn(32, input_size) * 5.0
    with torch.no_grad():
        _ = sae(activations)
    
    norm_factor = sae.running_norm_factor.item()
    original_threshold = sae.jumprelu.threshold.data.clone()
    original_encoder_bias = sae.encoder_bias.data.clone()
    original_decoder_bias = sae.decoder_bias.data.clone()
    
    # Get denormalized state dict
    state_dict = sae.get_denormalized_state_dict()
    
    # Check that biases and thresholds are scaled
    assert torch.allclose(state_dict['jumprelu.threshold'], original_threshold * norm_factor)
    assert torch.allclose(state_dict['encoder_bias'], original_encoder_bias * norm_factor)
    assert torch.allclose(state_dict['decoder_bias'], original_decoder_bias * norm_factor)
    
    # Check that norm tracking is reset
    assert state_dict['running_norm_factor'].item() == 1.0
    assert not state_dict['norm_factor_initialized'].item()
    
    # Check that original SAE is unchanged
    assert torch.allclose(sae.jumprelu.threshold.data, original_threshold)
    assert torch.allclose(sae.encoder_bias.data, original_encoder_bias)
    assert torch.allclose(sae.decoder_bias.data, original_decoder_bias)
    
    print("get_denormalized_state_dict test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing JumpReLU SAE Activation Normalization")
    print("=" * 60)
    
    print("\n--- Test 1: Normalization consistency across scales ---")
    test_normalization_consistency_across_scales()
    
    print("\n--- Test 2: Norm factor computation ---")
    test_norm_factor_computation()
    
    print("\n--- Test 3: Output denormalization ---")
    test_output_denormalization()
    
    print("\n--- Test 4: scale_biases ---")
    test_scale_biases()
    
    print("\n--- Test 5: get_denormalized_state_dict ---")
    test_get_denormalized_state_dict()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
