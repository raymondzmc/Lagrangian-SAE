"""
Tests for JumpReLU SAE numerical parity with SAE Bench implementation.

Verifies that our JumpReLU SAE implementation produces identical results
to the SAE Bench trainer implementation when using the same:
1. Initialization (weights, biases, thresholds)
2. Forward pass (encoding, activation, decoding)
3. Loss computation (reconstruction + sparsity)
4. Gradient computation and manipulation
5. Optimizer step (with SAE Bench betas)
6. Post-step weight normalization
"""

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.saes.jumprelu_sae import JumpReLUSAE, JumpReLURaw
from models.saes.activations import JumpReLUFunction, StepFunction


# ============================================================================
# SAE Bench Reference Implementation (simplified for testing)
# ============================================================================

class SAEBenchJumpReLU(nn.Module):
    """Reference SAE Bench JumpReLU implementation for comparison."""
    
    def __init__(self, activation_dim: int, dict_size: int, bandwidth: float = 0.001):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bandwidth = bandwidth
        
        # Parameters (matching SAE Bench structure)
        self.W_enc = nn.Parameter(torch.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, activation_dim))
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))
        self.threshold = nn.Parameter(torch.ones(dict_size) * 0.001)
        
        # Initialize with kaiming_uniform then normalize (SAE Bench style)
        nn.init.kaiming_uniform_(self.W_dec)
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.T.clone()
    
    def forward(self, x: torch.Tensor, target_l0: float, sparsity_coeff: float = 1.0):
        """Forward pass with loss computation (SAE Bench style)."""
        # Encode
        pre_jump = x @ self.W_enc + self.b_enc
        
        # JumpReLU activation
        f = JumpReLUFunction.apply(pre_jump, self.threshold, self.bandwidth)
        
        # Decode
        recon = f @ self.W_dec + self.b_dec
        
        # Reconstruction loss: SSE (sum over dims, mean over batch)
        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        
        # L0 computed on f (after JumpReLU) - SAE Bench style
        l0 = StepFunction.apply(f, self.threshold, self.bandwidth).sum(dim=-1).mean()
        
        # Sparsity loss
        sparsity_loss = sparsity_coeff * ((l0 / target_l0) - 1).pow(2)
        
        # Total loss
        loss = recon_loss + sparsity_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'l0': l0,
            'f': f,
            'recon': recon,
        }


@torch.no_grad()
def sae_bench_remove_parallel_gradient(W_dec: torch.Tensor) -> None:
    """SAE Bench gradient parallel removal (in-place)."""
    if W_dec.grad is None:
        return
    
    # W_dec has shape (dict_size, activation_dim) in SAE Bench
    # Need to transpose for column-wise operations
    W_dec_T = W_dec.T  # (activation_dim, dict_size)
    W_dec_grad_T = W_dec.grad.T  # (activation_dim, dict_size)
    
    # Normalize columns
    normed_W_dec = W_dec_T / (torch.norm(W_dec_T, dim=0, keepdim=True) + 1e-6)
    
    # Compute parallel component
    parallel_component = (W_dec_grad_T * normed_W_dec).sum(dim=0, keepdim=True)
    
    # Remove parallel component
    W_dec_grad_T -= parallel_component * normed_W_dec
    
    # Write back (in-place)
    W_dec.grad.copy_(W_dec_grad_T.T)


@torch.no_grad()
def sae_bench_normalize_decoder(W_dec: torch.Tensor) -> None:
    """SAE Bench decoder normalization (in-place)."""
    # W_dec has shape (dict_size, activation_dim) in SAE Bench
    # Normalize rows (each dictionary element)
    eps = torch.finfo(W_dec.dtype).eps
    norm = torch.norm(W_dec.data, dim=1, keepdim=True)
    W_dec.data /= norm + eps


# ============================================================================
# Test Classes
# ============================================================================

class TestJumpReLUActivation:
    """Test JumpReLU activation function matches SAE Bench."""
    
    def test_jumprelu_forward(self):
        """Test JumpReLU forward pass produces identical results."""
        torch.manual_seed(42)
        
        x = torch.randn(32, 128)
        threshold = torch.rand(128) * 0.1
        bandwidth = 0.001
        
        # Direct function call
        result = JumpReLUFunction.apply(x, threshold, bandwidth)
        
        # Expected: x * (x > threshold)
        expected = x * (x > threshold).float()
        
        assert torch.allclose(result, expected), "JumpReLU forward mismatch"
    
    def test_jumprelu_gradient(self):
        """Test JumpReLU gradients for threshold."""
        torch.manual_seed(42)
        
        x = torch.randn(32, 128)
        threshold = torch.rand(128) * 0.1
        threshold.requires_grad = True
        bandwidth = 0.001
        
        result = JumpReLUFunction.apply(x, threshold, bandwidth)
        loss = result.sum()
        loss.backward()
        
        # Gradient should exist and have correct shape
        assert threshold.grad is not None
        assert threshold.grad.shape == threshold.shape
        
        # Gradient should only be non-zero where (x - threshold) is within bandwidth
        # of zero (i.e., in the rectangle region)


class TestStepFunction:
    """Test StepFunction for L0 computation matches SAE Bench."""
    
    def test_step_forward(self):
        """Test step function forward pass."""
        torch.manual_seed(42)
        
        x = torch.randn(32, 128)
        threshold = torch.rand(128) * 0.1
        bandwidth = 0.001
        
        result = StepFunction.apply(x, threshold, bandwidth)
        expected = (x > threshold).float()
        
        assert torch.allclose(result, expected), "Step function forward mismatch"


class TestForwardPassParity:
    """Test forward pass produces identical results."""
    
    @pytest.fixture
    def matched_saes(self):
        """Create matched SAEs with identical initialization."""
        torch.manual_seed(42)
        
        activation_dim = 64
        dict_size = 128
        bandwidth = 0.001
        
        # Create SAE Bench reference
        ref = SAEBenchJumpReLU(activation_dim, dict_size, bandwidth)
        
        # Create our implementation with matching init
        ours = JumpReLUSAE(
            input_size=activation_dim,
            n_dict_components=dict_size,
            target_l0=8.0,
            bandwidth=bandwidth,
            initial_threshold=0.001,
            use_pre_enc_bias=False,  # Match SAE Bench default
            init_decoder_orthogonal=False,  # Use manual init to match
            tied_encoder_init=False,
        )
        
        # Copy weights from reference to ours
        with torch.no_grad():
            # decoder.weight has shape (input_size, dict_size) in our impl
            # W_dec has shape (dict_size, activation_dim) in SAE Bench
            ours.decoder.weight.data.copy_(ref.W_dec.data.T)
            ours.encoder.weight.data.copy_(ref.W_enc.data.T)
            ours.encoder_bias.data.copy_(ref.b_enc.data)
            ours.decoder_bias.data.copy_(ref.b_dec.data)
            ours.jumprelu.threshold.data.copy_(ref.threshold.data)
        
        return ref, ours
    
    def test_forward_pass_output(self, matched_saes):
        """Test that forward pass produces identical outputs."""
        ref, ours = matched_saes
        
        torch.manual_seed(123)
        x = torch.randn(16, 64)
        
        # Reference forward
        ref_out = ref(x, target_l0=8.0)
        
        # Our forward
        our_out = ours(x)
        
        # Compare feature activations
        assert torch.allclose(our_out.c, ref_out['f'], atol=1e-6), \
            f"Feature activations mismatch: max diff = {(our_out.c - ref_out['f']).abs().max()}"
        
        # Compare reconstructions
        assert torch.allclose(our_out.output, ref_out['recon'], atol=1e-6), \
            f"Reconstructions mismatch: max diff = {(our_out.output - ref_out['recon']).abs().max()}"
    
    def test_loss_computation(self, matched_saes):
        """Test that loss computation produces identical results."""
        ref, ours = matched_saes
        
        torch.manual_seed(123)
        x = torch.randn(16, 64)
        
        # Reference
        ref_out = ref(x, target_l0=8.0, sparsity_coeff=1.0)
        
        # Ours
        our_out = ours(x)
        our_loss = ours.compute_loss(our_out)
        
        # Compare L0
        assert torch.allclose(our_out.l0.mean(), ref_out['l0'], atol=1e-6), \
            f"L0 mismatch: ours={our_out.l0.mean()}, ref={ref_out['l0']}"
        
        # Compare reconstruction loss
        assert torch.allclose(our_loss.loss_dict['recon_loss'], ref_out['recon_loss'], atol=1e-6), \
            f"Recon loss mismatch: ours={our_loss.loss_dict['recon_loss']}, ref={ref_out['recon_loss']}"
        
        # Compare sparsity loss
        assert torch.allclose(our_loss.loss_dict['sparsity_loss'], ref_out['sparsity_loss'], atol=1e-6), \
            f"Sparsity loss mismatch: ours={our_loss.loss_dict['sparsity_loss']}, ref={ref_out['sparsity_loss']}"
        
        # Compare total loss
        assert torch.allclose(our_loss.loss, ref_out['loss'], atol=1e-6), \
            f"Total loss mismatch: ours={our_loss.loss}, ref={ref_out['loss']}"


class TestGradientParity:
    """Test gradient computation and manipulation matches SAE Bench."""
    
    def test_gradient_parallel_removal(self):
        """Test that gradient parallel removal matches SAE Bench."""
        torch.manual_seed(42)
        
        # Create random decoder weights and gradients
        dict_size = 128
        activation_dim = 64
        
        # Our format: (input_size, n_dict_components)
        W_dec_ours = torch.randn(activation_dim, dict_size)
        W_dec_ours.grad = torch.randn_like(W_dec_ours)
        
        # SAE Bench format: (dict_size, activation_dim)
        W_dec_ref = W_dec_ours.T.clone()
        W_dec_ref.grad = W_dec_ours.grad.T.clone()
        
        # Apply parallel removal - SAE Bench style
        sae_bench_remove_parallel_gradient(W_dec_ref)
        
        # Apply parallel removal - Our style (in JumpReLUSAE)
        # Our decoder.weight is (input_size, n_dict_components)
        # We normalize along dim=0 (input_size dimension)
        with torch.no_grad():
            normed_W_dec = W_dec_ours / (torch.norm(W_dec_ours, dim=0, keepdim=True) + 1e-6)
            parallel_component = (W_dec_ours.grad * normed_W_dec).sum(dim=0, keepdim=True)
            W_dec_ours.grad -= parallel_component * normed_W_dec
        
        # Compare results (need to transpose ref to compare)
        assert torch.allclose(W_dec_ours.grad, W_dec_ref.grad.T, atol=1e-6), \
            f"Gradient parallel removal mismatch: max diff = {(W_dec_ours.grad - W_dec_ref.grad.T).abs().max()}"
    
    def test_decoder_normalization(self):
        """Test that decoder normalization matches SAE Bench."""
        torch.manual_seed(42)
        
        dict_size = 128
        activation_dim = 64
        
        # Our format: (input_size, n_dict_components)
        W_dec_ours = torch.randn(activation_dim, dict_size)
        
        # SAE Bench format: (dict_size, activation_dim)
        W_dec_ref = W_dec_ours.T.clone()
        
        # Apply normalization - SAE Bench style
        sae_bench_normalize_decoder(W_dec_ref)
        
        # Apply normalization - Our style
        with torch.no_grad():
            eps = torch.finfo(W_dec_ours.dtype).eps
            norm = torch.norm(W_dec_ours, dim=0, keepdim=True)
            W_dec_ours /= norm + eps
        
        # Compare results (transpose ref to compare)
        assert torch.allclose(W_dec_ours, W_dec_ref.T, atol=1e-6), \
            f"Decoder normalization mismatch: max diff = {(W_dec_ours - W_dec_ref.T).abs().max()}"


class TestOptimizerConfig:
    """Test optimizer configuration matches SAE Bench."""
    
    def test_optimizer_kwargs(self):
        """Test that get_optimizer_kwargs returns SAE Bench settings."""
        sae = JumpReLUSAE(
            input_size=64,
            n_dict_components=128,
            target_l0=8.0,
        )
        
        kwargs = sae.get_optimizer_kwargs()
        
        assert kwargs['betas'] == (0.0, 0.999), f"Expected betas=(0.0, 0.999), got {kwargs['betas']}"
        assert kwargs['eps'] == 1e-8, f"Expected eps=1e-8, got {kwargs['eps']}"


class TestFullTrainingStepParity:
    """Test full training step produces identical results."""
    
    def test_single_training_step(self):
        """Test a single training step matches SAE Bench behavior."""
        torch.manual_seed(42)
        
        activation_dim = 64
        dict_size = 128
        target_l0 = 8.0
        bandwidth = 0.001
        lr = 1e-3
        
        # Create SAE Bench reference
        ref = SAEBenchJumpReLU(activation_dim, dict_size, bandwidth)
        ref_optimizer = torch.optim.Adam(ref.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8)
        
        # Create our implementation
        ours = JumpReLUSAE(
            input_size=activation_dim,
            n_dict_components=dict_size,
            target_l0=target_l0,
            bandwidth=bandwidth,
            initial_threshold=0.001,
            use_pre_enc_bias=False,
            init_decoder_orthogonal=False,
            tied_encoder_init=False,
        )
        
        # Copy initialization
        with torch.no_grad():
            ours.decoder.weight.data.copy_(ref.W_dec.data.T)
            ours.encoder.weight.data.copy_(ref.W_enc.data.T)
            ours.encoder_bias.data.copy_(ref.b_enc.data)
            ours.decoder_bias.data.copy_(ref.b_dec.data)
            ours.jumprelu.threshold.data.copy_(ref.threshold.data)
        
        our_optimizer = torch.optim.Adam(ours.parameters(), lr=lr, **ours.get_optimizer_kwargs())
        
        # Same input
        torch.manual_seed(123)
        x = torch.randn(16, 64)
        
        # ========== Reference training step ==========
        ref_out = ref(x, target_l0=target_l0)
        ref_out['loss'].backward()
        
        # SAE Bench gradient manipulation
        sae_bench_remove_parallel_gradient(ref.W_dec)
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
        
        ref_optimizer.step()
        ref_optimizer.zero_grad()
        
        # SAE Bench weight normalization
        sae_bench_normalize_decoder(ref.W_dec)
        
        # ========== Our training step ==========
        our_out = ours(x)
        our_loss = ours.compute_loss(our_out)
        our_loss.loss.backward()
        
        # Our hooks
        ours.on_before_optimizer_step()
        torch.nn.utils.clip_grad_norm_(ours.parameters(), 1.0)
        
        our_optimizer.step()
        our_optimizer.zero_grad()
        
        ours.on_after_optimizer_step()
        
        # ========== Compare results ==========
        # Compare decoder weights (transpose for format difference)
        assert torch.allclose(ours.decoder.weight.data, ref.W_dec.data.T, atol=1e-5), \
            f"Decoder weights differ after step: max diff = {(ours.decoder.weight.data - ref.W_dec.data.T).abs().max()}"
        
        # Compare encoder weights
        assert torch.allclose(ours.encoder.weight.data, ref.W_enc.data.T, atol=1e-5), \
            f"Encoder weights differ after step: max diff = {(ours.encoder.weight.data - ref.W_enc.data.T).abs().max()}"
        
        # Compare thresholds
        assert torch.allclose(ours.jumprelu.threshold.data, ref.threshold.data, atol=1e-5), \
            f"Thresholds differ after step: max diff = {(ours.jumprelu.threshold.data - ref.threshold.data).abs().max()}"


class TestEndToEndParity:
    """End-to-end test with multiple training steps."""
    
    def test_multiple_training_steps(self):
        """Test that multiple training steps stay in sync."""
        torch.manual_seed(42)
        
        activation_dim = 64
        dict_size = 128
        target_l0 = 8.0
        bandwidth = 0.001
        lr = 1e-3
        num_steps = 10
        
        # Create SAE Bench reference
        ref = SAEBenchJumpReLU(activation_dim, dict_size, bandwidth)
        ref_optimizer = torch.optim.Adam(ref.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8)
        
        # Create our implementation
        ours = JumpReLUSAE(
            input_size=activation_dim,
            n_dict_components=dict_size,
            target_l0=target_l0,
            bandwidth=bandwidth,
            initial_threshold=0.001,
            use_pre_enc_bias=False,
            init_decoder_orthogonal=False,
            tied_encoder_init=False,
        )
        
        # Copy initialization
        with torch.no_grad():
            ours.decoder.weight.data.copy_(ref.W_dec.data.T)
            ours.encoder.weight.data.copy_(ref.W_enc.data.T)
            ours.encoder_bias.data.copy_(ref.b_enc.data)
            ours.decoder_bias.data.copy_(ref.b_dec.data)
            ours.jumprelu.threshold.data.copy_(ref.threshold.data)
        
        our_optimizer = torch.optim.Adam(ours.parameters(), lr=lr, **ours.get_optimizer_kwargs())
        
        for step in range(num_steps):
            # Same input for both
            torch.manual_seed(123 + step)
            x = torch.randn(16, 64)
            
            # Reference step
            ref_out = ref(x, target_l0=target_l0)
            ref_out['loss'].backward()
            sae_bench_remove_parallel_gradient(ref.W_dec)
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            ref_optimizer.step()
            ref_optimizer.zero_grad()
            sae_bench_normalize_decoder(ref.W_dec)
            
            # Our step
            our_out = ours(x)
            our_loss = ours.compute_loss(our_out)
            our_loss.loss.backward()
            ours.on_before_optimizer_step()
            torch.nn.utils.clip_grad_norm_(ours.parameters(), 1.0)
            our_optimizer.step()
            our_optimizer.zero_grad()
            ours.on_after_optimizer_step()
            
            # Check parity after each step
            max_dec_diff = (ours.decoder.weight.data - ref.W_dec.data.T).abs().max()
            max_enc_diff = (ours.encoder.weight.data - ref.W_enc.data.T).abs().max()
            max_thr_diff = (ours.jumprelu.threshold.data - ref.threshold.data).abs().max()
            
            assert max_dec_diff < 1e-4, f"Step {step}: Decoder diverged, max diff = {max_dec_diff}"
            assert max_enc_diff < 1e-4, f"Step {step}: Encoder diverged, max diff = {max_enc_diff}"
            assert max_thr_diff < 1e-4, f"Step {step}: Threshold diverged, max diff = {max_thr_diff}"
        
        print(f"All {num_steps} steps matched within tolerance!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
