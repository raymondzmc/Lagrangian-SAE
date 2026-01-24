# activations.py
"""
Shared activation functions and autograd primitives for SAEs.

Contains:
- RectangleFunction: Rectangle function with custom gradient (used in differentiable L0)
- StepFunction: Step function for L0 with gradient to learned thresholds
- JumpReLUFunction: JumpReLU activation with learned thresholds
- JumpReLU: nn.Module wrapper for JumpReLU activation
"""
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch import nn
from typing import Callable


def softplus0(x: torch.Tensor) -> torch.Tensor:
    """Softplus shifted to pass through origin: softplus(x) - softplus(0)."""
    return F.softplus(x) - F.softplus(torch.zeros((), device=x.device, dtype=x.dtype))


def get_activation(activation: str | None = None) -> Callable:
    """Get activation function by name."""
    ACTIVATION_MAP: dict[str, Callable] = {
        'relu': F.relu,
        'softplus': F.softplus,
        'softplus0': softplus0,
        'none': torch.nn.Identity(),
    }
    if activation is None:
        return torch.nn.Identity()
    else:
        return ACTIVATION_MAP[activation]


class RectangleFunction(autograd.Function):
    """Rectangle function with custom gradient for use in differentiable L0.
    
    Forward: returns 1 if -0.5 < x < 0.5, else 0
    Backward: passes gradient through unchanged in the valid region
    
    This is a building block for smooth step function gradients. The rectangle
    function approximates the derivative of a smooth step (sigmoid) at the threshold.
    """
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class StepFunction(autograd.Function):
    """Step function for L0 computation with gradient to learned thresholds.
    
    Used when thresholds are learned parameters (e.g., JumpReLU SAE).
    
    Forward: H(x - threshold) = 1 if x > threshold, else 0
    Backward: Gradient only flows to threshold (not to x).
    
    Args:
        x: Pre-activations (raw encoder outputs, can be negative)
        threshold: Threshold value (raw, not log-space)
        bandwidth: Controls gradient width (smaller = sharper gradient)
    """
    
    @staticmethod
    def forward(ctx, x, threshold, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)
        return (x > threshold).to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bw = ctx.bandwidth
        
        # No gradient w.r.t. x for step function (hard decision)
        x_grad = torch.zeros_like(x)
        
        # Gradient w.r.t. threshold (∂L/∂θ)
        rect = (((x - threshold) / bw > -0.5) & ((x - threshold) / bw < 0.5)).to(grad_output.dtype)
        threshold_grad = -(1.0 / bw) * rect * grad_output
        
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLUFunction(autograd.Function):
    """JumpReLU activation function with learned thresholds.
    
    Forward: x * (x > threshold)
             Returns x if x > threshold, else 0 (hard sparsity)
    
    Backward: Gradient flows through for active units (x > threshold).
              Gradient also flows to threshold parameter.
    
    Args:
        x: Pre-activations (raw encoder outputs, can be negative)
        threshold: Threshold value (raw, not log-space)
        bandwidth: Controls gradient width for threshold learning
    """
    
    @staticmethod
    def forward(ctx, x, threshold, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)
        return x * (x > threshold).to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bw = ctx.bandwidth
        
        # Gradient w.r.t. x: pass through for active units
        x_grad = (x > threshold).to(grad_output.dtype) * grad_output
        
        # Gradient w.r.t. threshold (∂L/∂θ)
        rect = (((x - threshold) / bw > -0.5) & ((x - threshold) / bw < 0.5)).to(grad_output.dtype)
        threshold_grad = -(threshold / bw) * rect * grad_output
        
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLU(nn.Module):
    """JumpReLU activation module with learnable per-feature thresholds.
    
    Applies JumpReLU activation: x * (x > threshold) where threshold is learned.
    Uses raw threshold parameterization (not log-space) following SAE Bench.
    
    Args:
        feature_size: Number of features (dictionary components)
        bandwidth: Controls gradient width for threshold learning (default: 0.001)
        initial_threshold: Initial threshold value (default: 0.001)
        device: Device to create parameters on
    """
    
    def __init__(
        self, 
        feature_size: int, 
        bandwidth: float = 0.001, 
        initial_threshold: float = 0.001,
        device: str | torch.device = 'cpu'
    ):
        super().__init__()
        self.threshold = nn.Parameter(
            torch.ones(feature_size, device=device) * initial_threshold
        )
        self.bandwidth = bandwidth
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JumpReLU activation.
        
        Args:
            x: Input tensor of shape (..., feature_size)
            
        Returns:
            Output tensor of same shape with JumpReLU applied
        """
        return JumpReLUFunction.apply(x, self.threshold, self.bandwidth)
