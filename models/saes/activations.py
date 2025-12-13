# activations.py
"""
Shared activation functions and autograd primitives for SAEs.

Contains:
- RectangleFunction: Rectangle function with custom gradient (used in differentiable L0)
- StepFunction: Step function for L0 with gradient to learned log-thresholds
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
        return ((x > -0.5) & (x < 0.5)).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class StepFunction(autograd.Function):
    """Step function for L0 computation with gradient to learned log-thresholds.
    
    Used when thresholds are learned parameters (e.g., JumpReLU SAE).
    
    Forward: H(x - exp(log_threshold)) = 1 if x > exp(log_threshold), else 0
    Backward: Gradient only flows to log_threshold (not to x)
    
    Args:
        x: Pre-activations (after ReLU)
        log_threshold: Log of threshold (learned parameter, per-feature)
        bandwidth: Controls gradient width (smaller = sharper gradient)
    """
    
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        
        # No gradient w.r.t. x for step function (hard decision)
        x_grad = torch.zeros_like(x)
        
        # Gradient w.r.t. log_threshold using rectangle approximation
        # This encourages threshold to move to capture/release activations
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLUFunction(autograd.Function):
    """JumpReLU activation function with learned thresholds.
    
    Forward: x * (x > exp(log_threshold))
             Returns x if x > threshold, else 0 (hard sparsity)
    
    Backward: Gradient flows through for active units (x > threshold)
              Also provides gradient to log_threshold to learn optimal thresholds
    
    Args:
        x: Pre-activations (after ReLU)
        log_threshold: Log of threshold (learned parameter, per-feature)
        bandwidth: Controls gradient width for threshold learning
    """
    
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        
        # Gradient w.r.t. x: pass through for active units
        x_grad = (x > threshold).float() * grad_output
        
        # Gradient w.r.t. log_threshold: encourage threshold to capture/release activations
        # Multiply by threshold for log-scale gradient (chain rule: d/d(log_t) = t * d/dt)
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLU(nn.Module):
    """JumpReLU activation module with learnable per-feature thresholds.
    
    Applies JumpReLU activation: x * (x > threshold) where threshold is learned.
    The thresholds are stored in log-space for numerical stability and to ensure
    they remain positive.
    
    Args:
        feature_size: Number of features (dictionary components)
        bandwidth: Controls gradient width for threshold learning (default: 0.01)
        initial_threshold: Initial threshold value (default: 1.0, stored as log)
    """
    
    def __init__(
        self, 
        feature_size: int, 
        bandwidth: float = 0.01, 
        initial_threshold: float = 1.0,
        device: str | torch.device = 'cpu'
    ):
        super(JumpReLU, self).__init__()
        # Store thresholds in log-space (initialized to log(initial_threshold))
        self.log_threshold = nn.Parameter(
            torch.full((feature_size,), fill_value=torch.log(torch.tensor(initial_threshold)).item(), device=device)
        )
        self.bandwidth = bandwidth
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JumpReLU activation.
        
        Args:
            x: Input tensor of shape (..., feature_size)
            
        Returns:
            Output tensor of same shape with JumpReLU applied
        """
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)
    
    @property
    def threshold(self) -> torch.Tensor:
        """Get current threshold values (in linear space)."""
        return torch.exp(self.log_threshold)
