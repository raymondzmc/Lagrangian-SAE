import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def get_hook_shapes(tlens_model: HookedTransformer, hook_names: list[str]) -> dict[str, list[int]]:
    """Get the shapes of activations at the hook points labelled by hook_names"""
    # Sadly I can't see any way to easily get the shapes of activations at hook_points without
    # actually running the model.
    hook_shapes = {}

    def get_activation_shape_hook_function(activation: torch.Tensor, hook: HookPoint) -> None:
        hook_shapes[hook.name] = activation.shape

    def hook_names_filter(name: str) -> bool:
        return name in hook_names

    # Create test prompt on the same device as the model
    model_device = next(tlens_model.parameters()).device
    test_prompt = torch.tensor([0], device=model_device)
    
    # Ensure model is in eval mode and on correct device
    tlens_model.eval()
    tlens_model.to(model_device)
    
    tlens_model.run_with_hooks(
        test_prompt,
        return_type=None,
        fwd_hooks=[(hook_names_filter, get_activation_shape_hook_function)],
    )
    return hook_shapes