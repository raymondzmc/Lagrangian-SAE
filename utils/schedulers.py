from typing import Callable
import math
from typing import Literal
import torch
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from utils.logging import logger


def get_linear_lr_schedule(
    warmup_samples: int,
    cooldown_samples: int,
    n_samples: int | None,
    effective_batch_size: int,
    min_lr_factor: float = 0.0,
) -> Callable[[int], float]:
    """
    Generates a linear learning rate schedule function that incorporates warmup and cooldown phases.
    If warmup_samples and cooldown_samples are both 0, the learning rate will be constant at 1.0
    throughout training.

    Args:
        warmup_samples: The number of samples to use for warmup.
        cooldown_samples: The number of samples to use for cooldown.
        effective_batch_size: The effective batch size used during training.
        min_lr_factor: The minimum learning rate as a fraction of the maximum learning rate. Used
            in the cooldown phase.

    Returns:
        A function that takes a training step as input and returns the corresponding learning rate.

    Raises:
        ValueError: If the cooldown period starts before the warmup period ends.
        AssertionError: If a cooldown is requested but the total number of samples is not provided.
    """
    warmup_steps = warmup_samples // effective_batch_size
    cooldown_steps = cooldown_samples // effective_batch_size

    if n_samples is None:
        assert cooldown_samples == 0, "Cooldown requested but total number of samples not provided."
        cooldown_start = float("inf")
    else:
        # NOTE: There may be 1 fewer steps if batch_size < effective_batch_size, but this won't
        # make a big difference for most learning setups. The + 1 is to account for the scheduler
        # step that occurs after training has finished
        total_steps = math.ceil(n_samples / effective_batch_size) + 1
        # Calculate the start step for cooldown
        cooldown_start = total_steps - cooldown_steps

        # Check for overlap between warmup and cooldown
        assert (
            cooldown_start > warmup_steps
        ), "Cooldown starts before warmup ends. Adjust your parameters."

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            # Warmup phase: linearly increase learning rate
            return (step + 1) / warmup_steps
        elif step >= cooldown_start:
            # Cooldown phase: linearly decrease learning rate
            # Calculate how many steps have been taken in the cooldown phase
            steps_into_cooldown = step - cooldown_start
            # Linearly decrease the learning rate
            return max(min_lr_factor, 1 - (steps_into_cooldown / cooldown_steps))
        else:
            # Maintain maximum learning rate after warmup and before cooldown
            return 1.0

    return lr_schedule


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_factor: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        min_lr_factor,
        min_lr_factor
        + (1 - min_lr_factor)
        * 0.5
        * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_factor: float = 0.0,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine
    function between the initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly between 0 and the initial lr set in the optimizer.

    The min_lr_factor is used to set a minimum learning rate that is a fraction of the initial
    learning rate.

    Adapted from `transformers.get_cosine_schedule_with_warmup` to support a minimum learning rate.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the
            max value to 0 following a half-cosine).
        min_lr_factor (`float`, *optional*, defaults to 0.0):
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_factor=min_lr_factor,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_exponential_beta_schedule(
    initial_beta: float,
    final_beta: float,
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    """Generates an exponential beta annealing schedule.

    Beta remains constant at `initial_beta` during warmup, then decays
    exponentially towards `final_beta`.

    Args:
        initial_beta: The starting value of beta.
        final_beta: The target value of beta after decay.
        warmup_steps: The number of steps during which beta remains constant.
        total_steps: The total number of training steps.

    Returns:
        A function mapping training step to the corresponding beta value.

    Raises:
        AssertionError: If total_steps <= warmup_steps.
    """
    assert total_steps > warmup_steps, "Total steps must be greater than warmup steps."
    decay_steps = total_steps - warmup_steps

    # Calculate gamma such that initial_beta * (gamma ** decay_steps) = final_beta
    if initial_beta == 0: # Avoid division by zero if initial_beta is 0
        gamma = 0.0
    elif final_beta == initial_beta: # Avoid 0^(1/inf) if betas are equal
        gamma = 1.0
    else:
        gamma = (final_beta / initial_beta) ** (1.0 / decay_steps)

    def beta_schedule(step: int) -> float:
        if step < warmup_steps:
            return initial_beta
        else:
            # Calculate steps into the decay phase
            steps_into_decay = step - warmup_steps
            # Apply exponential decay
            current_beta = initial_beta * (gamma ** steps_into_decay)
            # Ensure beta doesn't overshoot final_beta due to float precision
            # This depends on whether we are annealing up or down
            if initial_beta < final_beta:
                return min(current_beta, final_beta)
            else:
                return max(current_beta, final_beta)

    return beta_schedule


def get_sparsity_coeff_schedule(
    initial_coeff: float,
    final_coeff: float,
    warmup_steps: int,
    total_steps: int,
    schedule_type: Literal["linear", "exponential"],
) -> Callable[[int], float]:
    """Create a schedule for the sparsity coefficient."""
    if schedule_type == "linear":
        def scheduler(step: int) -> float:
            if step < warmup_steps:
                return initial_coeff
            progress = max(0, step - warmup_steps) / max(1, total_steps - warmup_steps)
            return initial_coeff + (final_coeff - initial_coeff) * progress
    elif schedule_type == "exponential":
        if initial_coeff == 0:
            initial_coeff = 1e-6

        # Handle non-positive initial or final coeffs (excluding initial_coeff == 0 handled above)
        if initial_coeff < 0 or final_coeff <= 0:
            logger.warning(
                "Exponential sparsity coefficient annealing requires positive initial (>0) and final (>0) coeffs. "
                "Falling back to linear schedule."
            )
            return get_sparsity_coeff_schedule(initial_coeff, final_coeff, warmup_steps, total_steps, "linear")

        # Standard exponential schedule for positive initial and final coeffs
        def scheduler(step: int) -> float:
            if step < warmup_steps:
                return initial_coeff
            progress = max(0, step - warmup_steps) / max(1, total_steps - warmup_steps)
            return initial_coeff * (final_coeff / initial_coeff) ** progress
    else:
        raise ValueError(f"Unknown sparsity coefficient schedule type: {schedule_type}")
    return scheduler
