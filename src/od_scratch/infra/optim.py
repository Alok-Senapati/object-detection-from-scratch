"""Adapted factories only; caller owns scheduling and parameter groups."""

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)


def make_optimizer(
    parameters: Iterable[torch.Tensor] | list[dict[str, Any]],
    name: str = "adamw",
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> Optimizer:
    """Create adamw, adam, or sgd with optional per-group overrides.

    The caller owns parameter selection, including excluding frozen parameters.
    Unknown names raise ValueError; optimizer-specific constraints come from PyTorch.

    Args:
        parameters: Parameter iterable or parameter-group dictionaries; exclude frozen weights
            as needed.
        name: Optimizer name: adamw, adam, or sgd.
        lr: Default learning rate, unless overridden per group.
        weight_decay: Default weight-decay coefficient.
        momentum: Momentum used by SGD; ignored by Adam and AdamW.

    Returns:
        A PyTorch optimizer over the supplied parameters.

    Raises:
        ValueError: The name is unknown or optimizer options are invalid.
    """
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(
    optimizer: Optimizer,
    name: str,
    *,
    epochs: int,
    min_lr: float = 0.0,
    step_size: int = 10,
    factor: float = 0.1,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Step once per epoch after optimizer; plateau needs validation loss.

    Args:
        optimizer: Optimizer whose learning rates will be scheduled.
        name: Scheduler name: none, cosine, step, or plateau.
        epochs: Positive epoch budget; sets the cosine period.
        min_lr: Minimum learning rate for cosine or plateau.
        step_size: Epoch interval for step decay.
        factor: Multiplicative decay factor for step or plateau.

    Returns:
        A scheduler, or None when name is none. Step after optimizer updates;
        plateau additionally requires the validation metric.

    Raises:
        ValueError: The name is unknown, epochs is nonpositive, or scheduler options are
            invalid.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=factor)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, factor=factor, min_lr=min_lr)
    raise ValueError(f"Unknown scheduler: {name}")
