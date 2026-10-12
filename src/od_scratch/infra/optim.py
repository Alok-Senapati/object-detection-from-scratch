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
    """Step once per epoch after optimizer; plateau needs validation loss."""
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
