"""Adapted from CIFAR Timer; synchronize for meaningful accelerator timing."""

import time
from types import TracebackType

import torch

from od_scratch.infra.device import synchronize


class Timer:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.elapsed = 0.0
        self.start = 0.0

    def __enter__(self) -> "Timer":
        if self.device is not None:
            synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.device is not None:
            synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start


def gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    norms = {
        f"grad_norm/{name}": float(p.grad.detach().norm().item())
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    norms["grad_norm/total"] = sum(value**2 for value in norms.values()) ** 0.5
    return norms


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


class WeightedMean:
    """Generic sample-weighted scalar statistic; no accuracy assumptions."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        if weight <= 0:
            raise ValueError("weight must be positive")
        self.total += value * weight
        self.count += weight

    @property
    def mean(self) -> float:
        if not self.count:
            raise ValueError("no observations")
        return self.total / self.count
