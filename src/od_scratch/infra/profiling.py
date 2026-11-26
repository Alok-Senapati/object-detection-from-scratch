"""Adapted from CIFAR Timer; synchronize for meaningful accelerator timing."""

import time
from types import TracebackType

import torch

from od_scratch.infra.device import synchronize


class Timer:
    """Measure elapsed time with optional accelerator synchronization.

    Constructor arguments are documented in __init__.

    Attributes:
        device: Optional device synchronized at context entry and exit.
        elapsed: Duration in seconds, updated on context exit.
        start: Most recent perf_counter start timestamp in seconds.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """Initialize a timer with zero elapsed seconds.

        Args:
            device: Optional device to synchronize before and after timing.

        Returns:
            None.
        """
        self.device = device
        self.elapsed = 0.0
        self.start = 0.0

    def __enter__(self) -> "Timer":
        """Synchronize the device and begin timing.

        Returns:
            This timer instance, with its start timestamp reset.
        """
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
        """Synchronize the device and record elapsed seconds.

        Args:
            exc_type: Exception type from the context body, or None on success.
            exc: Exception instance from the context body, or None on success.
            tb: Exception traceback, or None on success.

        Returns:
            None. Exceptions from the context body are not suppressed.
        """
        if self.device is not None:
            synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start


def gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    """Return per-parameter L2 gradient norms and their combined L2 norm.

    Args:
        model: Model with gradients already computed.

    Returns:
        Per-parameter L2 norms under grad_norm/<name> and the combined grad_norm/total.
        Parameters without gradients are omitted; the total is zero if none exist.
    """
    norms = {
        f"grad_norm/{name}": float(p.grad.detach().norm().item())
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    norms["grad_norm/total"] = sum(value**2 for value in norms.values()) ** 0.5
    return norms


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Count total and requires_grad parameter elements, excluding buffers.

    Args:
        model: Model whose parameters are counted.

    Returns:
        Total and trainable parameter-element counts under total and trainable keys.
    """
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


class WeightedMean:
    """Accumulate a sample-weighted scalar mean.

    Attributes:
        total: Sum of each observation multiplied by its weight.
        count: Accumulated observation weight.
    """

    def __init__(self) -> None:
        """Initialize an empty weighted statistic.

        Returns:
            None.
        """
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        """Accumulate a weighted observation; nonpositive weights raise ValueError.

        Args:
            value: Scalar observation to accumulate.
            weight: Positive observation weight, usually a sample count.

        Returns:
            None.

        Raises:
            ValueError: The weight is nonpositive.
        """
        if weight <= 0:
            raise ValueError("weight must be positive")
        self.total += value * weight
        self.count += weight

    @property
    def mean(self) -> float:
        """Return the weighted mean, raising ValueError before any observations.

        Returns:
            Weighted sum divided by accumulated weight.

        Raises:
            ValueError: No observations have been added.
        """
        if not self.count:
            raise ValueError("no observations")
        return self.total / self.count
