"""Seed before constructing models/loaders or initializing CUDA."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed Python, NumPy and PyTorch and return a seeded DataLoader generator.

    Updates process-wide determinism settings. The seed must be in [0, 2**32).
    Deterministic execution can reject unsupported kernels; it does not guarantee
    identical results across hardware or library versions.

    Args:
        seed: Seed in [0, 2**32) shared by Python, NumPy, and PyTorch.
        deterministic: Enable strict deterministic algorithms and disable cuDNN benchmarking
            when True.

    Returns:
        A separately seeded torch.Generator for use with a DataLoader.

    Raises:
        ValueError: The seed is outside [0, 2**32).
    """
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    return torch.Generator().manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed worker Python/NumPy RNGs; pair with seed_everything's loader generator.

    Args:
        worker_id: DataLoader worker index; accepted for the callback API but not used directly.

    Returns:
        None.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
