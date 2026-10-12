"""Seed before constructing models/loaders or initializing CUDA."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
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
    """Pass as DataLoader worker_init_fn; also pass the returned generator."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
