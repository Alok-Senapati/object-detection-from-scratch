"""Atomic tensor/state checkpoints; caller supplies architecture and resume state."""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path, model: torch.nn.Module, *, training_state: dict[str, Any] | None = None
) -> None:
    """Atomically save model weights and caller-owned training state.

    Exact resume requires the caller to capture and restore all RNG and loader
    state; this helper does not guarantee exact resume.

    Args:
        path: Destination checkpoint path; parent directories are created.
        model: Model whose state dictionary is saved.
        training_state: Optional serializable optimizer, scheduler, epoch, RNG, or other resume
            state.

    Returns:
        None.

    Raises:
        OSError: The checkpoint cannot be written or atomically replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(
            {
                "schema_version": 1,
                "model_state": model.state_dict(),
                "training_state": training_state or {},
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: Path, model: torch.nn.Module, *, strict: bool = True) -> dict[str, Any]:
    """Load weights onto the supplied model and return caller-owned training state.

    Args:
        path: Checkpoint produced by save_checkpoint.
        model: Model to receive the saved weights.
        strict: Require matching state keys; tensor shapes must match even when False.

    Returns:
        The saved training-state dictionary; optimizer and RNG restoration remain the caller's
        responsibility.

    Raises:
        ValueError: The checkpoint schema or training state is invalid.
        RuntimeError: Model keys or tensor shapes are incompatible with the checkpoint.
        OSError: The checkpoint cannot be read.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    # strict=False tolerates missing/unexpected keys, NOT mismatched tensor shapes.
    model.load_state_dict(payload["model_state"], strict=strict)
    state = payload["training_state"]
    if not isinstance(state, dict):
        raise ValueError("Invalid training state")
    return state
