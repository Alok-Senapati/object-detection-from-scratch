"""Atomic tensor/state checkpoints; caller supplies architecture and resume state."""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path, model: torch.nn.Module, *, training_state: dict[str, Any] | None = None
) -> None:
    """State can include optimizer, scheduler, scaler, epoch and RNG tensors.

    Exact resume requires caller to capture/restore ALL RNGs and loader state.
    This helper deliberately makes no exact-resume guarantee.
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
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    # strict=False tolerates missing/unexpected keys, NOT mismatched tensor shapes.
    model.load_state_dict(payload["model_state"], strict=strict)
    state = payload["training_state"]
    if not isinstance(state, dict):
        raise ValueError("Invalid training state")
    return state
