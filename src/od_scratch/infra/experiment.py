"""Unique run folders, immutable configuration, JSONL and TensorBoard scalars."""

import json
import logging
import math
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

import torch
from torch.utils.tensorboard import SummaryWriter

from od_scratch.infra.config import RunConfig


def git_metadata() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def console_logger() -> logging.Logger:
    logger = logging.getLogger("od_scratch")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class Experiment:
    def __init__(self, config: RunConfig, *, tensorboard: bool = True) -> None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        self.path = Path(config.artifact_root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        metadata = {
            "run_id": run_id,
            "config": config.model_dump(),
            "git": git_metadata(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
        }
        (self.path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.writer = SummaryWriter(str(self.path / "tensorboard")) if tensorboard else None

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if step < 0 or not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("Metrics must be finite and step nonnegative")
        event = {"time": datetime.now(UTC).isoformat(), "step": step, "metrics": metrics}
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        if self.writer is not None:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, step)
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
