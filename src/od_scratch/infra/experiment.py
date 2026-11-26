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
    """Read the current checkout's commit and dirty state, or None outside Git.

    Returns:
        A dictionary with commit and dirty keys. Both values are None if Git metadata
        cannot be read from the current working directory.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def console_logger() -> logging.Logger:
    """Return the shared INFO logger without adding duplicate console handlers.

    Returns:
        The shared od_scratch INFO logger with one configured console handler.
    """
    logger = logging.getLogger("od_scratch")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class Experiment:
    """Persist run provenance and scalar events in a unique directory.

    Use as a context manager to close the optional TensorBoard writer reliably.
    Constructor arguments are documented in __init__.

    Attributes:
        path: Unique run directory under the configured artifact root.
        writer: TensorBoard SummaryWriter, or None when TensorBoard is disabled.
    """

    def __init__(self, config: RunConfig, *, tensorboard: bool = True) -> None:
        """Create a unique artifact directory and record run provenance.

        Args:
            config: Validated run settings to record in metadata.json.
            tensorboard: Create a TensorBoard writer in addition to JSONL logging when True.

        Returns:
            None.

        Raises:
            OSError: The run directory or metadata cannot be created.
        """
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
        """Append and flush scalars; reject negative steps and nonfinite values.

        Args:
            step: Nonnegative training or evaluation step.
            metrics: Scalar names mapped to finite values.

        Returns:
            None.

        Raises:
            ValueError: The step is negative or any metric is nonfinite.
            OSError: The JSONL event cannot be written.
        """
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
        """Close the TensorBoard writer when enabled.

        Returns:
            None.
        """
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> "Experiment":
        """Enter the experiment context.

        Returns:
            This experiment instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the writer when leaving the context.

        Args:
            exc_type: Exception type from the context body, or None on success.
            exc: Exception instance from the context body, or None on success.
            tb: Exception traceback, or None on success.

        Returns:
            None. Exceptions from the context body are not suppressed.
        """
        self.close()
