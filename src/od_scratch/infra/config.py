"""Validated TOML configuration with explicit, reproducible defaults."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    """Store validated run settings, rejecting unknown fields.

    Attributes:
        seed: RNG seed in [0, 2**32), defaulting to 42.
        deterministic: Whether to request deterministic algorithms; defaults to True.
        device: auto, cpu, cuda, or mps; defaults to auto.
        input_size: Planned square input size, at least 32; defaults to 320.
        batch_size: Positive batch size, defaulting to 2.
        num_workers: Nonnegative DataLoader worker count, defaulting to 0.
        artifact_root: Artifact directory, relative to the working directory unless
            absolute.

    Raises:
        pydantic.ValidationError: A field is invalid or an unknown key is supplied.
    """

    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    deterministic: bool = True
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    input_size: int = Field(default=320, ge=32)
    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=0, ge=0)
    artifact_root: str = "artifacts"


def load_config(path: Path) -> RunConfig:
    """Read TOML and validate settings, propagating parse and validation errors.

    Args:
        path: TOML configuration file.

    Returns:
        Validated run settings, with defaults for omitted fields.

    Raises:
        OSError: The file cannot be read.
        tomllib.TOMLDecodeError: The file is not valid TOML.
        pydantic.ValidationError: Settings contain unknown keys or invalid values.
    """
    with path.open("rb") as handle:
        return RunConfig.model_validate(tomllib.load(handle))
