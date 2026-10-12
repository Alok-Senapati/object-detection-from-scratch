"""Validated TOML configuration with explicit, reproducible defaults."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    deterministic: bool = True
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    input_size: int = Field(default=320, ge=32)
    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=0, ge=0)
    artifact_root: str = "artifacts"


def load_config(path: Path) -> RunConfig:
    with path.open("rb") as handle:
        return RunConfig.model_validate(tomllib.load(handle))
