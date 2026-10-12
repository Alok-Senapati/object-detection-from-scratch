"""CLI environment probe; makes no model or dataset downloads."""

import json
import platform
from typing import Any

import torch

from od_scratch.infra.device import memory_stats, select_device


def diagnose() -> dict[str, Any]:
    device = select_device()
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "memory": memory_stats(device),
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
    return result


def main() -> None:
    print(json.dumps(diagnose(), indent=2))
