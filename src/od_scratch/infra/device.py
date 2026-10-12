"""Device selection and allocator statistics (not whole-system GPU usage)."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available or not available[requested]:
        raise ValueError(f"Unavailable or unsupported device: {requested}")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}  # Missing measurement is never reported as zero VRAM.
    mib = 1024**2
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / mib,
        "reserved_mib": torch.cuda.memory_reserved(device) / mib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / mib,
        "total_mib": torch.cuda.get_device_properties(device).total_memory / mib,
    }
