"""Device selection and allocator statistics (not whole-system GPU usage)."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA, then MPS, then CPU; reject unsupported or unavailable requests.

    Args:
        requested: auto, cpu, cuda, or mps; auto prefers CUDA, then MPS, then CPU.

    Returns:
        The selected available PyTorch device.

    Raises:
        ValueError: The requested device is unsupported or unavailable.
    """
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
    """Wait for queued CUDA/MPS work; CPU execution needs no explicit synchronization.

    Args:
        device: Device whose queued CUDA or MPS work must finish; CPU is a no-op.

    Returns:
        None.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_stats(device: torch.device) -> dict[str, float]:
    """Return CUDA allocator/device capacity in MiB, or no measurements for CPU/MPS.

    Args:
        device: Device to inspect.

    Returns:
        CUDA allocated, reserved, peak allocated, and total capacity values in MiB.
        CPU and MPS return an empty dictionary, not zero memory usage.
    """
    if device.type != "cuda":
        return {}  # Missing measurement is never reported as zero VRAM.
    mib = 1024**2
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / mib,
        "reserved_mib": torch.cuda.memory_reserved(device) / mib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / mib,
        "total_mib": torch.cuda.get_device_properties(device).total_memory / mib,
    }
