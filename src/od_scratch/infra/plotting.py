"""Adapted scalar plotting pattern; no class-specific metrics or images."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_scalars(
    series: Mapping[str, Sequence[float]], path: Path, *, ylabel: str = "Value"
) -> None:
    """Save named scalar sequences against zero-based steps and close the figure.

    Args:
        series: Names mapped to scalar sequences, each plotted against zero-based steps.
        path: Output image path; parent directories are created.
        ylabel: Vertical-axis label.

    Returns:
        None.

    Raises:
        OSError: The output directory or figure cannot be written.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        for name, values in series.items():
            ax.plot(range(len(values)), values, label=name)
        ax.set(xlabel="Step", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
