"""Render VOC image batches and pixel-edge boxes without changing their targets."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch import Tensor

from od_scratch.data.dataset import VOC_CLASSES, Target

ARTIFACTS_PATH = Path(__file__).parents[3] / "artifacts" / "visualize"


def visualize_batch(
    images: list[Tensor],
    targets: list[Target],
    save_filename: str | None = None,
    *,
    output_dir: Path = ARTIFACTS_PATH,
    show: bool = True,
) -> None:
    """Draw a variable-size batch with class names and T/D annotation flags.

    Inputs are detached and copied to CPU for plotting. All created figures are
    closed, including when rendering or saving raises an exception.

    Args:
        images: RGB CHW tensors in [0, 1], as returned by VOCDataset.
        targets: Matching targets with pixel-edge xyxy boxes and zero-based labels.
        save_filename: Optional output filename; Matplotlib adds .png if omitted.
        output_dir: Directory created when saving a figure.
        show: Display the figure interactively; disable for headless execution.

    Returns:
        None.

    Raises:
        ValueError: The batch is empty, counts disagree, or a label is out of range.
        OSError: The output directory or figure cannot be written.
    """
    if not images or len(images) != len(targets):
        raise ValueError("Expected a nonempty batch with matching images and targets")
    for target in targets:
        count = len(target["boxes"])
        if target["boxes"].shape != (count, 4) or any(
            target[key].shape != (count,) for key in ("labels", "truncated", "difficult")
        ):
            raise ValueError("Boxes, labels and flags must have matching object counts")
        if any(label < 0 or label >= len(VOC_CLASSES) for label in target["labels"].tolist()):
            raise ValueError("Label is outside the VOC class range")

    cols = min(3, len(images))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    try:
        axes = axes.flatten()
        cmap = plt.get_cmap("tab20")
        for ax, image, target in zip(axes, images, targets, strict=False):
            height, width = image.shape[-2:]
            # Align image boundaries with the dataset's pixel-edge box coordinates.
            ax.imshow(image.detach().cpu().permute(1, 2, 0).numpy(), extent=(0, width, height, 0))
            for box, class_id, truncated, difficult in zip(
                target["boxes"].detach().cpu().tolist(),
                target["labels"].tolist(),
                target["truncated"].tolist(),
                target["difficult"].tolist(),
                strict=True,
            ):
                xmin, ymin, xmax, ymax = box
                color = cmap(class_id)
                ax.add_patch(
                    Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                    )
                )
                label = VOC_CLASSES[class_id]
                if truncated:
                    label += " (T)"
                if difficult:
                    label += " (D)"
                ax.text(
                    xmin,
                    ymin,
                    label,
                    color="white",
                    fontsize=10,
                    weight="bold",
                    bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.85),
                )
            orig_height, orig_width = target["orig_size"].tolist()
            ax.set_title(f"{target['image_id'].item()} ({orig_width}x{orig_height})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        if save_filename:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / save_filename, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
    finally:
        plt.close(fig)
