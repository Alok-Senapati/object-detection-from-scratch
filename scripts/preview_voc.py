from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from od_scratch.data.voc import parse_voc_annotation

BASE_DATA_PATH = Path(__file__).parents[1] / "data" / "VOCdevkit" / "VOC2007"
IMAGES_PATH = BASE_DATA_PATH / "JPEGImages"
ANNOTATIONS_PATH = BASE_DATA_PATH / "Annotations"
artifacts_path = Path(__file__).parents[1] / "artifacts" / "visualize"


def visualize_voc_image(xml_path: Path, save_path: bool = True) -> None:
    ann = parse_voc_annotation(xml_path)
    image_file_path = IMAGES_PATH / ann["filename"]
    with Image.open(image_file_path) as source:
        image = source.convert("RGB")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)

    names = [obj["name"] for obj in ann["objects"]]
    cmap = plt.get_cmap("tab10")
    colors = {n: cmap(i % 10) for i, n in enumerate(names)}

    for obj in ann["objects"]:
        xmin, ymin, xmax, ymax = obj["box_xyxy"]
        color = colors[obj["name"]]

        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label = obj["name"]
        if obj["truncated"]:
            label += " (T)"
        if obj["difficult"]:
            label += " (D)"

        ax.text(
            xmin,
            max(ymin - 4, 10),
            label,
            color="white",
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.85),
        )

    ax.set_title(f"{ann['filename']}  ({ann['width']}x{ann['height']})")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        artifacts_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_path / ann["filename"], dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    print(BASE_DATA_PATH)
    visualize_voc_image(ANNOTATIONS_PATH / "000043.xml")
