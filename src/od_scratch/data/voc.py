from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CLASS_TO_ID = {name: index for index, name in enumerate(VOC_CLASSES)}


def parse_voc_annotation(path: Path) -> dict[str, Any]:
    """Parse one Pascal VOC annotation into a validated dictionary.

    VOC boxes use 1-based inclusive pixel coordinates. This function converts
    them to zero-based pixel-edge coordinates:

        [xmin - 1, ymin - 1, xmax, ymax]

    Under this convention:
        box_width = x2 - x1
        box_height = y2 - y1

    Objects retain their XML order. Difficult and truncated objects are kept,
    and an annotation without objects returns an empty objects list.

    Args:
        path: Path to a Pascal VOC XML annotation.

    Returns:
        A dictionary containing:
            filename: Image filename.
            width: Image width as a positive integer.
            height: Image height as a positive integer.
            objects: List of dictionaries containing name, box_xyxy,
                truncated and difficult.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        ET.ParseError: If the XML is malformed.
        ValueError: If required fields are missing, numeric values are invalid,
            dimensions are nonpositive, flags are not 0/1, or boxes are invalid.
    """

    def required_text(element: ET.Element, tag: str) -> str:
        """Read a required, nonempty child field."""
        text = element.findtext(tag)
        if text is None or not text.strip():
            raise ValueError(f"{path}: missing or empty <{tag}> inside <{element.tag}>.")
        return text.strip()

    def read_flag(element: ET.Element, tag: str) -> bool:
        """Read a binary flag, defaulting to False when absent."""
        value = element.findtext(tag, "0").strip()
        if value not in {"0", "1"}:
            raise ValueError(f"{path}: <{tag}> must be '0' or '1', got {value!r}.")
        return value == "1"

    root = ET.parse(path).getroot()
    filename = required_text(root, "filename")

    size = root.find("size")
    if size is None:
        raise ValueError(f"{path}: missing <size> element.")

    width = int(required_text(size, "width"))
    height = int(required_text(size, "height"))

    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: image dimensions must be positive, got {width}×{height}.")

    result: dict[str, Any] = {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": [],
    }

    for index, obj in enumerate(root.findall("object")):
        name = required_text(obj, "name")

        bbox = obj.find("bndbox")
        if bbox is None:
            raise ValueError(f"{path}: object {index} ({name!r}) is missing <bndbox>.")

        # Convert VOC's inclusive pixel indices to zero-based pixel edges.
        x1 = float(required_text(bbox, "xmin")) - 1
        y1 = float(required_text(bbox, "ymin")) - 1
        x2 = float(required_text(bbox, "xmax"))
        y2 = float(required_text(bbox, "ymax"))

        # Reject invalid boxes rather than silently clipping annotation errors.
        # These comparisons also reject NaN and infinite coordinates.
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"{path}: object {index} ({name!r}) has invalid box "
                f"{[x1, y1, x2, y2]} for image size {width}×{height}."
            )

        result["objects"].append(
            {
                "name": name,
                "box_xyxy": [x1, y1, x2, y2],
                "truncated": read_flag(obj, "truncated"),
                "difficult": read_flag(obj, "difficult"),
            }
        )

    return result


def load_sample(
    xml_path: str, images_dir: str, class_to_id: dict[str, int]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    annotations = parse_voc_annotation(Path(xml_path))
    image_filepath = Path(images_dir) / annotations["filename"]

    with Image.open(image_filepath) as src_img:
        if src_img.size != (annotations["width"], annotations["height"]):
            raise ValueError("Image dimensions disagree with annotation")
        pixels = np.array(src_img.convert("RGB"), copy=True)
    image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div_(255)

    target = {
        "boxes": torch.tensor(
            [obj["box_xyxy"] for obj in annotations["objects"]], dtype=torch.float32
        ).reshape(-1, 4),
        "labels": torch.tensor(
            [class_to_id[obj["name"]] for obj in annotations["objects"]], dtype=torch.int64
        ),
        "difficult": torch.tensor(
            [obj["difficult"] for obj in annotations["objects"]], dtype=torch.bool
        ),
        "truncated": torch.tensor(
            [obj["truncated"] for obj in annotations["objects"]], dtype=torch.bool
        ),
    }

    return image, target


if __name__ == "__main__":
    image, target = load_sample(
        "../../../data/VOCdevkit/VOC2007/Annotations/000042.xml",
        "../../../data/VOCdevkit/VOC2007/JPEGImages",
        CLASS_TO_ID,
    )
    print(target)
