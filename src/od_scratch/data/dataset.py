"""VOC image/target loading; geometry and augmentation remain separate stages."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from od_scratch.data.voc import parse_voc_annotation

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
Target = dict[str, Tensor]
Sample = tuple[Tensor, Target]
DATASET_BASEPATH = Path(__file__).parents[3] / "data" / "VOCdevkit" / "VOC2007"


class VOCDataset(Dataset[Sample]):
    """Read VOC images and annotations at their original resolution.

    Images are RGB float32 CHW tensors in [0, 1]. Labels are zero-based with no
    background class. Boxes use zero-based pixel-edge xyxy coordinates. Difficult
    and truncated objects are retained; resizing and augmentation are separate.
    Constructor arguments and validation errors are documented in __init__.

    Attributes:
        root: VOC directory containing images, annotations, and split manifests.
        ids: Image IDs in manifest order.
    """

    def __init__(
        self,
        root: str | Path = DATASET_BASEPATH,
        split: Literal["train", "val", "trainval", "test"] = "train",
    ) -> None:
        """Read the split manifest and verify referenced files exist.

        Args:
            root: VOC root directory; defaults to data/VOCdevkit/VOC2007 in this checkout.
            split: Official train, val, trainval, or test image manifest to read.

        Returns:
            None.

        Raises:
            ValueError: The split is unsupported or its IDs are empty, duplicated, or invalid.
            FileNotFoundError: The manifest or a referenced image/annotation is missing.
        """
        if split not in ["train", "val", "trainval", "test"]:
            raise ValueError(f"Unsupported VOC Split: {split}")
        self.root = Path(root)
        manifest = self.root / "ImageSets" / "Main" / f"{split}.txt"
        self.ids = tuple(line.strip() for line in manifest.read_text().splitlines() if line.strip())
        if not self.ids:
            raise ValueError(f"Empty split manifest: {manifest}")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"Duplicate image ids in {manifest}")
        for image_id in self.ids:
            if not all(char.isalnum() or char in "_-" for char in image_id):
                raise ValueError(f"Invalid image ID: {image_id!r}")
            for folder, suffix in (("JPEGImages", ".jpg"), ("Annotations", ".xml")):
                path = self.root / folder / f"{image_id}{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self) -> int:
        """Return the number of samples in the split.

        Returns:
            Number of image IDs in the split manifest.
        """
        return len(self.ids)

    def __getitem__(self, index: int) -> Sample:
        """Load one image and its validated, unfiltered annotation tensors.

        Args:
            index: Split position; negative positions follow Python sequence indexing.

        Returns:
            An RGB float32 image of shape (3, H, W) in [0, 1] and a target dictionary.
            Targets contain float32 boxes (N, 4), int64 labels (N,), boolean difficult
            and truncated flags (N,), scalar int64 image_id equal to index, and
            int64 orig_size [height, width]. Empty annotations use N = 0.

        Raises:
            IndexError: The index is outside the split.
            ValueError: Annotation values, class names, image dimensions, or filenames disagree.
            OSError: An image or annotation cannot be read.
            xml.etree.ElementTree.ParseError: The annotation XML is malformed.
        """
        image_id = self.ids[index]
        annotations_path = self.root / "Annotations" / f"{image_id}.xml"
        ann = parse_voc_annotation(annotations_path)
        if ann["filename"] != f"{image_id}.jpg":
            raise ValueError(f"{annotations_path}: filename does not match image ID")
        with Image.open(self.root / "JPEGImages" / f"{image_id}.jpg") as source:
            if source.size != (ann["width"], ann["height"]):
                raise ValueError(f"{image_id}: image dimensions disagree with annotation.")
            pixels = np.array(source.convert("RGB"), copy=True)
        image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div_(255)
        objects = ann["objects"]
        for obj in objects:
            if obj["name"] not in CLASS_TO_ID:
                raise ValueError(f"{annotations_path}: unknown class {obj['name']}")

        target = {
            "boxes": torch.tensor(
                [obj["box_xyxy"] for obj in objects], dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.tensor(
                [CLASS_TO_ID[obj["name"]] for obj in objects], dtype=torch.int64
            ),
            "difficult": torch.tensor([obj["difficult"] for obj in objects], dtype=torch.bool),
            "truncated": torch.tensor([obj["truncated"] for obj in objects], dtype=torch.bool),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "orig_size": torch.tensor([ann["height"], ann["width"]], dtype=torch.int64),
        }
        return image, target


def detection_collate(batch: list[Sample]) -> tuple[list[Tensor], list[Target]]:
    """Return image and target lists, preserving variable sizes and object counts.

    Args:
        batch: Image/target samples, including an optional empty batch.

    Returns:
        Parallel lists of images and targets in their original order, without stacking.
    """
    return [image for image, _ in batch], [target for _, target in batch]
