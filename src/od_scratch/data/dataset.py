"""VOC image/target loading; geometry and augmentation remain separate stages."""

from pathlib import Path

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


class VOCDataset(Dataset[Sample]):
    """Read a VOC root containing JPEGImages, Annotations and ImageSets/Main.

    Images are RGB float32 CHW in [0, 1], at original resolution. Labels are
    zero-based (no background class). Boxes use zero-based pixel-edge xyxy.
    Both difficult and truncated flags are preserved; no objects are filtered.
    This loader does not resize, normalize, augment or pad images.
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        if split not in {"train", "val", "trainval", "test"}:
            raise ValueError(f"Unsupported VOC split: {split!r}")
        self.root = Path(root)
        manifest = self.root / "ImageSets" / "Main" / f"{split}.txt"
        self.ids = tuple(line.strip() for line in manifest.read_text().splitlines() if line.strip())
        if not self.ids:
            raise ValueError(f"Empty split manifest: {manifest}")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"Duplicate image IDs in {manifest}")
        for image_id in self.ids:
            if not all(char.isalnum() or char in "_-" for char in image_id):
                raise ValueError(f"Invalid image ID: {image_id!r}")
            for folder, suffix in (("JPEGImages", ".jpg"), ("Annotations", ".xml")):
                path = self.root / folder / f"{image_id}{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Sample:
        image_id = self.ids[index]
        annotation_path = self.root / "Annotations" / f"{image_id}.xml"
        ann = parse_voc_annotation(annotation_path)
        if ann["filename"] != f"{image_id}.jpg":
            raise ValueError(f"{annotation_path}: filename does not match image ID")
        with Image.open(self.root / "JPEGImages" / f"{image_id}.jpg") as source:
            if source.size != (ann["width"], ann["height"]):
                raise ValueError(f"{image_id}: image dimensions disagree with annotation")
            pixels = np.array(source.convert("RGB"), copy=True)
        image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div_(255)
        objects = ann["objects"]
        for obj in objects:
            if obj["name"] not in CLASS_TO_ID:
                raise ValueError(f"{annotation_path}: unknown class {obj['name']!r}")
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
    """Keep variable image sizes and object counts; stack only after resizing."""
    return [image for image, _ in batch], [target for _, target in batch]
