"""Exercise the real image/XML/manifest boundary without downloading VOC."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from od_scratch.data.dataset import VOCDataset, detection_collate


def make_voc(root: Path) -> Path:
    for folder in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (root / folder).mkdir(parents=True)
    for image_id, size, objects in (
        (
            "000001",
            (10, 8),
            "<object><name>cat</name><difficult>1</difficult>"
            "<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>8</ymax>"
            "</bndbox></object>",
        ),
        ("000002", (6, 4), ""),
    ):
        Image.new("RGB", size, "white").save(root / "JPEGImages" / f"{image_id}.jpg")
        (root / "Annotations" / f"{image_id}.xml").write_text(
            f"<annotation><filename>{image_id}.jpg</filename><size>"
            f"<width>{size[0]}</width><height>{size[1]}</height></size>{objects}</annotation>"
        )
    (root / "ImageSets/Main/train.txt").write_text("000001\n000002\n")
    return root


def test_loading_and_variable_size_batch(tmp_path: Path) -> None:
    dataset = VOCDataset(make_voc(tmp_path))
    images, targets = next(iter(DataLoader(dataset, batch_size=2, collate_fn=detection_collate)))
    assert len(dataset) == 2
    assert [tuple(image.shape) for image in images] == [(3, 8, 10), (3, 4, 6)]
    assert images[0].dtype == torch.float32
    assert images[0].min() == images[0].max() == 1
    assert targets[0]["boxes"].tolist() == [[0, 0, 10, 8]]
    assert targets[0]["boxes"].dtype == torch.float32
    assert targets[0]["labels"].tolist() == [7]
    assert targets[0]["labels"].dtype == torch.int64
    assert targets[0]["difficult"].tolist() == [True]
    assert targets[0]["truncated"].tolist() == [False]
    assert targets[1]["boxes"].shape == (0, 4)
    assert targets[1]["labels"].shape == (0,)
    assert targets[1]["difficult"].dtype == torch.bool
    assert targets[1]["orig_size"].tolist() == [4, 6]
    assert [target["image_id"].item() for target in targets] == [0, 1]


@pytest.mark.parametrize("contents", ["", "000001\n000001\n", "../escape\n"])
def test_bad_manifests(tmp_path: Path, contents: str) -> None:
    root = make_voc(tmp_path)
    (root / "ImageSets/Main/train.txt").write_text(contents)
    with pytest.raises(ValueError):
        VOCDataset(root)


def test_missing_image(tmp_path: Path) -> None:
    root = make_voc(tmp_path)
    (root / "JPEGImages/000002.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        VOCDataset(root)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("<width>10</width>", "<width>11</width>", "dimensions"),
        ("<name>cat</name>", "<name>unknown</name>", "unknown class"),
        ("000001.jpg", "other.jpg", "filename"),
    ],
)
def test_annotation_image_disagreement(tmp_path: Path, old: str, new: str, message: str) -> None:
    root = make_voc(tmp_path)
    path = root / "Annotations/000001.xml"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        VOCDataset(root)[0]
