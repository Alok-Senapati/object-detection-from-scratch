"""Exercise VOC loading with tiny local fixtures, without downloading a dataset."""

from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from od_scratch.data.dataset import CLASS_TO_ID, VOCDataset, detection_collate
from od_scratch.data.voc import load_sample


def write_sample(
    root: Path,
    image_id: str = "sample",
    size: tuple[int, int] = (10, 8),
    objects: bool = True,
    mode: str = "RGB",
) -> Path:
    for folder in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, 255).save(root / "JPEGImages" / f"{image_id}.jpg")
    object_xml = (
        "<object><name>cat</name><difficult>1</difficult><bndbox>"
        f"<xmin>1</xmin><ymin>1</ymin><xmax>{size[0]}</xmax><ymax>{size[1]}</ymax>"
        "</bndbox></object>"
        "<object><name>person</name><truncated>1</truncated><bndbox>"
        "<xmin>2</xmin><ymin>3</ymin><xmax>2</xmax><ymax>3</ymax></bndbox></object>"
        if objects
        else ""
    )
    path = root / "Annotations" / f"{image_id}.xml"
    path.write_text(
        f"<annotation><filename>{image_id}.jpg</filename><size>"
        f"<width>{size[0]}</width><height>{size[1]}</height></size>"
        f"{object_xml}</annotation>",
        encoding="utf-8",
    )
    (root / "ImageSets/Main/train.txt").write_text(f"{image_id}\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["RGB", "L"])
@pytest.mark.parametrize("empty", [False, True])
def test_sample_contract(tmp_path: Path, mode: str, empty: bool) -> None:
    xml = write_sample(tmp_path, objects=not empty, mode=mode)
    dataset = VOCDataset(str(tmp_path))
    image, target = dataset[0]
    standalone_image, standalone_target = load_sample(
        str(xml), str(tmp_path / "JPEGImages"), CLASS_TO_ID
    )
    assert len(dataset) == 1
    assert image.shape == (3, 8, 10)
    assert image.dtype == torch.float32
    assert image.is_contiguous()
    assert 0 <= image.min() <= image.max() <= 1
    if mode == "L":
        assert torch.equal(image, torch.ones_like(image))
    assert torch.equal(image, standalone_image)
    for key, value in standalone_target.items():
        assert torch.equal(target[key], value)
    count = 0 if empty else 2
    assert target["boxes"].shape == (count, 4)
    assert target["boxes"].dtype == torch.float32
    for key, dtype in (
        ("labels", torch.int64),
        ("difficult", torch.bool),
        ("truncated", torch.bool),
    ):
        assert target[key].shape == (count,)
        assert target[key].dtype == dtype
    assert target["image_id"].shape == ()
    assert target["image_id"].dtype == torch.int64
    assert target["image_id"].item() == 0
    assert target["orig_size"].tolist() == [8, 10]
    assert target["orig_size"].dtype == torch.int64
    if not empty:
        assert target["boxes"].tolist() == [[0, 0, 10, 8], [1, 2, 2, 3]]
        assert target["labels"].tolist() == [7, 14]
        assert target["difficult"].tolist() == [True, False]
        assert target["truncated"].tolist() == [False, True]


def test_batch_preserves_variable_shapes_and_manifest_order(tmp_path: Path) -> None:
    write_sample(tmp_path, "first")
    write_sample(tmp_path, "second", (5, 4), objects=False)
    (tmp_path / "ImageSets/Main/train.txt").write_text(" second \n\nfirst\n", encoding="utf-8")
    dataset = VOCDataset(tmp_path)
    images, targets = next(iter(DataLoader(dataset, batch_size=2, collate_fn=detection_collate)))
    assert [tuple(image.shape) for image in images] == [(3, 4, 5), (3, 8, 10)]
    assert [tuple(target["boxes"].shape) for target in targets] == [(0, 4), (2, 4)]
    assert [target["image_id"].item() for target in targets] == [0, 1]
    assert detection_collate([]) == ([], [])
    with pytest.raises(IndexError):
        dataset[2]


@pytest.mark.parametrize("split", ["train", "val", "trainval", "test"])
def test_supported_splits(tmp_path: Path, split: str) -> None:
    write_sample(tmp_path)
    (tmp_path / f"ImageSets/Main/{split}.txt").write_text("sample\n", encoding="utf-8")
    assert len(VOCDataset(tmp_path, cast(Literal["train", "val", "trainval", "test"], split))) == 1


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("\n ", "Empty split"),
        ("sample\nsample", "Duplicate"),
        ("../sample", "Invalid image ID"),
        ("sample 1", "Invalid image ID"),
    ],
)
def test_invalid_manifest(tmp_path: Path, manifest: str, message: str) -> None:
    write_sample(tmp_path)
    (tmp_path / "ImageSets/Main/train.txt").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        VOCDataset(tmp_path)


@pytest.mark.parametrize(
    "missing", ["JPEGImages/sample.jpg", "Annotations/sample.xml", "ImageSets/Main/train.txt"]
)
def test_missing_files(tmp_path: Path, missing: str) -> None:
    write_sample(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError):
        VOCDataset(tmp_path)


def test_unsupported_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported VOC Split"):
        VOCDataset(tmp_path, cast(Literal["train"], "invalid"))


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        ("sample.jpg", "other.jpg", "filename"),
        ("<width>10", "<width>11", "dimensions"),
        ("<name>cat", "<name>unknown", "unknown class"),
    ],
)
def test_annotation_image_disagreement(
    tmp_path: Path, before: str, after: str, message: str
) -> None:
    xml = write_sample(tmp_path)
    xml.write_text(xml.read_text().replace(before, after), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        VOCDataset(tmp_path)[0]


def test_load_sample_custom_labels_and_size_validation(tmp_path: Path) -> None:
    xml = write_sample(tmp_path)
    _, target = load_sample(str(xml), str(tmp_path / "JPEGImages"), {"cat": 42, "person": 99})
    assert target["labels"].tolist() == [42, 99]
    Image.new("RGB", (12, 8)).save(tmp_path / "JPEGImages/sample.jpg")
    with pytest.raises(ValueError, match="dimensions"):
        load_sample(str(xml), str(tmp_path / "JPEGImages"), CLASS_TO_ID)
