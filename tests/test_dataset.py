"""Exercise VOC loading with tiny local fixtures, without downloading a dataset."""

from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from od_scratch.data.dataset import VOCDataset, detection_collate
from od_scratch.geometry.boxes import (
    box_area,
    box_iou,
    clip_xyxy,
    cxcywh_to_xyxy,
    denormalize_xyxy,
    normalize_xyxy,
    xyxy_to_cxcywh,
)


def write_sample(
    root: Path,
    image_id: str = "sample",
    size: tuple[int, int] = (10, 8),
    objects: bool = True,
    mode: str = "RGB",
) -> Path:
    """Write a small image, annotation, and train manifest for dataset tests.

    Args:
        root: Temporary VOC root to populate.
        image_id: Shared image and annotation file stem.
        size: Image dimensions as (width, height).
        objects: Include cat and person objects when True.
        mode: Pillow image mode, such as RGB or L.

    Returns:
        Path to the generated annotation XML.
    """
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


def test_loaded_boxes_geometry_roundtrip(tmp_path: Path) -> None:
    """Preserve VOC pixel-edge boxes through the geometry processing pipeline.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
    write_sample(tmp_path)
    image, target = VOCDataset(tmp_path)[0]
    height, width = image.shape[-2:]
    boxes = target["boxes"]
    normalized = normalize_xyxy(boxes, width, height)
    restored = denormalize_xyxy(cxcywh_to_xyxy(xyxy_to_cxcywh(normalized)), width, height)
    clipped = clip_xyxy(restored, width, height)
    expected = torch.tensor([[0.0, 0.0, 10.0, 8.0], [1.0, 2.0, 2.0, 3.0]])
    torch.testing.assert_close(clipped, expected)
    torch.testing.assert_close(boxes, expected)
    torch.testing.assert_close(box_area(clipped), torch.tensor([80.0, 1.0]))
    torch.testing.assert_close(
        box_iou(clipped, boxes), torch.tensor([[1.0, 1 / 80], [1 / 80, 1.0]])
    )


@pytest.mark.parametrize("mode", ["RGB", "L"])
@pytest.mark.parametrize("empty", [False, True])
def test_sample_contract(tmp_path: Path, mode: str, empty: bool) -> None:
    """Preserve image and target shapes, types, coordinates, and annotation flags.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        mode: Pillow image mode under test.
        empty: Whether to exercise the empty-data variant.

    Returns:
        None.
    """
    write_sample(tmp_path, objects=not empty, mode=mode)
    dataset = VOCDataset(str(tmp_path))
    image, target = dataset[0]
    assert len(dataset) == 1
    assert image.shape == (3, 8, 10)
    assert image.dtype == torch.float32
    assert image.is_contiguous()
    assert 0 <= image.min() <= image.max() <= 1
    if mode == "L":
        assert torch.equal(image, torch.ones_like(image))
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
    """Batch variable-sized samples in manifest order without padding.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
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
    """Load each supported VOC manifest name.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        split: VOC manifest name under test.

    Returns:
        None.
    """
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
    """Reject empty, duplicate, or malformed image IDs in a split manifest.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        manifest: Manifest text used to exercise validation.
        message: Expected validation-error message or pattern.

    Returns:
        None.
    """
    write_sample(tmp_path)
    (tmp_path / "ImageSets/Main/train.txt").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        VOCDataset(tmp_path)


@pytest.mark.parametrize(
    "missing", ["JPEGImages/sample.jpg", "Annotations/sample.xml", "ImageSets/Main/train.txt"]
)
def test_missing_files(tmp_path: Path, missing: str) -> None:
    """Reject a split with a missing manifest, image, or annotation.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        missing: Relative fixture path to remove.

    Returns:
        None.
    """
    write_sample(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError):
        VOCDataset(tmp_path)


def test_unsupported_split(tmp_path: Path) -> None:
    """Reject unsupported split names before reading files.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
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
    """Reject inconsistent annotation metadata or unknown classes.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        before: Annotation text to replace.
        after: Replacement annotation text.
        message: Expected validation-error message or pattern.

    Returns:
        None.
    """
    xml = write_sample(tmp_path)
    xml.write_text(xml.read_text().replace(before, after), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        VOCDataset(tmp_path)[0]


def test_dataset_rejects_incorrect_image_size(tmp_path: Path) -> None:
    """Reject image dimensions that disagree with the annotation.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
    write_sample(tmp_path)
    Image.new("RGB", (12, 8)).save(tmp_path / "JPEGImages/sample.jpg")
    with pytest.raises(ValueError, match="dimensions"):
        VOCDataset(tmp_path)[0]


@pytest.mark.parametrize("pixel", [0, 64, 128, 255])
def test_dataset_scales_pixels_to_unit_range(tmp_path: Path, pixel: int) -> None:
    """Convert byte pixel intensities to the expected floating-point values.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        pixel: Uniform pixel intensity in the fixture image.

    Returns:
        None.
    """
    write_sample(tmp_path)
    Image.new("RGB", (10, 8), (pixel, pixel, pixel)).save(tmp_path / "JPEGImages/sample.jpg")
    image, _ = VOCDataset(tmp_path)[0]
    torch.testing.assert_close(image, torch.full((3, 8, 10), pixel / 255))
