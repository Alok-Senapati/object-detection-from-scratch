"""Validate VOC coordinate conversion and malformed annotation handling."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from od_scratch.data.voc import parse_voc_annotation


def annotation(tmp_path: Path, objects: str = "", size: str = "10") -> Path:
    path = tmp_path / "sample.xml"
    path.write_text(
        f"<annotation><filename>sample.jpg</filename><size><width>{size}</width>"
        f"<height>8</height></size>{objects}</annotation>",
        encoding="utf-8",
    )
    return path


def object_xml(box: str = "1,1,10,8", flags: str = "") -> str:
    coords = dict(zip(("xmin", "ymin", "xmax", "ymax"), box.split(","), strict=True))
    return (
        "<object><name>cat</name><bndbox>"
        + "".join(f"<{key}>{value}</{key}>" for key, value in coords.items())
        + f"</bndbox>{flags}</object>"
    )


def test_coordinates_flags_and_order(tmp_path: Path) -> None:
    path = annotation(
        tmp_path,
        object_xml(flags="<difficult>1</difficult>")
        + object_xml("10,8,10,8", "<truncated>1</truncated>"),
    )
    result = parse_voc_annotation(path)
    assert (result["filename"], result["width"], result["height"]) == ("sample.jpg", 10, 8)
    assert result["objects"] == [
        {"name": "cat", "box_xyxy": [0, 0, 10, 8], "difficult": True, "truncated": False},
        {"name": "cat", "box_xyxy": [9, 7, 10, 8], "difficult": False, "truncated": True},
    ]


def test_empty_annotation(tmp_path: Path) -> None:
    assert parse_voc_annotation(annotation(tmp_path))["objects"] == []


def test_parser_does_not_require_image_file(tmp_path: Path) -> None:
    path = annotation(tmp_path, object_xml())
    assert not (tmp_path / "sample.jpg").exists()
    result = parse_voc_annotation(path)
    assert result["filename"] == "sample.jpg"
    assert result["objects"][0]["box_xyxy"] == [0, 0, 10, 8]


@pytest.mark.parametrize(
    "field", ["filename", "size", "size/width", "size/height", "object/name", "object/bndbox/xmin"]
)
@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_required_fields(tmp_path: Path, field: str, empty: bool) -> None:
    path = annotation(tmp_path, object_xml())
    tree = ET.parse(path)
    element = tree.find(field)
    assert element is not None
    if empty:
        element.clear()
    else:
        parent_path, _, _ = field.rpartition("/")
        parent = tree.find(parent_path) if parent_path else tree.getroot()
        assert parent is not None
        parent.remove(element)
    tree.write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="missing or empty|missing <size>"):
        parse_voc_annotation(path)


@pytest.mark.parametrize("box", ["0,1,10,8", "1,1,11,8", "5,1,3,8", "nan,1,10,8", "1,1,inf,8"])
def test_invalid_boxes(tmp_path: Path, box: str) -> None:
    with pytest.raises(ValueError, match="invalid box"):
        parse_voc_annotation(annotation(tmp_path, object_xml(box)))


@pytest.mark.parametrize("size", ["0", "-1", "bad"])
def test_invalid_dimensions(tmp_path: Path, size: str) -> None:
    with pytest.raises(ValueError):
        parse_voc_annotation(annotation(tmp_path, size=size))


@pytest.mark.parametrize(
    "objects",
    [
        "<object/>",
        "<object><name>cat</name></object>",
        object_xml(flags="<difficult>2</difficult>"),
    ],
)
def test_invalid_objects(tmp_path: Path, objects: str) -> None:
    with pytest.raises(ValueError):
        parse_voc_annotation(annotation(tmp_path, objects))


def test_missing_and_malformed_xml(tmp_path: Path) -> None:
    path = tmp_path / "missing.xml"
    with pytest.raises(FileNotFoundError):
        parse_voc_annotation(path)
    path.write_text("<annotation>", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        parse_voc_annotation(path)
