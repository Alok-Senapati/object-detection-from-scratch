"""Rendering contracts and CLI checks using synthetic VOC data and the Agg backend."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from od_scratch.data.dataset import Target
from od_scratch.utils.visualizer import visualize_batch


def sample(empty: bool = False) -> tuple[torch.Tensor, Target]:
    image = torch.ones(3, 20, 30, requires_grad=True)
    target = {
        "boxes": torch.empty(0, 4) if empty else torch.tensor([[1.0, 2.0, 15.0, 18.0]]),
        "labels": torch.tensor([] if empty else [7], dtype=torch.int64),
        "truncated": torch.tensor([] if empty else [True], dtype=torch.bool),
        "difficult": torch.tensor([] if empty else [True], dtype=torch.bool),
        "image_id": torch.tensor(3),
        "orig_size": torch.tensor([20, 30]),
    }
    return image, target


@pytest.mark.parametrize("count", [1, 2, 5, 6])
def test_render_saves_and_closes(tmp_path: Path, count: int) -> None:
    image, target = sample(empty=True)
    before = plt.get_fignums()
    with patch.object(plt, "show") as show:
        visualize_batch(
            [image] * count,
            [target] * count,
            "batch.png",
            output_dir=tmp_path / "nested",
            show=False,
        )
    show.assert_not_called()
    with Image.open(tmp_path / "nested/batch.png") as saved:
        assert saved.format == "PNG"
        assert saved.width > 0 and saved.height > 0
        saved.verify()
    assert plt.get_fignums() == before


def test_overlay_geometry_flags_and_no_mutation() -> None:
    image, target = sample()
    original = {key: value.clone() for key, value in target.items()}
    figures: list[Figure] = []

    def inspect() -> None:
        fig = plt.gcf()
        figures.append(fig)
        ax = fig.axes[0]
        assert ax.get_title() == "3 (30x20)"
        assert list(ax.images[0].get_extent()) == [0, 30, 20, 0]
        rect = ax.patches[0]
        assert isinstance(rect, Rectangle)
        assert rect.get_xy() == (1.0, 2.0)
        assert (rect.get_width(), rect.get_height()) == (14.0, 16.0)
        assert [text.get_text() for text in ax.texts] == ["cat (T) (D)"]

    with patch.object(plt, "show", side_effect=inspect) as show:
        visualize_batch([image], [target])
    show.assert_called_once()
    assert len(figures) == 1
    assert not plt.fignum_exists(figures[0].number)
    assert image.requires_grad
    for key, value in original.items():
        assert torch.equal(target[key], value)


@pytest.mark.parametrize("case", ["empty", "batch", "flags", "box", "negative", "large"])
def test_invalid_targets_do_not_leak_figures(case: str) -> None:
    image, target = sample()
    images, targets = [image], [target]
    if case == "empty":
        images, targets = [], []
    elif case == "batch":
        targets = []
    elif case == "flags":
        target["difficult"] = torch.empty(0, dtype=torch.bool)
    elif case == "box":
        target["boxes"] = torch.zeros(1, 3)
    else:
        target["labels"] = torch.tensor([-1 if case == "negative" else 20])
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        visualize_batch(images, targets, show=False)
    assert plt.get_fignums() == before


def test_save_failure_closes_figure(tmp_path: Path) -> None:
    image, target = sample()
    before = plt.get_fignums()
    with patch.object(Figure, "savefig", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            visualize_batch([image], [target], "batch.png", output_dir=tmp_path, show=False)
    assert plt.get_fignums() == before


def test_cli_dataset_to_png(tmp_path: Path) -> None:
    root = tmp_path / "voc"
    for directory in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (root / directory).mkdir(parents=True)
    Image.new("RGB", (30, 20), (60, 120, 180)).save(root / "JPEGImages/sample.jpg")
    (root / "Annotations/sample.xml").write_text(
        "<annotation><filename>sample.jpg</filename><size><width>30</width>"
        "<height>20</height></size><object><name>cat</name><bndbox>"
        "<xmin>2</xmin><ymin>3</ymin><xmax>15</xmax><ymax>18</ymax>"
        "</bndbox><difficult>1</difficult></object></annotation>",
        encoding="utf-8",
    )
    (root / "ImageSets/Main/val.txt").write_text("sample\n", encoding="utf-8")
    repo = Path(__file__).parents[1]
    common = ["--root", str(root), "--split", "val", "--batch-size", "1"]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    checked = subprocess.run(
        [sys.executable, "scripts/check_dataloader.py", *common],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "Samples: 1" in checked.stdout
    assert "torch.Size([3, 20, 30])" in checked.stdout
    subprocess.run(
        [
            sys.executable,
            "scripts/visualize_dataset.py",
            *common,
            "--no-show",
            "--output-dir",
            str(tmp_path / "plots"),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    outputs = list((tmp_path / "plots").glob("visualize_val_batch_*.png"))
    assert len(outputs) == 1
    with Image.open(outputs[0]) as saved:
        saved.verify()


@pytest.mark.parametrize("script", ["check_dataloader.py", "visualize_dataset.py"])
def test_cli_rejects_nonpositive_batch_size(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts" / script), "--batch-size", "0"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 2
    assert "--batch-size must be positive" in result.stderr
