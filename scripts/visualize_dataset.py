"""Save and optionally display the first batch of a local VOC split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from torch.utils.data import DataLoader

from od_scratch.data.dataset import DATASET_BASEPATH, VOCDataset, detection_collate
from od_scratch.utils.visualizer import ARTIFACTS_PATH, visualize_batch


def main() -> None:
    """Render annotated samples; --no-show supports automated headless checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_BASEPATH)
    parser.add_argument("--split", choices=("train", "val", "trainval", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_PATH)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    dataset = VOCDataset(root=args.root, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )
    images, targets = next(iter(loader))
    visualize_batch(
        images,
        targets,
        save_filename=f"visualize_{args.split}_batch_{time.time_ns()}.png",
        output_dir=args.output_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
