"""Print tensor contracts for the first batch of a local VOC split."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from od_scratch.data.dataset import DATASET_BASEPATH, VOCDataset, detection_collate


def main() -> None:
    """Load the requested split and report image and target shapes and dtypes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_BASEPATH)
    parser.add_argument("--split", choices=("train", "val", "trainval", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    dataset = VOCDataset(root=args.root, split=args.split)

    print("Samples:", len(dataset))
    image, target = dataset[0]
    print(f"Image: {image.shape}, {image.dtype}")

    for key, value in target.items():
        print(key, value.shape, value.dtype)
    print("=" * 100)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )
    images, targets = next(iter(loader))
    for image, target in zip(images, targets, strict=True):
        print("-" * 100)
        print(f"Image: {image.shape}, dtype={image.dtype}")
        for key, value in target.items():
            print(f"{key}={value}, {value.shape}, dtype={value.dtype}")
        print("-" * 100)


if __name__ == "__main__":
    main()
