"""Inspect a real training batch without a model or GPU."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from od_scratch.data.dataset import VOC_CLASSES, VOCDataset, detection_collate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/VOCdevkit/VOC2007"))
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    dataset = VOCDataset(args.root, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=detection_collate)
    images, targets = next(iter(loader))
    print(f"Train images: {len(dataset)}")
    for image, target in zip(images, targets, strict=True):
        index = int(target["image_id"].item())
        labels = target["labels"].tolist()
        print(f"\n{dataset.ids[index]}: image={tuple(image.shape)} {image.dtype}")
        print(f"boxes={tuple(target['boxes'].shape)} {target['boxes'].dtype}")
        print(f"labels={labels}; names={[VOC_CLASSES[label] for label in labels]}")
        print(f"difficult={target['difficult'].tolist()}")
        print(target["boxes"])


if __name__ == "__main__":
    main()
