# Resume checkpoint — 23 November 2026

## Evidence and next milestone

Main at `2ce3764` contains reusable experiment infrastructure, a validated VOC XML parser and a box preview. Geometry/model/loss/training/evaluation packages contain placeholders. No detector checkpoint or measured AP is recorded. Earlier user-reported split counts were train 2501, val 2510, test 4952; these are not independently verified in this checkout.

This increment supplies `VOCDataset`, `detection_collate` and an inspection command. Synthetic image/XML tests verify the loading contract. It does not complete Stage 4: geometry-aware transforms and a split/duplicate-content audit still precede training. Dataset targets preserve difficult objects for an explicit later loss/evaluation policy.

Validation on agent CPU: locked CPU sync succeeded; Ruff lint/format, strict Mypy (26 source files) and all 31 tests passed, including eight new dataset integration cases. CLI help also passed. This checkout has no real VOC data or CUDA training evidence; these checks establish loader behavior, not model quality or full-dataset validity.

Our next milestone is a tiny training-set overfit with visually plausible decoded boxes. Work toward it in five increments:

1. Data contract + box conversions and IoU.
2. Joint image/box resizing with reversible letterbox; audit splits.
3. Small spatial backbone/head + decoder + target assignment, including collisions.
4. Separate box/objectness/classification losses + tiny-set training.
5. NMS + independently checked AP, then a full scratch baseline.

Then run the planned matched-backbone transfer experiments, followed by the separate official YOLO fine-tuning reference. Completion needs actual runs and error analysis, not just implementation.

## Run now on your laptop

From the repository root, after getting this branch:

```bash
uv run --no-sync python scripts/check.py
uv run --no-sync python scripts/inspect_voc.py --root data/VOCdevkit/VOC2007
```

Use your existing CUDA environment. No new dependency is required. Inspection reads two training images; it does not load or inspect the test split. If data is elsewhere, change `--root`.

Images are float32 `[3,H,W]` in `[0,1]`; targets contain float32 boxes `[N,4]`, int64 labels `[N]` and boolean flags `[N]`. Empty boxes retain shape `[0,4]`. The batch holds lists because both image dimensions and object counts can vary. `image_id` is a split-local dataset index; `dataset.ids[index]` resolves the VOC ID. Class IDs are fixed zero-based VOC order; cat=7 and person=14. Keep this mapping when writing the custom head; a library detector may require a different background-label convention.

## First learner implementation: Stage 2

Create `src/od_scratch/geometry/boxes.py`. Implement these vectorized PyTorch functions:

```python
def xyxy_to_cxcywh(boxes: Tensor) -> Tensor: ...
def cxcywh_to_xyxy(boxes: Tensor) -> Tensor: ...
```

Contract: floating finite boxes of shape `[...,4]`; preserve shape/device/dtype and gradients; do not mutate the input. Empty `[0,4]` must work. Inputs are valid boxes (nonnegative widths/heights); clipping and normalization come next.

Why: XML gives corner coordinates; a detector can predict centers and sizes. Conversions must describe the same rectangle. For corner box `[20,10,100,50]`, width=80, height=40, center=(60,30), so center-size is `[60,30,80,40]`. Recover corners by subtracting/adding half sizes from/to the center. There is no `+1` after the parser has converted to pixel edges.

Your different assignment: use a batch of boxes with shape `[2,2,4]`:

```python
boxes = torch.tensor(
    [
        [[0.0, 0.0, 640.0, 480.0], [420.0, 60.0, 620.0, 180.0]],
        [[10.0, 20.0, 10.0, 20.0], [15.5, 9.5, 25.5, 29.5]],
    ]
)
```

Expected center-size output:

```text
[[[320.,240.,640.,480.], [520.,120.,200.,120.]],
 [[ 10., 20.,  0.,  0.], [20.5,19.5, 10., 20.]]]
```

Check the roundtrip using `torch.testing.assert_close`, check empty input and verify the original tensor is unchanged. Then explain why dividing all four coordinates by image width would fail on a 640×480 image. Share your implementation and the inspection output for review; the mentor will review detection mechanisms before the next increment.
