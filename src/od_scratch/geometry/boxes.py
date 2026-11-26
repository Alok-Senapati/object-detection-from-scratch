"""Tensor box geometry using continuous pixel edges (no inclusive-coordinate +1).

Use finite floating-point tensors for geometry, especially with gradients. Unary
operations accept (..., 4), including a single (4,) box and empty batches;
pairwise operations require (N, 4) and (M, 4) on the same device. Outputs are new
tensors on the input device and retain floating-point precision. PyTorch's normal
type promotion applies to mixed or integer inputs; integer division produces floats.

Coordinates may be negative or outside an image. Conversions do not reorder
corners, normalize does not clip, and clipping does not repair inverted boxes.
Area treats a nonpositive width or height as zero. Inputs are never modified.
Tensor shapes, dtypes, devices and finite values are caller responsibilities;
only image dimensions are explicitly validated here.
"""

import torch
from torch import Tensor


def validate_width_height(width: int, height: int) -> None:
    """Require positive image dimensions; raise ValueError naming the invalid value.

    Args:
        width: Positive image width in pixels.
        height: Positive image height in pixels.

    Returns:
        None.

    Raises:
        ValueError: Either dimension is nonpositive.
    """
    if width <= 0:
        raise ValueError(f"width must be positive: {width}")
    if height <= 0:
        raise ValueError(f"height must be positive: {height}")


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert (..., 4) corners to (center_x, center_y, width, height).

    Width and height are signed differences, so inverted corners remain inverted.
    Fractional centers are preserved rather than rounded to integer pixels.

    Args:
        boxes: Finite floating-point xyxy boxes with shape (..., 4).

    Returns:
        Center/size boxes with shape (..., 4), preserving device and floating dtype.
    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return torch.stack((cx, cy, w, h), dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (..., 4) center/size boxes to (x1, y1, x2, y2).

    Negative sizes yield inverted corners; zero sizes yield coincident edges.

    Args:
        boxes: Finite (center_x, center_y, width, height) boxes with shape (..., 4).

    Returns:
        Corner boxes with shape (..., 4), preserving device and floating dtype.
    """
    cx, cy, w, h = boxes.unbind(dim=-1)

    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2

    return torch.stack((x1, y1, x2, y2), dim=-1)


def normalize_xyxy(boxes: Tensor, width: int, height: int) -> Tensor:
    """Divide x coordinates by width and y coordinates by height, preserving shape.

    In-bounds pixel edges map to [0, 1]. Out-of-bounds coordinates remain outside
    that range. Nonpositive image dimensions raise ValueError, even for empty input.

    Args:
        boxes: Finite floating-point xyxy boxes with shape (..., 4).
        width: Positive image width in pixels.
        height: Positive image height in pixels.

    Returns:
        Corners scaled by image dimensions, with shape (..., 4).

    Raises:
        ValueError: Either image dimension is nonpositive.
    """
    validate_width_height(width, height)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)

    return torch.stack((x1 / width, y1 / height, x2 / width, y2 / height), dim=-1)


def denormalize_xyxy(boxes: Tensor, width: int, height: int) -> Tensor:
    """Scale normalized corners to pixel edges without rounding or clipping.

    Inverts normalize_xyxy up to floating-point precision. Nonpositive image
    dimensions raise ValueError, even for empty input.

    Args:
        boxes: Finite normalized xyxy boxes with shape (..., 4); values may exceed [0, 1].
        width: Positive image width in pixels.
        height: Positive image height in pixels.

    Returns:
        Pixel-edge corners with shape (..., 4).

    Raises:
        ValueError: Either image dimension is nonpositive.
    """
    validate_width_height(width, height)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)

    return torch.stack((x1 * width, y1 * height, x2 * width, y2 * height), dim=-1)


def clip_xyxy(boxes: Tensor, width: int, height: int) -> Tensor:
    """Clamp x edges to [0, width] and y edges to [0, height], preserving shape.

    The upper bounds are width/height, not width-1/height-1, for pixel edges.
    Fully outside boxes may collapse to zero area. Inverted corners are not
    reordered. Nonpositive image dimensions raise ValueError.

    Args:
        boxes: Finite floating-point xyxy boxes with shape (..., 4).
        width: Positive image width in pixels.
        height: Positive image height in pixels.

    Returns:
        Clamped pixel-edge corners with shape (..., 4).

    Raises:
        ValueError: Either image dimension is nonpositive.
    """
    validate_width_height(width, height)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)

    return torch.stack(
        (x1.clip(0, width), y1.clip(0, height), x2.clip(0, width), y2.clip(0, height)), dim=-1
    )


def box_area(boxes: Tensor) -> Tensor:
    """Return areas with shape (...) for (..., 4) xyxy boxes.

    Units are squared coordinate units (pixels squared for pixel coordinates).
    Clamp each side independently so two inverted sides cannot yield positive area.

    Args:
        boxes: Finite floating-point xyxy boxes with shape (..., 4).

    Returns:
        Nonnegative areas with shape (...), in squared coordinate units.
    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)

    box_width = (x2 - x1).clamp(min=0)
    box_height = (y2 - y1).clamp(min=0)

    return box_height * box_width


def intersection_area(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Return the (N, M) pairwise intersection areas of (N, 4) and (M, 4) boxes.

    Disjoint, touching, degenerate and inverted boxes have zero intersection.
    Empty inputs retain their pairwise axes, e.g. (0, 4) versus (M, 4) gives (0, M).

    Args:
        boxes1: Finite xyxy boxes with shape (N, 4).
        boxes2: Finite xyxy boxes with shape (M, 4), on the same device as boxes1.

    Returns:
        Pairwise intersection areas with shape (N, M), in squared coordinate units.
    """
    # Singleton axes broadcast every box in the first set against the second set.
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_boxes = torch.cat((top_left, bottom_right), dim=-1)
    return box_area(intersection_boxes)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Return dimensionless pairwise IoU with shape (N, M), in [0, 1].

    Inputs are (N, 4) and (M, 4) xyxy tensors. Zero-union pairs, including two
    zero-area boxes, receive zero rather than NaN. Empty sets are supported.
    The computation preserves autograd away from the usual geometric kinks.

    Args:
        boxes1: Finite xyxy boxes with shape (N, 4).
        boxes2: Finite xyxy boxes with shape (M, 4), on the same device as boxes1.

    Returns:
        Pairwise intersection-over-union ratios with shape (N, M). Zero-union pairs are zero.
    """
    i_area = intersection_area(boxes1, boxes2)
    b1_area = box_area(boxes1)
    b2_area = box_area(boxes2)

    u_area = b1_area[:, None] + b2_area[None, :] - i_area
    valid = u_area > 0
    # Avoid evaluating division by zero: masking its result alone can poison gradients.
    safe_union = torch.where(valid, u_area, torch.ones_like(u_area))

    iou = torch.where(
        valid,
        i_area / safe_union,
        torch.zeros_like(i_area),
    )

    return iou
