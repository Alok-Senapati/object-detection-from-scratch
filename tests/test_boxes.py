"""Numeric and boundary contracts for continuous pixel-edge box geometry."""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from od_scratch.geometry.boxes import (
    box_area,
    box_iou,
    clip_xyxy,
    cxcywh_to_xyxy,
    denormalize_xyxy,
    intersection_area,
    normalize_xyxy,
    validate_width_height,
    xyxy_to_cxcywh,
)

Unary = Callable[[Tensor], Tensor]
Pairwise = Callable[[Tensor, Tensor], Tensor]
UNARY: list[Unary] = [
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
    box_area,
    lambda boxes: normalize_xyxy(boxes, 80, 40),
    lambda boxes: denormalize_xyxy(boxes, 80, 40),
    lambda boxes: clip_xyxy(boxes, 80, 40),
]
UNARY_IDS = ["to_center", "to_corners", "area", "normalize", "denormalize", "clip"]
PAIRWISE: list[Pairwise] = [intersection_area, box_iou]


@pytest.mark.parametrize(("width", "height"), [(1, 1), (640, 480), (1, 1000)])
def test_positive_dimensions(width: int, height: int) -> None:
    """Accept positive image dimensions, including one-pixel images.

    Args:
        width: Image width for this test case.
        height: Image height for this test case.

    Returns:
        None.
    """
    validate_width_height(width, height)


@pytest.mark.parametrize(
    "operation", [validate_width_height, normalize_xyxy, denormalize_xyxy, clip_xyxy]
)
@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (0, 40, "width must be positive: 0"),
        (-5, 40, "width must be positive: -5"),
        (80, 0, "height must be positive: 0"),
        (80, -7, "height must be positive: -7"),
        (0, 0, "width must be positive: 0"),
    ],
)
def test_invalid_dimensions(
    operation: Callable[..., object],
    width: int,
    height: int,
    message: str,
) -> None:
    """Reject nonpositive dimensions with the correct value in the error.

    Args:
        operation: Geometry function under test.
        width: Image width for this test case.
        height: Image height for this test case.
        message: Expected validation-error message or pattern.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match=f"^{message}$"):
        if operation is validate_width_height:
            operation(width, height)
        else:
            operation(torch.empty(0, 4), width, height)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conversion_known_values_and_roundtrips(dtype: torch.dtype) -> None:
    """Match hand-calculated conversions and recover original boxes after roundtrips.

    Args:
        dtype: Floating-point precision under test.

    Returns:
        None.
    """
    corners = torch.tensor(
        [[10, 20, 50, 100], [-3, -5, 2, 4], [7, 9, 7, 9], [8, 6, 2, 0]], dtype=dtype
    )
    centers = torch.tensor(
        [[30, 60, 40, 80], [-0.5, -0.5, 5, 9], [7, 9, 0, 0], [5, 3, -6, -6]], dtype=dtype
    )
    torch.testing.assert_close(xyxy_to_cxcywh(corners), centers)
    torch.testing.assert_close(cxcywh_to_xyxy(centers), corners)
    torch.testing.assert_close(cxcywh_to_xyxy(xyxy_to_cxcywh(corners)), corners)
    torch.testing.assert_close(xyxy_to_cxcywh(cxcywh_to_xyxy(centers)), centers)


def test_normalization_non_square_image_and_outside_edges() -> None:
    """Scale each axis correctly without clipping coordinates outside the image.

    Returns:
        None.
    """
    boxes = torch.tensor(
        [[200.0, 40.0, 600.0, 360.0], [0.0, 0.0, 800.0, 400.0], [-80.0, -80.0, 880.0, 600.0]]
    )
    expected = torch.tensor([[0.25, 0.1, 0.75, 0.9], [0.0, 0.0, 1.0, 1.0], [-0.1, -0.2, 1.1, 1.5]])
    torch.testing.assert_close(normalize_xyxy(boxes, 800, 400), expected)
    torch.testing.assert_close(denormalize_xyxy(expected, 800, 400), boxes)
    torch.testing.assert_close(denormalize_xyxy(normalize_xyxy(boxes, 800, 400), 800, 400), boxes)
    torch.testing.assert_close(normalize_xyxy(boxes, 1, 1), boxes)


def test_clip_boundaries_outside_and_inverted_boxes() -> None:
    """Clamp to pixel edges while preserving in-bounds and inverted coordinates.

    Returns:
        None.
    """
    boxes = torch.tensor(
        [
            [-20.0, 100.0, 700.0, 520.0],
            [660.0, 100.0, 700.0, 200.0],
            [-30.0, -20.0, -10.0, -5.0],
            [0.0, 0.0, 640.0, 480.0],
            [10.0, 20.0, 50.0, 100.0],
            [50.0, 100.0, 10.0, 20.0],
        ]
    )
    expected = torch.tensor(
        [
            [0.0, 100.0, 640.0, 480.0],
            [640.0, 100.0, 640.0, 200.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 640.0, 480.0],
            [10.0, 20.0, 50.0, 100.0],
            [50.0, 100.0, 10.0, 20.0],
        ]
    )
    clipped = clip_xyxy(boxes, 640, 480)
    torch.testing.assert_close(clipped, expected)
    torch.testing.assert_close(clip_xyxy(clipped, 640, 480), expected)


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ([0.0, 0.0, 1.0, 1.0], 1.0),
        ([1.0, 2.0, 4.0, 6.0], 12.0),
        ([-3.0, -2.0, 1.0, 3.0], 20.0),
        ([0.0, 0.0, 0.5, 0.25], 0.125),
        ([2.0, 1.0, 2.0, 5.0], 0.0),
        ([1.0, 2.0, 5.0, 2.0], 0.0),
        ([4.0, 1.0, 2.0, 5.0], 0.0),
        ([1.0, 5.0, 4.0, 2.0], 0.0),
        ([4.0, 5.0, 2.0, 1.0], 0.0),
        ([0.0, 0.0, 0.0, 0.0], 0.0),
    ],
)
def test_area_including_degenerate_and_inverted(box: list[float], expected: float) -> None:
    """Compute continuous areas and assign zero to degenerate or inverted boxes.

    Args:
        box: Box coordinates or coordinate text for this test case.
        expected: Expected numeric result.

    Returns:
        None.
    """
    area = box_area(torch.tensor(box))
    assert area.shape == ()
    assert area.item() == pytest.approx(expected)


def test_pairwise_intersection_and_iou_known_matrix() -> None:
    """Match hand-calculated pairwise overlaps and transpose under argument reversal.

    Returns:
        None.
    """
    first = torch.tensor([[0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 12.0, 12.0]])
    second = torch.tensor(
        [
            [0.0, 0.0, 4.0, 4.0],
            [2.0, 2.0, 6.0, 6.0],
            [1.0, 1.0, 2.0, 2.0],
            [4.0, 0.0, 6.0, 2.0],
            [4.0, 4.0, 5.0, 5.0],
            [20.0, 20.0, 21.0, 21.0],
        ]
    )
    expected_intersection = torch.tensor([[16.0, 4.0, 1.0, 0.0, 0.0, 0.0], [0.0] * 6])
    expected_iou = torch.tensor([[1.0, 1 / 7, 1 / 16, 0.0, 0.0, 0.0], [0.0] * 6])
    torch.testing.assert_close(intersection_area(first, second), expected_intersection)
    torch.testing.assert_close(box_iou(first, second), expected_iou)
    torch.testing.assert_close(box_iou(second, first), expected_iou.T)
    torch.testing.assert_close(intersection_area(second, first), expected_intersection.T)


def test_degenerate_pairs_have_zero_iou_and_finite_gradients() -> None:
    """Keep zero-union IoU values and their gradients finite.

    Returns:
        None.
    """
    invalid = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 3.0], [0.0, 2.0, 3.0, 2.0], [3.0, 3.0, 1.0, 1.0]],
        requires_grad=True,
    )
    valid = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    for other in (invalid, valid):
        torch.testing.assert_close(
            intersection_area(invalid, other), torch.zeros(len(invalid), len(other))
        )
        iou = box_iou(invalid, other)
        torch.testing.assert_close(iou, torch.zeros_like(iou))
        torch.autograd.backward(iou.sum())
    assert invalid.grad is not None
    assert torch.isfinite(invalid.grad).all()


@pytest.mark.parametrize("operation", UNARY, ids=UNARY_IDS)
@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4), (0, 4), (2, 0, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unary_shapes_precision_and_no_mutation(
    operation: Unary,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    """Preserve unary shapes and precision without aliasing or modifying inputs.

    Args:
        operation: Geometry function under test.
        shape: Input tensor shape for this test case.
        dtype: Floating-point precision under test.

    Returns:
        None.
    """
    boxes = torch.ones(shape, dtype=dtype)
    original = boxes.clone()
    result = operation(boxes)
    assert result.shape == (shape[:-1] if operation is box_area else shape)
    assert result.dtype == dtype
    assert result.device == boxes.device
    torch.testing.assert_close(boxes, original)
    result.add_(9)
    torch.testing.assert_close(boxes, original)


@pytest.mark.parametrize("operation", PAIRWISE)
@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 3), (2, 0), (1, 1), (2, 3)])
def test_pairwise_shapes_precision_and_no_mutation(operation: Pairwise, n: int, m: int) -> None:
    """Preserve pairwise axes and precision, including empty box sets.

    Args:
        operation: Geometry function under test.
        n: Number of boxes in the first set.
        m: Number of boxes in the second set.

    Returns:
        None.
    """
    first = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float64).repeat(n, 1)
    second = torch.tensor([[1.0, 1.0, 3.0, 3.0]], dtype=torch.float64).repeat(m, 1)
    originals = (first.clone(), second.clone())
    result = operation(first, second)
    assert result.shape == (n, m)
    assert result.dtype == torch.float64
    assert result.device == first.device
    result.add_(10)
    torch.testing.assert_close(first, originals[0])
    torch.testing.assert_close(second, originals[1])


@pytest.mark.parametrize("operation", UNARY, ids=UNARY_IDS)
def test_noncontiguous_unary(operation: Unary) -> None:
    """Match contiguous results for unary operations on strided tensors.

    Args:
        operation: Geometry function under test.

    Returns:
        None.
    """
    boxes = torch.arange(24.0, dtype=torch.float64).reshape(4, 6).T
    assert not boxes.is_contiguous()
    torch.testing.assert_close(operation(boxes), operation(boxes.contiguous()))


@pytest.mark.parametrize("operation", PAIRWISE)
def test_noncontiguous_pairwise(operation: Pairwise) -> None:
    """Match contiguous results for pairwise operations on strided tensors.

    Args:
        operation: Geometry function under test.

    Returns:
        None.
    """
    boxes = torch.arange(24.0, dtype=torch.float64).reshape(4, 6).T
    torch.testing.assert_close(
        operation(boxes, boxes[:2]), operation(boxes.contiguous(), boxes[:2].contiguous())
    )


def test_integer_inputs_preserve_fractional_conversion_and_iou() -> None:
    """Retain fractional results when integer coordinates require division.

    Returns:
        None.
    """
    boxes = torch.tensor([[0, 0, 3, 5]])
    torch.testing.assert_close(xyxy_to_cxcywh(boxes), torch.tensor([[1.5, 2.5, 3.0, 5.0]]))
    torch.testing.assert_close(
        cxcywh_to_xyxy(torch.tensor([[2, 3, 3, 5]])), torch.tensor([[0.5, 0.5, 3.5, 5.5]])
    )
    torch.testing.assert_close(normalize_xyxy(boxes, 6, 10), torch.tensor([[0.0, 0.0, 0.5, 0.5]]))
    assert box_area(boxes).item() == 15
    torch.testing.assert_close(
        box_iou(boxes, torch.tensor([[0, 0, 6, 10]])), torch.tensor([[0.25]])
    )


@pytest.mark.parametrize("operation", UNARY, ids=UNARY_IDS)
@pytest.mark.parametrize("shape", [(), (3,), (2, 5)])
def test_malformed_unary_shapes_fail(operation: Unary, shape: tuple[int, ...]) -> None:
    """Reject tensors whose final axis cannot supply four coordinates.

    Args:
        operation: Geometry function under test.
        shape: Input tensor shape for this test case.

    Returns:
        None.
    """
    with pytest.raises((ValueError, IndexError)):
        operation(torch.ones(shape))


@pytest.mark.parametrize("operation", UNARY, ids=UNARY_IDS)
def test_unary_autograd(operation: Unary) -> None:
    """Match numerical gradients for unary geometry away from geometric kinks.

    Args:
        operation: Geometry function under test.

    Returns:
        None.
    """
    boxes = torch.tensor([[2.0, 3.0, 8.0, 9.0]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(operation, (boxes,))


@pytest.mark.parametrize("operation", PAIRWISE)
def test_pairwise_autograd(operation: Pairwise) -> None:
    """Match numerical gradients for overlapping pairwise box operations.

    Args:
        operation: Geometry function under test.

    Returns:
        None.
    """
    first = torch.tensor([[0.0, 0.0, 4.0, 5.0]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[1.0, 2.0, 6.0, 7.0]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(operation, (first, second))


def test_iou_invariants_on_random_valid_boxes() -> None:
    """Preserve IoU symmetry, identity, translation, and scale invariance.

    Returns:
        None.
    """
    generator = torch.Generator().manual_seed(42)
    lower = torch.rand(20, 2, generator=generator, dtype=torch.float64) * 10 - 5
    upper = lower + torch.rand(20, 2, generator=generator, dtype=torch.float64) * 5 + 0.1
    boxes = torch.cat((lower, upper), dim=-1)
    iou = box_iou(boxes, boxes)
    torch.testing.assert_close(iou, iou.T)
    torch.testing.assert_close(iou.diag(), torch.ones(20, dtype=torch.float64))
    assert ((iou >= 0) & (iou <= 1)).all()
    torch.testing.assert_close(box_iou(boxes + 100, boxes + 100), iou)
    torch.testing.assert_close(box_iou(boxes * 3, boxes * 3), iou)
    tiny = boxes * 1e-8
    torch.testing.assert_close(box_iou(tiny, tiny), iou)
    torch.testing.assert_close(cxcywh_to_xyxy(xyxy_to_cxcywh(boxes)), boxes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_matches_cpu() -> None:
    """Match CPU geometry results while preserving CUDA output placement.

    Returns:
        None.
    """
    boxes = torch.tensor([[1.0, 2.0, 5.0, 7.0], [2.0, 3.0, 8.0, 9.0]], dtype=torch.float64)
    for operation in UNARY:
        result = operation(boxes.cuda())
        assert result.device.type == "cuda"
        torch.testing.assert_close(result.cpu(), operation(boxes))
    for pairwise in PAIRWISE:
        result = pairwise(boxes.cuda(), boxes.cuda())
        assert result.device.type == "cuda"
        torch.testing.assert_close(result.cpu(), pairwise(boxes, boxes))
