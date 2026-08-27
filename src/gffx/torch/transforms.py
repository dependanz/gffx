"""Camera and coordinate transforms on the PyTorch CPU backend.

Semantics belong to TRANSFORMS_ACCEPTANCE_V0_1.md and CAMERA_CONTRACT_V0_1.md.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._batching import resolve_offsets
from ._common import check_eps, materialize, translate_native_error

__all__ = ["transform_points", "perspective_divide"]

DEFAULT_EPS = 2.0 ** -20


def _check_points(points: torch.Tensor, name: str, width: int) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor, not %s" % (name, type(points).__name__))
    if points.device.type != "cpu":
        raise ValueError("%s must be on the cpu device; received %s" % (name, points.device))
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError("%s must be float32 or float64; received %s" % (name, points.dtype))
    if points.dim() != 2 or points.shape[1] != width:
        raise ValueError(
            "%s must have shape [N, %d]; received %s" % (name, width, tuple(points.shape))
        )
    if any(stride <= 0 for stride in points.stride()):
        raise ValueError("%s must be densely strided, not a broadcast view" % (name,))
    if not points.is_contiguous():
        raise ValueError("%s must be dense and C-contiguous" % (name,))


class _TransformPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrices, point_offsets):
        try:
            homogeneous = torch.ops.gffx.transform_points(points, matrices, point_offsets)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(points, matrices, point_offsets)
        return homogeneous

    @staticmethod
    def backward(ctx, grad_homogeneous):
        points, matrices, point_offsets = ctx.saved_tensors
        try:
            grad_points, grad_matrices = torch.ops.gffx.transform_points_backward(
                points, matrices, point_offsets, materialize(grad_homogeneous)
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return grad_points, grad_matrices, None


class _PerspectiveDivide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, homogeneous, eps):
        try:
            ndc, valid = torch.ops.gffx.perspective_divide(homogeneous, eps)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(homogeneous)
        ctx.eps = eps
        ctx.mark_non_differentiable(valid)
        return ndc, valid

    @staticmethod
    def backward(ctx, grad_ndc, grad_valid):
        (homogeneous,) = ctx.saved_tensors
        if grad_ndc is None:
            return None, None
        try:
            grad_homogeneous = torch.ops.gffx.perspective_divide_backward(
                homogeneous, ctx.eps, materialize(grad_ndc)
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return grad_homogeneous, None


def transform_points(
    points: torch.Tensor,
    matrices: torch.Tensor,
    point_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply one 4x4 matrix per batch element to packed points, returning homogeneous [P,4].

    ``matrices`` is ``[B,4,4]`` and ``point_offsets`` an int32 ``[B+1]`` packing boundary. With a
    single matrix and no offsets, the single-element packing is synthesised.
    """
    _check_points(points, "points", 3)
    if not isinstance(matrices, torch.Tensor):
        raise TypeError("matrices must be a torch.Tensor")
    if matrices.device.type != "cpu":
        raise ValueError("matrices must be on the cpu device; received %s" % (matrices.device,))
    if matrices.dtype != points.dtype:
        raise TypeError(
            "matrices must match the points dtype %s; received %s" % (points.dtype, matrices.dtype)
        )
    if matrices.dim() != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("matrices must have shape [B, 4, 4]; received %s"
                         % (tuple(matrices.shape),))
    if not matrices.is_contiguous():
        raise ValueError("matrices must be dense and C-contiguous")
    point_offsets = resolve_offsets(point_offsets, points.shape[0], "point_offsets")
    if point_offsets.numel() - 1 != matrices.shape[0]:
        raise ValueError(
            "point_offsets declares %d batch elements but matrices supplies %d"
            % (point_offsets.numel() - 1, matrices.shape[0])
        )
    return _TransformPoints.apply(points, matrices, point_offsets)


def perspective_divide(
    homogeneous: torch.Tensor, eps: float = DEFAULT_EPS
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Divide homogeneous [P,4] by w, returning ``(ndc[P,3], valid[P])``.

    A point whose ``|w|`` does not exceed ``eps`` is invalid: its ndc is exactly zero rather than
    infinite, and ``valid`` says so.
    """
    _check_points(homogeneous, "homogeneous", 4)
    return _PerspectiveDivide.apply(homogeneous, check_eps(eps))
