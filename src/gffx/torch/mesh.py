"""Differentiable mesh operations on the PyTorch CPU backend.

Semantics belong to FACE_GEOMETRY_ACCEPTANCE_V0_1.md and are not restated here. This module only
converts, validates, and attaches the gradient.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ._common import check_pair, materialize, translate_native_error

__all__ = ["face_geometry"]

DEFAULT_EPS = 2.0 ** -20


class _FaceGeometry(torch.autograd.Function):
    """Attaches the C reference's gradient to the registered forward.

    torch::autograd::Function is not part of the LibTorch Stable ABI, so the gradient is attached
    here rather than in the native translation unit. This is a supported custom-operation
    mechanism, and it keeps exactly one implementation of the formula: backward calls the same
    gffx_mesh_face_geometry_backward the C tests cover, and composes nothing of its own.
    """

    @staticmethod
    def forward(ctx, vertices, faces, eps):
        try:
            normals, areas, valid = torch.ops.gffx.face_geometry(vertices, faces, eps)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(vertices, faces)
        ctx.eps = eps
        # valid is boolean topology-of-the-moment, nondifferentiable, and carries no cotangent.
        ctx.mark_non_differentiable(valid)
        return normals, areas, valid

    @staticmethod
    def backward(ctx, grad_normals, grad_areas, grad_valid):
        vertices, faces = ctx.saved_tensors
        has_grad_normals = grad_normals is not None
        has_grad_areas = grad_areas is not None
        if not has_grad_normals and not has_grad_areas:
            return None, None, None

        # The ABI distinguishes an absent cotangent from a zero one, so presence travels as a flag
        # and the unused tensor is a placeholder the native side never reads.
        if not has_grad_normals:
            grad_normals = torch.empty(0, dtype=vertices.dtype)
        if not has_grad_areas:
            grad_areas = torch.empty(0, dtype=vertices.dtype)

        try:
            grad_vertices = torch.ops.gffx.face_geometry_backward(
                vertices, faces, ctx.eps,
                materialize(grad_normals), materialize(grad_areas),
                has_grad_normals, has_grad_areas,
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        # faces is integer topology and eps is a static argument; neither takes a gradient.
        return grad_vertices, None, None


def face_geometry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-face unit normals, areas, and validity.

    Returns ``(unit_normals[F,3], areas[F], valid[F])`` in the vertices dtype, with ``valid`` a
    bool tensor. A face whose doubled area does not exceed ``eps`` is invalid: its normal and area
    are exactly zero rather than NaN.

    ``vertices`` must be a contiguous CPU ``float32`` or ``float64`` tensor of shape ``[V,3]``;
    ``faces`` a contiguous CPU ``int32`` tensor of shape ``[F,3]``. Non-contiguous and ``int64``
    inputs are refused rather than converted, so no copy happens without the caller asking.
    """
    vertices, faces, eps = check_pair(vertices, faces, eps)
    return _FaceGeometry.apply(vertices, faces, eps)
