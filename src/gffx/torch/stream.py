"""Inference-only preallocated surface for the PyTorch CPU backend.

API_CONTRACT_V0_1.md section 11 fixes what these entry points are: the same kernels with the
allocation removed. They are not a faster variant computing something different, and they preserve
the functional surface's numerical, sentinel, ordering, and error semantics exactly.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from ._common import check_pair, translate_native_error

__all__ = ["face_geometry_out", "face_geometry_workspace_size"]

DEFAULT_EPS = 2.0 ** -20


def face_geometry_workspace_size(
    vertices: torch.Tensor, faces: torch.Tensor
) -> int:
    """Bytes of workspace ``face_geometry_out`` requires for these shapes and dtype.

    Exposed so a streaming host can allocate once at setup, which is what makes the
    no-allocation-after-warm-up rule reachable. The scalar CPU reference currently reports zero,
    but a caller must use the query rather than assuming that.
    """
    check_pair(vertices, faces, DEFAULT_EPS)
    # The query is a pure function of shapes, dtype and device; running the operation is not
    # required to ask it.
    workspace = torch.empty(0, dtype=torch.uint8)
    try:
        normals = torch.empty((faces.shape[0], 3), dtype=vertices.dtype)
        areas = torch.empty((faces.shape[0],), dtype=vertices.dtype)
        valid = torch.empty((faces.shape[0],), dtype=torch.bool)
        torch.ops.gffx.face_geometry_out(
            vertices, faces, DEFAULT_EPS, normals, areas, valid, workspace
        )
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return int(workspace.numel())


def face_geometry_out(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eps: float = DEFAULT_EPS,
    *,
    outputs: Sequence[torch.Tensor],
    workspace: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write per-face geometry into caller-allocated buffers.

    ``outputs`` is ``(unit_normals[F,3], areas[F], valid[F])`` and ``workspace`` a ``uint8`` tensor
    of at least ``face_geometry_workspace_size`` bytes. Gradient-tracked inputs are refused: this
    surface is nondifferentiable in v0.1, and silently dropping a gradient a caller expected would
    be worse than refusing the call.
    """
    vertices, faces, eps = check_pair(vertices, faces, eps)

    if vertices.requires_grad:
        raise ValueError(
            "the streaming surface is nondifferentiable in v0.1 and refuses gradient-tracked "
            "inputs. Use gffx.torch.mesh.face_geometry for a differentiable call, or detach the "
            "vertices if no gradient is wanted."
        )
    if len(outputs) != 3:
        raise ValueError(
            "outputs must be (unit_normals, areas, valid); received %d tensors" % (len(outputs),)
        )
    normals, areas, valid = outputs
    face_count = faces.shape[0]
    for tensor, name, shape, dtype in (
        (normals, "unit_normals", (face_count, 3), vertices.dtype),
        (areas, "areas", (face_count,), vertices.dtype),
        (valid, "valid", (face_count,), torch.bool),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("%s must be a torch.Tensor" % (name,))
        if tensor.device.type != "cpu":
            raise ValueError("%s must be on the cpu device" % (name,))
        if tuple(tensor.shape) != shape:
            raise ValueError(
                "%s must have shape %s to match faces and vertices; received %s"
                % (name, shape, tuple(tensor.shape))
            )
        if tensor.dtype != dtype:
            raise TypeError("%s must be %s; received %s" % (name, dtype, tensor.dtype))
        if not tensor.is_contiguous():
            raise ValueError("%s must be dense and C-contiguous" % (name,))
    if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8:
        raise TypeError("workspace must be a uint8 torch.Tensor")

    try:
        torch.ops.gffx.face_geometry_out(
            vertices, faces, eps, normals, areas, valid, workspace
        )
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return normals, areas, valid
