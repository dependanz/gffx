"""Spatial queries on the PyTorch CPU backend.

Semantics belong to PROXIMITY_ACCEPTANCE_V0_1.md. Two limits stated there are visible in this
surface and are not adapter choices: selection is by squared distance with no square root, and
``closest_point_on_mesh`` propagates only the ``distance_squared`` cotangent, because the envelope
theorem makes that gradient exact within a fixed closest-feature region while the ``closest`` and
``barycentric`` Jacobians are region-dependent.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._batching import resolve_offsets
from ._common import (
    check_eps, check_faces, check_same_device, check_vertices, materialize, translate_native_error,
)

__all__ = ["knn", "closest_point_on_mesh"]

DEFAULT_EPS = 2.0 ** -20


class _Knn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, reference, query_offsets, reference_offsets, neighbor_count):
        try:
            distance_squared, reference_index, valid = torch.ops.gffx.knn(
                query, reference, query_offsets, reference_offsets, neighbor_count
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(query, reference, reference_index, valid)
        ctx.mark_non_differentiable(reference_index, valid)
        return distance_squared, reference_index, valid

    @staticmethod
    def backward(ctx, grad_distance_squared, grad_index, grad_valid):
        query, reference, reference_index, valid = ctx.saved_tensors
        if grad_distance_squared is None:
            return None, None, None, None, None
        try:
            grad_query, grad_reference = torch.ops.gffx.knn_backward(
                query, reference, reference_index, valid, materialize(grad_distance_squared)
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return grad_query, grad_reference, None, None, None


class _ClosestPointOnMesh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, vertices, faces, point_offsets, vertex_offsets, face_offsets, eps):
        try:
            outputs = torch.ops.gffx.closest_point_on_mesh(
                points, vertices, faces, point_offsets, vertex_offsets, face_offsets, eps
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        distance_squared, face_index, barycentric, closest, valid = outputs
        ctx.save_for_backward(points, vertices, faces, face_index, barycentric, closest, valid)
        # Only distance_squared carries a gradient; the record explains why the others cannot.
        ctx.mark_non_differentiable(face_index, valid)
        return distance_squared, face_index, barycentric, closest, valid

    @staticmethod
    def backward(ctx, grad_distance_squared, grad_face_index, grad_barycentric, grad_closest,
                 grad_valid):
        points, vertices, faces, face_index, barycentric, closest, valid = ctx.saved_tensors
        for name, cotangent in (("barycentric", grad_barycentric), ("closest", grad_closest)):
            if cotangent is not None and cotangent.abs().sum().item() != 0.0:
                raise NotImplementedError(
                    "closest_point_on_mesh propagates only the distance_squared cotangent in "
                    "v0.1; a gradient was requested through %s, whose Jacobian is "
                    "region-dependent. See PROXIMITY_ACCEPTANCE_V0_1.md." % (name,)
                )
        if grad_distance_squared is None:
            return (None,) * 7
        try:
            grad_points, grad_vertices = torch.ops.gffx.closest_point_on_mesh_backward(
                points, vertices, faces, face_index, barycentric, closest, valid,
                materialize(grad_distance_squared),
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return grad_points, grad_vertices, None, None, None, None, None


def knn(
    query: torch.Tensor,
    reference: torch.Tensor,
    neighbor_count: int,
    query_offsets: Optional[torch.Tensor] = None,
    reference_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K nearest reference points per query, within a batch element.

    Returns ``(distance_squared[Q,K], reference_index[Q,K], valid[Q,K])``. Distances are squared,
    with no square root taken, so the result carries no rounding from a root. An element holding
    fewer than K references pads with ``+inf``, ``-1`` and ``False``. Complexity is O(P*R) per
    element; v0.1 ships no spatial acceleration structure.
    """
    check_vertices(query, "query")
    check_vertices(reference, "reference")
    if query.dtype != reference.dtype:
        raise TypeError(
            "query and reference must share a dtype; received %s and %s"
            % (query.dtype, reference.dtype)
        )
    if not isinstance(neighbor_count, int) or neighbor_count <= 0:
        raise ValueError("neighbor_count must be a positive integer; received %r"
                         % (neighbor_count,))
    check_same_device(query, reference)
    query_offsets = resolve_offsets(query_offsets, query.shape[0], "query_offsets", query.device)
    reference_offsets = resolve_offsets(
        reference_offsets, reference.shape[0], "reference_offsets", reference.device)
    if query_offsets.numel() != reference_offsets.numel():
        raise ValueError("query_offsets and reference_offsets must declare the same batch count")
    return _Knn.apply(query, reference, query_offsets, reference_offsets, neighbor_count)


def closest_point_on_mesh(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eps: float = DEFAULT_EPS,
    point_offsets: Optional[torch.Tensor] = None,
    vertex_offsets: Optional[torch.Tensor] = None,
    face_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closest point on a triangle mesh for each query point.

    Returns ``(distance_squared[P], face_index[P], barycentric[P,3], closest[P,3], valid[P])``.
    The distance is **unsigned**: there is no inside/outside sign. Only ``distance_squared``
    carries a gradient.
    """
    check_vertices(points, "points")
    check_vertices(vertices, "vertices")
    check_faces(faces)
    if points.dtype != vertices.dtype:
        raise TypeError(
            "points and vertices must share a dtype; received %s and %s"
            % (points.dtype, vertices.dtype)
        )
    check_same_device(points, vertices, faces)
    point_offsets = resolve_offsets(point_offsets, points.shape[0], "point_offsets", points.device)
    vertex_offsets = resolve_offsets(
        vertex_offsets, vertices.shape[0], "vertex_offsets", vertices.device)
    face_offsets = resolve_offsets(face_offsets, faces.shape[0], "face_offsets", faces.device)
    return _ClosestPointOnMesh.apply(
        points, vertices, faces, point_offsets, vertex_offsets, face_offsets, check_eps(eps)
    )
