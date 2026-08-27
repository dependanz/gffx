"""Differentiable mesh operations on the PyTorch CPU backend.

Semantics belong to FACE_GEOMETRY_ACCEPTANCE_V0_1.md and are not restated here. This module only
converts, validates, and attaches the gradient.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ._batching import resolve_offsets
from ._common import (
    check_eps, check_faces, check_pair, check_vertices, materialize, translate_native_error,
)

__all__ = [
    "face_geometry", "vertex_normals", "gather_faces", "build_edge_topology",
    "sample_surface", "WEIGHTING_AREA", "WEIGHTING_UNIFORM",
]

WEIGHTING_AREA = 1
WEIGHTING_UNIFORM = 2

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


class _VertexNormals(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, faces, eps, weighting):
        try:
            normals = torch.ops.gffx.vertex_normals(vertices, faces, eps, weighting)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(vertices, faces)
        ctx.eps = eps
        ctx.weighting = weighting
        return normals

    @staticmethod
    def backward(ctx, grad_normals):
        vertices, faces = ctx.saved_tensors
        if grad_normals is None:
            return None, None, None, None
        try:
            grad_vertices = torch.ops.gffx.vertex_normals_backward(
                vertices, faces, ctx.eps, ctx.weighting, materialize(grad_normals)
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return grad_vertices, None, None, None


class _GatherFaces(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, faces):
        try:
            gathered = torch.ops.gffx.gather_faces(vertices, faces)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(vertices, faces)
        return gathered

    @staticmethod
    def backward(ctx, grad_gathered):
        vertices, faces = ctx.saved_tensors
        if grad_gathered is None:
            return None, None
        try:
            grad_vertices = torch.ops.gffx.gather_faces_backward(
                vertices, faces, materialize(grad_gathered)
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return grad_vertices, None


class _SampleSurface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, faces, vertex_offsets, face_offsets, sample_count, rng_key,
                rng_counter, eps):
        try:
            points, face_index, barycentric, next_counter = torch.ops.gffx.sample_surface(
                vertices, faces, vertex_offsets, face_offsets, sample_count, rng_key,
                rng_counter, eps,
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(vertices, faces, face_index, barycentric)
        # Selection, probabilities, indices and the counter are all nondifferentiable; only the
        # sampled positions carry a gradient, and they are linear in the vertices once the face
        # and weights are fixed.
        ctx.mark_non_differentiable(face_index, next_counter)
        return points, face_index, barycentric, next_counter

    @staticmethod
    def backward(ctx, grad_points, grad_face_index, grad_barycentric, grad_next_counter):
        vertices, faces, face_index, barycentric = ctx.saved_tensors
        if grad_points is None:
            return (None,) * 8
        try:
            grad_vertices = torch.ops.gffx.sample_surface_backward(
                vertices, faces, face_index, barycentric, materialize(grad_points)
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return grad_vertices, None, None, None, None, None, None, None


def vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eps: float = DEFAULT_EPS,
    weighting: int = WEIGHTING_AREA,
) -> torch.Tensor:
    """Accumulated, normalized per-vertex normals, returning ``[V,3]``.

    ``weighting`` selects the face contribution: ``WEIGHTING_AREA`` weights by face area, giving
    the smooth shading normal, and ``WEIGHTING_UNIFORM`` weights every incident face equally. A
    vertex whose accumulated sum has magnitude at or below ``eps``, including an isolated vertex,
    gets the exact zero vector rather than a normalized noise direction.
    """
    vertices, faces, eps = check_pair(vertices, faces, eps)
    if weighting not in (WEIGHTING_AREA, WEIGHTING_UNIFORM):
        raise ValueError(
            "weighting must be WEIGHTING_AREA or WEIGHTING_UNIFORM; received %r" % (weighting,)
        )
    return _VertexNormals.apply(vertices, faces, eps, weighting)


def gather_faces(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-face corner positions, returning ``[F,3,3]``.

    A pure gather with no arithmetic: values including NaN and infinity are copied bit for bit,
    and there is no eps and no validity output.
    """
    check_vertices(vertices)
    check_faces(faces)
    return _GatherFaces.apply(vertices, faces)


def build_edge_topology(faces, face_offsets=None):
    """Canonical undirected edges and their incident faces.

    Returns ``(edges, edge_face_offsets, edge_faces, mesh_edge_offsets)``. Rows ``[0,E)`` of
    ``edges`` are valid and every trailing row is ``(-1,-1)``; ``mesh_edge_offsets`` ends at E. A
    non-manifold edge keeps **every** incident face rather than the first two, and degenerate
    self-edges are retained.

    Every output is integer topology and therefore nondifferentiable, so this has no backward. It
    is setup-class work and is not permitted inside a claimed real-time frame path; its outputs
    are also the only ones in v0.1 a streaming host may hold across frames, being a function of
    topology alone.
    """
    check_faces(faces)
    face_offsets = resolve_offsets(face_offsets, faces.shape[0], "face_offsets")
    try:
        return torch.ops.gffx.build_edge_topology(faces, face_offsets)
    except RuntimeError as error:
        raise translate_native_error(error) from None


def sample_surface(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sample_count: int,
    rng_key: torch.Tensor,
    rng_counter: torch.Tensor,
    eps: float = DEFAULT_EPS,
    vertex_offsets=None,
    face_offsets=None,
):
    """Area-weighted uniform sampling of triangle surfaces.

    Returns ``(points[B,S,3], face_index[B,S], barycentric[B,S,3], next_counter[2])``.

    Randomness is Philox4x32-10, counter-based and stateless: GFFX owns no random state and never
    touches a framework global generator. The 128-bit counter embeds ``(batch, sample)`` rather
    than iterating a stream, so a sample's value does not depend on how many preceded it and
    reproducibility survives any future parallelization. Pass ``next_counter`` as the following
    call's ``rng_counter`` to continue a stream.
    """
    check_vertices(vertices)
    check_faces(faces)
    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer; received %r"
                         % (sample_count,))
    for name, tensor in (("rng_key", rng_key), ("rng_counter", rng_counter)):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint32:
            raise TypeError("%s must be a uint32 torch.Tensor of shape [2]" % (name,))
        if tensor.dim() != 1 or tensor.numel() != 2:
            raise ValueError("%s must have shape [2]; received %s" % (name, tuple(tensor.shape)))
    vertex_offsets = resolve_offsets(vertex_offsets, vertices.shape[0], "vertex_offsets")
    face_offsets = resolve_offsets(face_offsets, faces.shape[0], "face_offsets")
    return _SampleSurface.apply(
        vertices, faces, vertex_offsets, face_offsets, sample_count, rng_key, rng_counter,
        check_eps(eps),
    )
