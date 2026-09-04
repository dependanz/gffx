"""Inference-only preallocated surface for the PyTorch CPU backend.

API_CONTRACT_V0_1.md section 11 fixes what these entry points are: the same kernels with the
allocation removed. They are not a faster variant computing something different, and they preserve
the functional surface's numerical, sentinel, ordering, and error semantics exactly.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import torch

from ._batching import check_offsets
from ._common import check_eps, check_pair, translate_native_error

__all__ = [
    "face_geometry_out", "face_geometry_workspace_size", "workspace_sizes",
    "vertex_normals_out", "gather_faces_out", "transform_points_out",
    "perspective_divide_out", "knn_out", "closest_point_on_mesh_out",
    "sample_surface_out", "rasterize_out", "interpolate_out", "WorkspaceSizes",
]

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


class WorkspaceSizes(NamedTuple):
    """Workspace byte requirements for one set of shapes, queried once at setup.

    A streaming host allocates the maximum it will need before its frame loop starts, which is
    what makes the no-allocation-after-warm-up rule reachable. Several of these are currently zero
    for the scalar CPU reference; a caller must still ask rather than assume, because a future
    backend may need scratch where this one does not.
    """

    vertex_normals: int
    transform_points: int
    perspective_divide: int
    closest_point_on_mesh: int
    sample_surface: int
    rasterize: int

    @property
    def maximum(self) -> int:
        """One buffer big enough for any of them, which is what a host usually allocates."""
        return max(self)


def workspace_sizes(
    vertex_count: int = 0,
    face_count: int = 0,
    point_count: int = 0,
    neighbor_count: int = 1,
    sample_count: int = 0,
    batch_count: int = 1,
    image_height: int = 1,
    image_width: int = 1,
    faces_per_pixel: int = 1,
    dtype: torch.dtype = torch.float32,
) -> WorkspaceSizes:
    """Query every operation's workspace requirement for a fixed set of shapes."""
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64; received %r" % (dtype,))
    try:
        values = torch.ops.gffx.workspace_sizes(
            vertex_count, face_count, point_count, neighbor_count, sample_count, batch_count,
            image_height, image_width, faces_per_pixel, dtype == torch.float64,
        )
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return WorkspaceSizes(*(int(value) for value in values))


def _check_stream_inputs(*tensors: torch.Tensor) -> None:
    """Refuse gradient-tracked inputs on every streaming entry point.

    The surface is nondifferentiable in v0.1. Silently dropping a gradient a caller expected would
    be worse than refusing the call, because the loss would simply stop improving with no error to
    explain it.
    """
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            raise ValueError(
                "the streaming surface is nondifferentiable in v0.1 and refuses gradient-tracked "
                "inputs. Use the functional surface for a differentiable call, or detach the "
                "input if no gradient is wanted."
            )


def _check_output(tensor, name, shape, dtype) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor" % (name,))
    if tensor.device.type != "cpu":
        raise ValueError("%s must be on the cpu device" % (name,))
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            "%s must have shape %s; received %s" % (name, tuple(shape), tuple(tensor.shape))
        )
    if tensor.dtype != dtype:
        raise TypeError("%s must be %s; received %s" % (name, dtype, tensor.dtype))
    if not tensor.is_contiguous():
        raise ValueError("%s must be dense and C-contiguous" % (name,))


def _check_workspace(workspace: torch.Tensor) -> torch.Tensor:
    if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8:
        raise TypeError("workspace must be a uint8 torch.Tensor")
    if not workspace.is_contiguous():
        raise ValueError("workspace must be dense and C-contiguous")
    return workspace


def vertex_normals_out(vertices, faces, eps=DEFAULT_EPS, weighting=1, *, out, workspace):
    """Write per-vertex normals into a caller-allocated ``[V,3]`` tensor."""
    vertices, faces, eps = check_pair(vertices, faces, eps)
    _check_stream_inputs(vertices, faces)
    _check_output(out, "out", (vertices.shape[0], 3), vertices.dtype)
    try:
        torch.ops.gffx.vertex_normals_out(
            vertices, faces, eps, weighting, out, _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return out


def gather_faces_out(vertices, faces, *, out, workspace):
    """Write per-face corner positions into a caller-allocated ``[F,3,3]`` tensor."""
    vertices, faces, _ = check_pair(vertices, faces, DEFAULT_EPS)
    _check_stream_inputs(vertices, faces)
    _check_output(out, "out", (faces.shape[0], 3, 3), vertices.dtype)
    try:
        torch.ops.gffx.gather_faces_out(vertices, faces, out, _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return out


def transform_points_out(points, matrices, point_offsets, *, out, workspace):
    """Write homogeneous coordinates into a caller-allocated ``[P,4]`` tensor.

    Offsets are required here rather than synthesised: allocating one would be exactly the hidden
    allocation this surface exists to avoid.
    """
    _check_stream_inputs(points, matrices)
    check_offsets(point_offsets, points.shape[0], "point_offsets")
    _check_output(out, "out", (points.shape[0], 4), points.dtype)
    try:
        torch.ops.gffx.transform_points_out(
            points, matrices, point_offsets, out, _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return out


def perspective_divide_out(homogeneous, eps=DEFAULT_EPS, *, ndc, valid, workspace):
    """Write NDC coordinates and validity into caller-allocated tensors."""
    _check_stream_inputs(homogeneous)
    _check_output(ndc, "ndc", (homogeneous.shape[0], 3), homogeneous.dtype)
    _check_output(valid, "valid", (homogeneous.shape[0],), torch.bool)
    try:
        torch.ops.gffx.perspective_divide_out(
            homogeneous, check_eps(eps), ndc, valid, _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return ndc, valid


def knn_out(query, reference, neighbor_count, query_offsets, reference_offsets, *,
            distance_squared, reference_index, valid, workspace):
    """Write K-nearest results into caller-allocated tensors."""
    _check_stream_inputs(query, reference)
    check_offsets(query_offsets, query.shape[0], "query_offsets")
    check_offsets(reference_offsets, reference.shape[0], "reference_offsets")
    shape = (query.shape[0], neighbor_count)
    _check_output(distance_squared, "distance_squared", shape, query.dtype)
    _check_output(reference_index, "reference_index", shape, torch.int32)
    _check_output(valid, "valid", shape, torch.bool)
    try:
        torch.ops.gffx.knn_out(
            query, reference, query_offsets, reference_offsets, neighbor_count,
            distance_squared, reference_index, valid, _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return distance_squared, reference_index, valid


def closest_point_on_mesh_out(points, vertices, faces, point_offsets, vertex_offsets,
                              face_offsets, eps=DEFAULT_EPS, *, distance_squared, face_index,
                              barycentric, closest, valid, workspace):
    """Write closest-point results into caller-allocated tensors."""
    _check_stream_inputs(points, vertices, faces)
    check_offsets(point_offsets, points.shape[0], "point_offsets")
    check_offsets(vertex_offsets, vertices.shape[0], "vertex_offsets")
    check_offsets(face_offsets, faces.shape[0], "face_offsets")
    count = points.shape[0]
    _check_output(distance_squared, "distance_squared", (count,), points.dtype)
    _check_output(face_index, "face_index", (count,), torch.int32)
    _check_output(barycentric, "barycentric", (count, 3), points.dtype)
    _check_output(closest, "closest", (count, 3), points.dtype)
    _check_output(valid, "valid", (count,), torch.bool)
    try:
        torch.ops.gffx.closest_point_on_mesh_out(
            points, vertices, faces, point_offsets, vertex_offsets, face_offsets,
            check_eps(eps), distance_squared, face_index, barycentric, closest, valid,
            _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return distance_squared, face_index, barycentric, closest, valid


def sample_surface_out(vertices, faces, vertex_offsets, face_offsets, sample_count, rng_key,
                       rng_counter, eps=DEFAULT_EPS, *, points, face_index, barycentric,
                       next_counter, workspace):
    """Write surface samples into caller-allocated tensors, advancing the counter explicitly."""
    _check_stream_inputs(vertices, faces)
    check_offsets(vertex_offsets, vertices.shape[0], "vertex_offsets")
    check_offsets(face_offsets, faces.shape[0], "face_offsets")
    batch = face_offsets.numel() - 1
    _check_output(points, "points", (batch, sample_count, 3), vertices.dtype)
    _check_output(face_index, "face_index", (batch, sample_count), torch.int32)
    _check_output(barycentric, "barycentric", (batch, sample_count, 3), vertices.dtype)
    _check_output(next_counter, "next_counter", (2,), torch.uint32)
    try:
        torch.ops.gffx.sample_surface_out(
            vertices, faces, vertex_offsets, face_offsets, sample_count, rng_key, rng_counter,
            check_eps(eps), points, face_index, barycentric, next_counter,
            _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return points, face_index, barycentric, next_counter


def rasterize_out(ndc_vertices, faces, vertex_offsets, face_offsets, image_height, image_width,
                  faces_per_pixel=1, blur_radius_px=0.0, cull_mode=2, eps=DEFAULT_EPS, *,
                  face_index, barycentric, depth, signed_distance, workspace):
    """Write rasterized fragments into caller-allocated tensors."""
    _check_stream_inputs(ndc_vertices, faces)
    check_offsets(vertex_offsets, ndc_vertices.shape[0], "vertex_offsets")
    check_offsets(face_offsets, faces.shape[0], "face_offsets")
    batch = face_offsets.numel() - 1
    shape = (batch, image_height, image_width, faces_per_pixel)
    _check_output(face_index, "face_index", shape, torch.int32)
    _check_output(barycentric, "barycentric", shape + (3,), ndc_vertices.dtype)
    _check_output(depth, "depth", shape, ndc_vertices.dtype)
    _check_output(signed_distance, "signed_distance", shape, ndc_vertices.dtype)
    try:
        torch.ops.gffx.rasterize_out(
            ndc_vertices, faces, vertex_offsets, face_offsets, image_height, image_width,
            faces_per_pixel, float(blur_radius_px), cull_mode, check_eps(eps),
            face_index, barycentric, depth, signed_distance, _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return face_index, barycentric, depth, signed_distance


def interpolate_out(face_index, barycentric, face_attributes, *, out, workspace):
    """Write interpolated attributes into a caller-allocated tensor."""
    _check_stream_inputs(barycentric, face_attributes)
    _check_output(out, "out", tuple(face_index.shape) + (face_attributes.shape[2],),
                  face_attributes.dtype)
    try:
        torch.ops.gffx.interpolate_out(
            face_index, barycentric, face_attributes, out, _check_workspace(workspace))
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return out
