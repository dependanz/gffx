"""Differentiable rasterization on the PyTorch CPU backend.

Semantics belong to RASTERIZE_ACCEPTANCE_V0_1.md. Two of them shape this surface. All coverage,
distance and barycentric arithmetic runs in pixel space, because ``signed_distance`` is
contractually in squared pixel units and a non-square image would otherwise scale the axes
differently. And ``blur_radius_px`` is what makes silhouette gradients nonzero at all: hard
rasterization has an exactly zero gradient with respect to vertex position at an edge.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._batching import resolve_offsets
from ._common import check_eps, check_faces, check_vertices, materialize, translate_native_error

__all__ = ["rasterize", "interpolate", "CULL_NONE", "CULL_BACK", "CULL_FRONT"]

DEFAULT_EPS = 2.0 ** -20

# Values mirror GFFX_CULL_* in include/gffx/render.h, where the enumeration starts at 1 so that a
# zeroed struct cannot be read as a valid mode.
CULL_NONE = 1
CULL_BACK = 2
CULL_FRONT = 3


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ndc_vertices, faces, vertex_offsets, face_offsets, image_height,
                image_width, faces_per_pixel, blur_radius_px, cull_mode, eps):
        try:
            face_index, barycentric, depth, signed_distance = torch.ops.gffx.rasterize(
                ndc_vertices, faces, vertex_offsets, face_offsets, image_height, image_width,
                faces_per_pixel, blur_radius_px, cull_mode, eps,
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(ndc_vertices, faces, face_index)
        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.mark_non_differentiable(face_index)
        return face_index, barycentric, depth, signed_distance

    @staticmethod
    def backward(ctx, grad_face_index, grad_barycentric, grad_depth, grad_signed_distance):
        ndc_vertices, faces, face_index = ctx.saved_tensors
        cotangents = (grad_barycentric, grad_depth, grad_signed_distance)
        if all(cotangent is None for cotangent in cotangents):
            return (None,) * 10
        # The ABI takes all three; an absent one is a zero cotangent here rather than a null view,
        # because the kernel's signature has no per-cotangent presence flag.
        zeros = [
            torch.zeros_like(reference) if cotangent is None else materialize(cotangent)
            for cotangent, reference in zip(
                cotangents,
                (
                    torch.empty(
                        face_index.shape + (3,), dtype=ndc_vertices.dtype),
                    torch.empty(face_index.shape, dtype=ndc_vertices.dtype),
                    torch.empty(face_index.shape, dtype=ndc_vertices.dtype),
                ),
            )
        ]
        try:
            grad_ndc = torch.ops.gffx.rasterize_backward(
                ndc_vertices, faces, ctx.image_height, ctx.image_width, face_index,
                zeros[0], zeros[1], zeros[2],
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return grad_ndc, None, None, None, None, None, None, None, None, None


class _Interpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, face_index, barycentric, face_attributes):
        try:
            attributes = torch.ops.gffx.interpolate(face_index, barycentric, face_attributes)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(face_index, barycentric, face_attributes)
        return attributes

    @staticmethod
    def backward(ctx, grad_attributes):
        face_index, barycentric, face_attributes = ctx.saved_tensors
        try:
            grad_barycentric, grad_face_attributes = torch.ops.gffx.interpolate_backward(
                face_index, barycentric, face_attributes, materialize(grad_attributes)
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        return None, grad_barycentric, grad_face_attributes


def rasterize(
    ndc_vertices: torch.Tensor,
    faces: torch.Tensor,
    image_height: int,
    image_width: int,
    faces_per_pixel: int = 1,
    blur_radius_px: float = 0.0,
    cull_mode: int = CULL_BACK,
    eps: float = DEFAULT_EPS,
    vertex_offsets: Optional[torch.Tensor] = None,
    face_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize NDC triangles into per-pixel fragments.

    Returns ``(face_index[B,H,W,K], barycentric[B,H,W,K,3], depth[B,H,W,K],
    signed_distance[B,H,W,K])``. ``signed_distance`` is in **squared pixel** units, negative
    inside the triangle and positive outside; a positive ``blur_radius_px`` admits fragments
    outside the triangle, which is what gives the silhouette a nonzero gradient.
    """
    check_vertices(ndc_vertices, "ndc_vertices")
    check_faces(faces)
    for name, value in (("image_height", image_height), ("image_width", image_width),
                        ("faces_per_pixel", faces_per_pixel)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("%s must be a positive integer; received %r" % (name, value))
    if cull_mode not in (CULL_NONE, CULL_BACK, CULL_FRONT):
        raise ValueError(
            "cull_mode must be CULL_NONE, CULL_BACK or CULL_FRONT; received %r" % (cull_mode,)
        )
    blur = float(blur_radius_px)
    if not (blur >= 0.0) or blur != blur:
        raise ValueError("blur_radius_px must be finite and non-negative; received %r"
                         % (blur_radius_px,))
    vertex_offsets = resolve_offsets(vertex_offsets, ndc_vertices.shape[0], "vertex_offsets")
    face_offsets = resolve_offsets(face_offsets, faces.shape[0], "face_offsets")
    return _Rasterize.apply(
        ndc_vertices, faces, vertex_offsets, face_offsets, image_height, image_width,
        faces_per_pixel, blur, cull_mode, check_eps(eps),
    )


def interpolate(
    face_index: torch.Tensor, barycentric: torch.Tensor, face_attributes: torch.Tensor
) -> torch.Tensor:
    """Interpolate per-face-corner attributes across rasterized fragments.

    ``face_attributes`` is ``[F,3,C]``; the result is the fragment shape with a trailing ``C``.
    Background fragments, where ``face_index`` is -1, take the contract's background value rather
    than reading out of range.
    """
    if not isinstance(face_index, torch.Tensor) or face_index.dtype != torch.int32:
        raise TypeError("face_index must be an int32 torch.Tensor from rasterize")
    for name, tensor in (("barycentric", barycentric), ("face_attributes", face_attributes)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("%s must be a torch.Tensor" % (name,))
        if tensor.device.type != "cpu":
            raise ValueError("%s must be on the cpu device" % (name,))
        if tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError("%s must be float32 or float64; received %s" % (name, tensor.dtype))
        if not tensor.is_contiguous():
            raise ValueError("%s must be dense and C-contiguous" % (name,))
    if barycentric.dtype != face_attributes.dtype:
        raise TypeError(
            "barycentric and face_attributes must share a dtype; received %s and %s"
            % (barycentric.dtype, face_attributes.dtype)
        )
    if face_attributes.dim() != 3 or face_attributes.shape[1] != 3:
        raise ValueError(
            "face_attributes must have shape [F, 3, C]; received %s"
            % (tuple(face_attributes.shape),)
        )
    return _Interpolate.apply(face_index, barycentric, face_attributes)
