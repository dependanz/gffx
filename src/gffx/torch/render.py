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
from ._common import (
    check_eps, check_faces, check_same_device, check_vertices, materialize, translate_native_error,
)

__all__ = [
    "rasterize", "interpolate", "texture", "texture_pyramid",
    "CULL_NONE", "CULL_BACK", "CULL_FRONT",
    "FILTER_NEAREST", "FILTER_BILINEAR", "MIP_NEAREST", "MIP_LINEAR",
    "WRAP_REPEAT", "WRAP_CLAMP", "WRAP_MIRROR", "WRAP_BORDER",
]

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
    check_same_device(ndc_vertices, faces)
    vertex_offsets = resolve_offsets(
        vertex_offsets, ndc_vertices.shape[0], "vertex_offsets", ndc_vertices.device)
    face_offsets = resolve_offsets(face_offsets, faces.shape[0], "face_offsets", faces.device)
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
    check_same_device(face_index, barycentric, face_attributes)
    for name, tensor in (("barycentric", barycentric), ("face_attributes", face_attributes)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("%s must be a torch.Tensor" % (name,))
        if tensor.device.type not in ("cpu", "cuda"):
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


# Values mirror GFFX_FILTER_*, GFFX_MIP_* and GFFX_WRAP_* in include/gffx/render.h, where each
# enumeration starts at 1 so that a zeroed struct cannot be read as a valid mode.
FILTER_NEAREST = 1
FILTER_BILINEAR = 2
MIP_NEAREST = 1
MIP_LINEAR = 2
WRAP_REPEAT = 1
WRAP_CLAMP = 2
WRAP_MIRROR = 3
WRAP_BORDER = 4


class _TexturePyramid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, texture, levels):
        try:
            pyramid, level_offsets = torch.ops.gffx.texture_pyramid(texture, levels)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(level_offsets)
        ctx.texture_shape = tuple(texture.shape)
        ctx.mark_non_differentiable(level_offsets)
        return pyramid, level_offsets

    @staticmethod
    def backward(ctx, grad_pyramid, grad_level_offsets):
        if grad_pyramid is None:
            return None, None
        (level_offsets,) = ctx.saved_tensors
        height, width, channels = ctx.texture_shape
        grad_texture = torch.ops.gffx.texture_pyramid_backward(
            level_offsets, height, width, channels, materialize(grad_pyramid)
        )
        return grad_texture, None


class _Texture(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pyramid, level_offsets, texture_height, texture_width, channel_count,
                coordinates, derivatives, lod, filter_mode, mip_filter, wrap_u, wrap_v, border):
        has_derivatives = derivatives is not None
        has_lod = lod is not None
        has_border = border is not None
        # The ABI reads an absent optional as a null view, and these placeholders are never read
        # when their flag is false. An empty tensor rather than a zero-filled one, so a bug that
        # ignored a flag would fault rather than silently sample with zeros.
        empty = pyramid.new_empty(0)
        try:
            samples = torch.ops.gffx.texture(
                pyramid, level_offsets, texture_height, texture_width, channel_count,
                coordinates,
                derivatives if has_derivatives else empty,
                lod if has_lod else empty,
                has_derivatives, has_lod,
                filter_mode, mip_filter, wrap_u, wrap_v,
                border if has_border else empty, has_border,
            )
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(
            pyramid, level_offsets, coordinates,
            derivatives if has_derivatives else empty,
            lod if has_lod else empty,
            border if has_border else empty,
        )
        ctx.extents = (texture_height, texture_width)
        ctx.flags = (has_derivatives, has_lod, has_border)
        ctx.modes = (filter_mode, mip_filter, wrap_u, wrap_v)
        return samples

    @staticmethod
    def backward(ctx, grad_samples):
        if grad_samples is None:
            return (None,) * 13
        pyramid, level_offsets, coordinates, derivatives, lod, border = ctx.saved_tensors
        texture_height, texture_width = ctx.extents
        has_derivatives, has_lod, has_border = ctx.flags
        filter_mode, mip_filter, wrap_u, wrap_v = ctx.modes
        grad_pyramid, grad_coordinates = torch.ops.gffx.texture_backward(
            pyramid, level_offsets, texture_height, texture_width, coordinates, derivatives, lod,
            has_derivatives, has_lod, filter_mode, mip_filter, wrap_u, wrap_v, border, has_border,
            materialize(grad_samples),
        )
        # Only the pyramid and the coordinates carry gradients. The extents, channel count, filter,
        # wrap and mip selections are discrete choices with no derivative, and NEAREST filtering
        # returns an exactly zero coordinate gradient rather than an absent one, which is a
        # statement the contract makes deliberately.
        return (grad_pyramid, None, None, None, None, grad_coordinates,
                None, None, None, None, None, None, None)


def texture_pyramid(texture: torch.Tensor, levels: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a mip chain from ``texture`` shaped ``[H, W, C]``.

    ``levels=0`` builds the full chain, ``L = floor(log2(max(H, W))) + 1``. Level 0 is the input
    copied bit for bit; each later level is the arithmetic mean of every two-by-two block of the
    level above it, summed in a fixed order because floating-point addition is not associative and
    a reordering would make the result depend on how the work was scheduled.

    Returns the levels in one contiguous buffer and the ``[L + 1]`` offsets that address them, so a
    streaming caller builds a pyramid once and reuses it.
    """
    if texture.dim() != 3:
        raise ValueError(
            f"texture must be [H, W, C]; received a tensor with {texture.dim()} dimensions"
        )
    return _TexturePyramid.apply(materialize(texture), int(levels))


def texture(
    pyramid: torch.Tensor,
    level_offsets: torch.Tensor,
    texture_height: int,
    texture_width: int,
    coordinates: torch.Tensor,
    *,
    channel_count: Optional[int] = None,
    derivatives: Optional[torch.Tensor] = None,
    lod: Optional[torch.Tensor] = None,
    filter: int = FILTER_BILINEAR,
    mip_filter: int = MIP_NEAREST,
    wrap_u: int = WRAP_REPEAT,
    wrap_v: int = WRAP_REPEAT,
    border: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample ``pyramid`` at ``coordinates`` shaped ``[N, 2]``, returning ``[N, C]``.

    ``texture_height`` and ``texture_width`` are required because a pyramid and its offsets fix
    only the product ``H * W * C`` per level, from which the two extents cannot be recovered. They
    are cross-checked against ``level_offsets[1] - level_offsets[0]``.

    Coordinates are normalised to ``[0, 1]`` with ``(0, 0)`` at the first texel of the first row and
    ``v`` increasing with row index, so the output of :func:`interpolate` is a valid input
    unchanged. This differs from the ``[-1, 1]`` convention used by ``grid_sample`` and from v-up
    conventions; composition with :func:`interpolate` is the property being preserved.

    At most one of ``derivatives`` or ``lod`` may be given, and the operation never derives its own.
    Screen-space differencing would make each sample depend on its neighbours, which breaks both
    per-element independence and the ability to sample an unstructured point set. A caller
    rasterising through :func:`rasterize` holds neighbouring interpolated coordinates and can
    difference them.

    ``channel_count`` is derived from ``level_offsets`` when omitted, which reads two elements back
    to the host. Pass it explicitly inside a frame loop to avoid that synchronisation.
    """
    if derivatives is not None and lod is not None:
        raise ValueError("at most one of derivatives or lod may be given")
    if coordinates.dim() != 2 or coordinates.size(1) != 2:
        raise ValueError(f"coordinates must be [N, 2]; received {tuple(coordinates.shape)}")
    if channel_count is None:
        level_size = int(level_offsets[1].item()) - int(level_offsets[0].item())
        extent = int(texture_height) * int(texture_width)
        if extent <= 0 or level_size % extent != 0:
            raise ValueError(
                f"cannot derive the channel count: level 0 holds {level_size} elements, which is "
                f"not a multiple of {texture_height} x {texture_width}"
            )
        channel_count = level_size // extent
    return _Texture.apply(
        materialize(pyramid), level_offsets, int(texture_height), int(texture_width),
        int(channel_count), materialize(coordinates),
        None if derivatives is None else materialize(derivatives),
        None if lod is None else materialize(lod),
        int(filter), int(mip_filter), int(wrap_u), int(wrap_v),
        None if border is None else materialize(border),
    )
