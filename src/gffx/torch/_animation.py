"""Validation/autograd only; animation mathematics execute in the C11 core."""
from __future__ import annotations

import torch
from ._common import materialize, translate_native_error

_FINITE_CHUNK_ELEMENTS = 65536


def _floating(value, name, shape=None, reference=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{name} must be on CPU or CUDA")
    if reference is not None:
        if value.dtype != reference.dtype:
            raise TypeError(f"{name} must match the floating dtype")
        if value.device != reference.device:
            raise ValueError(f"{name} must match the device")
    if shape is not None and (value.ndim != len(shape) or any(
            wanted is not None and actual != wanted for actual, wanted in zip(value.shape, shape))):
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_contiguous() or any(stride <= 0 for stride in value.stride()):
        raise ValueError(f"{name} must have positive strides and dense C-contiguous storage")
    # Dense inputs can be viewed without a copy. Bound temporary masks and any
    # dtype-sized isfinite intermediates independently of corrective-basis size.
    # Each check stays on the caller's device; no host tensor fallback is used.
    flat = value.view(-1)
    for start in range(0, flat.numel(), _FINITE_CHUNK_ELEMENTS):
        if not bool(torch.isfinite(flat[start:start + _FINITE_CHUNK_ELEMENTS]).all()):
            raise ValueError(f"{name} must contain finite values")


class _BlendShapes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, template, deltas, weights):
        try:
            output = torch.ops.gffx.blend_shapes(template, deltas, weights)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        ctx.save_for_backward(template, deltas, weights)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        try:
            needed = ctx.needs_input_grad
            gradients = torch.ops.gffx.blend_shapes_backward(*ctx.saved_tensors, materialize(gradient), *needed)
            return tuple(value if need else None for value, need in zip(gradients, needed))
        except RuntimeError as error:
            raise translate_native_error(error) from None


def blend_shapes(template, deltas, weights):
    """Add shared displacements [K,V,3] using signed per-frame weights [B,K].

    The template is [V,3]; the fresh result is [B,V,3]. All three inputs retain
    first-order gradients. No normalization, promotion or transfer is performed.
    """
    _floating(template, "template", (None, 3))
    _floating(deltas, "deltas", (None, template.shape[0], 3), template)
    _floating(weights, "weights", (None, deltas.shape[0]), template)
    return _BlendShapes.apply(template, deltas, weights)


def _index(value, name, shape, reference):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int32:
        raise TypeError(f"{name} must be an int32 tensor")
    if value.device != reference.device:
        raise ValueError(f"{name} must match the geometry device")
    if value.ndim != len(shape) or any(w is not None and a != w for a, w in zip(value.shape, shape)):
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_contiguous() or any(s <= 0 for s in value.stride()):
        raise ValueError(f"{name} must have positive strides and dense C-contiguous storage")


class _NativeAnimation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, *inputs):
        ctx.name = name
        ctx.save_for_backward(*inputs)
        try:
            return getattr(torch.ops.gffx, name)(*inputs)
        except RuntimeError as error:
            raise translate_native_error(error) from None

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        try:
            needed = ctx.needs_input_grad[1:]
            flags = needed if ctx.name == "skin_lbs" else (needed[0], needed[3]) if ctx.name == "evaluate_barycentric" else ()
            result = getattr(torch.ops.gffx, ctx.name + "_backward")(*ctx.saved_tensors, materialize(gradient), *flags)
        except RuntimeError as error:
            raise translate_native_error(error) from None
        if ctx.name == "rodrigues":
            return None, result
        if ctx.name == "forward_kinematics":
            return None, result, None
        if ctx.name == "skin_lbs":
            return None, *(value if need else None for value, need in zip(result, needed))
        return None, result[0] if needed[0] else None, None, None, result[1] if needed[3] else None


def rodrigues(axis_angle):
    """Right-handed exponential map [...,3] radians -> [...,3,3], differentiable at zero."""
    _floating(axis_angle, "axis_angle")
    if not 1 <= axis_angle.ndim < 64 or axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle must have shape [...,3] with rank below 64")
    return _NativeAnimation.apply("rodrigues", axis_angle)


def forward_kinematics(local, parents):
    """Multiply an arbitrary-order forest of shared parents; root parent is -1.

    local[B,J,4,4] contains general matrices. Caller owns inverse-bind correction
    and affine restrictions; each local matrix entry retains its derivative.
    """
    _floating(local, "local", (None, None, 4, 4))
    _index(parents, "parents", (local.shape[1],), local)
    return _NativeAnimation.apply("forward_kinematics", local, parents)


def skin_lbs(vertices, transforms, weights):
    """Dense linear blend skinning with shared, unnormalized weights[V,J].

    vertices[B,V,3] and transforms[B,J,4,4] produce [B,V,3]; only the upper3x4
    transform rows act on points. Bind-pose inverses are composed by the caller.
    """
    _floating(vertices, "vertices", (None, None, 3))
    _floating(transforms, "transforms", (vertices.shape[0], None, 4, 4), vertices)
    _floating(weights, "weights", (vertices.shape[1], transforms.shape[1]), vertices)
    return _NativeAnimation.apply("skin_lbs", vertices, transforms, weights)


def evaluate_barycentric(vertices, faces, face_index, barycentric):
    """Evaluate shared unstructured landmarks, with vertex and coefficient gradients.

    No clipping or normalization is performed. faces[F,3] and face_index[N]
    are int32; barycentric[N,3] applies to every frame of vertices[B,V,3].
    """
    _floating(vertices, "vertices", (None, None, 3))
    _index(faces, "faces", (None, 3), vertices)
    _index(face_index, "face_index", (None,), vertices)
    _floating(barycentric, "barycentric", (face_index.shape[0], 3), vertices)
    return _NativeAnimation.apply("evaluate_barycentric", vertices, faces, face_index, barycentric)
