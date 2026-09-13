"""Caller-owned animation outputs and scratch; inference only, not callback safety."""
import torch
from ._animation import _floating, _index
from ._common import translate_native_error


def _validate(name, inputs):
    expected = {"blend_shapes": 3, "rodrigues": 1, "forward_kinematics": 2,
                "skin_lbs": 3, "evaluate_barycentric": 4}
    if name not in expected or len(inputs) != expected[name]:
        raise ValueError("unknown animation operation or wrong operand count")
    reference = inputs[0]
    if name == "blend_shapes":
        t, d, w = inputs
        _floating(t, "template", (None, 3)); _floating(d, "deltas", (None, t.shape[0], 3), t)
        _floating(w, "weights", (None, d.shape[0]), t)
        return 1, w.shape[0], t.shape[0], d.shape[0], (w.shape[0], t.shape[0], 3)
    if name == "rodrigues":
        _floating(reference, "axis_angle")
        if not 1 <= reference.ndim < 64 or reference.shape[-1] != 3:
            raise ValueError("axis_angle must have shape [...,3] with rank below64")
        return 2, 1, reference.numel() // 3, 0, (*reference.shape, 3)
    if name == "forward_kinematics":
        local, parents = inputs
        _floating(local, "local", (None, None, 4, 4)); _index(parents, "parents", (local.shape[1],), local)
        return 3, local.shape[0], local.shape[1], 0, tuple(local.shape)
    _floating(reference, "vertices", (None, None, 3))
    if name == "skin_lbs":
        v, t, w = inputs
        _floating(t, "transforms", (v.shape[0], None, 4, 4), v)
        _floating(w, "weights", (v.shape[1], t.shape[1]), v)
        return 4, v.shape[0], v.shape[1], t.shape[1], tuple(v.shape)
    v, f, index, w = inputs
    _index(f, "faces", (None, 3), v); _index(index, "face_index", (None,), v)
    _floating(w, "barycentric", (index.shape[0], 3), v)
    return 5, v.shape[0], v.shape[1], index.shape[0], (v.shape[0], index.shape[0], 3)


def animation_workspace_size(operation, *inputs):
    """Query native scratch bytes for one of the five named animation operations."""
    op, b, v, k, _ = _validate(operation, inputs)
    try:
        return int(torch.ops.gffx._animation_workspace_size(inputs[0], op, b, v, k))
    except RuntimeError as error:
        raise translate_native_error(error) from None


def _out(name, inputs, out, workspace):
    _, _, _, _, shape = _validate(name, inputs)
    reference = inputs[0]
    # Output values may be uninitialized; validate layout/type without reading them.
    if not isinstance(out, torch.Tensor) or out.dtype != reference.dtype:
        raise TypeError("out must be a tensor matching the input dtype")
    if out.device != reference.device or tuple(out.shape) != shape:
        raise ValueError("out must match the operation's device and output shape")
    if not out.is_contiguous() or any(s <= 0 for s in out.stride()):
        raise ValueError("out must be dense C-contiguous storage with positive strides")
    if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8:
        raise TypeError("workspace must be a uint8 tensor")
    if workspace.ndim != 1 or not workspace.is_contiguous() or any(s <= 0 for s in workspace.stride()):
        raise ValueError("workspace must be a contiguous one-dimensional tensor")
    if workspace.device != reference.device:
        raise ValueError("workspace device must match inputs")
    if out.requires_grad or any(t.requires_grad for t in inputs):
        raise ValueError("animation _out is inference-only; detach inputs explicitly or use functional operations")
    try:
        getattr(torch.ops.gffx, name + "_out")(*inputs, out, workspace)
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return out


def blend_shapes_out(template, deltas, weights, *, out, workspace):
    return _out("blend_shapes", (template, deltas, weights), out, workspace)


def rodrigues_out(axis_angle, *, out, workspace):
    return _out("rodrigues", (axis_angle,), out, workspace)


def forward_kinematics_out(local, parents, *, out, workspace):
    return _out("forward_kinematics", (local, parents), out, workspace)


def skin_lbs_out(vertices, transforms, weights, *, out, workspace):
    return _out("skin_lbs", (vertices, transforms, weights), out, workspace)


def evaluate_barycentric_out(vertices, faces, face_index, barycentric, *, out, workspace):
    return _out("evaluate_barycentric", (vertices, faces, face_index, barycentric), out, workspace)
