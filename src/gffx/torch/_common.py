"""Shared conversion and error-mapping helpers for the PyTorch adapter.

Validation lives here rather than in the native translation unit for a concrete reason: the
LibTorch Stable ABI offers no way to raise a specific Python exception type, so anything checked
across that boundary can only surface as a RuntimeError. Checking here lets each condition raise
the type fixed by TORCH_ADAPTER_ACCEPTANCE_V0_1.md section 2 and carry a message written for a
person rather than for a parser.

The native layer still validates everything it is given. This is a second, earlier check with
better diagnostics, not a replacement for the kernel's own; the two disagreeing would be an adapter
defect, which is why the native side reports that case as internal rather than accommodating it.
"""

from __future__ import annotations

import re
from typing import Tuple

import torch

# Status codes are fixed by the ABI and mapped by section 5 of the record. The mapping is a
# lookup rather than a chain of conditionals so that adding a status is a one-line change and an
# unmapped one is visibly absent.
_STATUS_EXCEPTIONS = {
    1: ValueError,            # INVALID_ARGUMENT
    2: NotImplementedError,   # UNSUPPORTED
    3: RuntimeError,          # INSUFFICIENT_WORKSPACE: the adapter sizes it, so this is our bug
    4: OverflowError,         # OVERFLOW
    5: RuntimeError,          # BACKEND_FAILURE
    6: ImportError,           # ABI_MISMATCH: the loaded core does not match the built adapter
    7: RuntimeError,          # INTERNAL_ERROR
}

_STATUS_NAMES = {
    1: "INVALID_ARGUMENT",
    2: "UNSUPPORTED",
    3: "INSUFFICIENT_WORKSPACE",
    4: "OVERFLOW",
    5: "BACKEND_FAILURE",
    6: "ABI_MISMATCH",
    7: "INTERNAL_ERROR",
}

_STATUS_PATTERN = re.compile(r"gffx-status:(\d+):\s*(.*)", re.DOTALL)

VERTEX_DTYPES = (torch.float32, torch.float64)


def translate_native_error(error: Exception) -> Exception:
    """Map a native failure onto the exception type the record fixes for its status.

    The native layer encodes the status in the message because it cannot raise a typed Python
    exception across the Stable ABI. An unrecognised status becomes a RuntimeError naming the
    numeric code rather than being silently downgraded to success or to a generic message.
    """
    text = str(error)
    match = _STATUS_PATTERN.search(text)
    if match is None:
        return error
    status = int(match.group(1))
    diagnostic = match.group(2).strip() or "the operation reported no diagnostic"
    name = _STATUS_NAMES.get(status, "status %d" % status)
    exception_type = _STATUS_EXCEPTIONS.get(status, RuntimeError)
    return exception_type("gffx %s: %s" % (name, diagnostic))


def materialize(tensor: torch.Tensor) -> torch.Tensor:
    """Return a densely strided copy when the tensor has a non-positive stride.

    An incoming cotangent is often an expanded view with stride 0: `loss.sum().backward()`
    broadcasts a scalar one across the output, and broadcasting is expressed as a zero stride
    rather than as a copy. `.contiguous()` alone does not fix this, because torch treats a
    size-1 dimension as contiguous whatever its stride, so a single-element cotangent keeps
    stride 0 and reaches the ABI, which requires positive strides. That is why this check is on
    the stride rather than on `is_contiguous()`.

    The copy happens only when needed, so the ordinary multi-face path still passes the caller's
    memory through untouched.
    """
    if any(stride <= 0 for stride in tensor.stride()):
        dense = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        dense.copy_(tensor)
        return dense
    return tensor.contiguous()


def check_vertices(vertices: torch.Tensor, name: str = "vertices") -> None:
    """Reject anything the conversion boundary does not accept, with the documented type."""
    if not isinstance(vertices, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor, not %s" % (name, type(vertices).__name__))
    if vertices.device.type not in ("cpu", "cuda"):
        raise ValueError(
            "%s must be on the cpu or cuda device; received a tensor on %s"
            % (name, vertices.device)
        )
    if vertices.dtype not in VERTEX_DTYPES:
        raise TypeError(
            "%s must be float32 or float64, which selects the computation dtype; received %s"
            % (name, vertices.dtype)
        )
    if vertices.dim() != 2 or vertices.shape[1] != 3:
        raise ValueError("%s must have shape [V, 3]; received %s" % (name, tuple(vertices.shape)))
    if any(stride <= 0 for stride in vertices.stride()):
        # A broadcast or expanded view. Caught separately because torch reports a size-1
        # dimension as contiguous whatever its stride, so the check below would miss it.
        raise ValueError(
            "%s has a non-positive stride, which means it is a broadcast or expanded view rather "
            "than real storage; gffx requires densely strided input. Call .contiguous() or "
            ".clone() explicitly if a copy is acceptable." % (name,)
        )
    if not vertices.is_contiguous():
        # Deliberately not repaired with .contiguous(): that is a hidden allocation and a hidden
        # copy, which the streaming surface forbids and which a caller tuning a frame loop must be
        # able to see.
        raise ValueError(
            "%s must be dense and C-contiguous; received a non-contiguous view. Call "
            ".contiguous() explicitly if a copy is acceptable." % (name,)
        )


def check_faces(faces: torch.Tensor, name: str = "faces") -> None:
    if not isinstance(faces, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor, not %s" % (name, type(faces).__name__))
    if faces.device.type not in ("cpu", "cuda"):
        raise ValueError(
            "%s must be on the cpu or cuda device; received a tensor on %s" % (name, faces.device)
        )
    if faces.dtype != torch.int32:
        # int64 is what a caller most often has, and a narrowing conversion can drop indices with
        # no local evidence, so it is refused by name rather than converted.
        raise TypeError(
            "%s must be int32, the packed index dtype every gffx operation uses; received %s. "
            "Convert explicitly with faces.to(torch.int32) after checking the index range."
            % (name, faces.dtype)
        )
    if faces.dim() != 2 or faces.shape[1] != 3:
        raise ValueError("%s must have shape [F, 3]; received %s" % (name, tuple(faces.shape)))
    if not faces.is_contiguous():
        raise ValueError("%s must be dense and C-contiguous" % (name,))


def check_eps(eps: float) -> float:
    value = float(eps)
    if not (value >= 0.0) or value != value or value == float("inf"):
        raise ValueError("eps must be finite and non-negative; received %r" % (eps,))
    return value


def check_same_device(*tensors: torch.Tensor) -> None:
    """Every tensor in one call must live on one device.

    Checked here rather than left to the backend because the backend can only report the symptom.
    A CUDA kernel handed a CPU tensor says the view is not a device view, which is true and
    unhelpful; naming the mismatch points at what the caller actually did. GFFX never moves data
    between devices on a caller's behalf, so a mismatch is always a caller error rather than
    something to repair silently.
    """
    devices = {tensor.device for tensor in tensors if isinstance(tensor, torch.Tensor)}
    if len(devices) > 1:
        raise ValueError(
            "every tensor in one call must be on the same device; received %s. gffx never moves "
            "data between devices for you, so move them yourself with .to(device)."
            % (", ".join(str(device) for device in sorted(devices, key=str)),)
        )


def check_pair(vertices: torch.Tensor, faces: torch.Tensor, eps: float) -> Tuple[
    torch.Tensor, torch.Tensor, float
]:
    check_vertices(vertices)
    check_faces(faces)
    check_same_device(vertices, faces)
    return vertices, faces, check_eps(eps)
