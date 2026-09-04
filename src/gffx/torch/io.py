"""Mesh interchange on the PyTorch CPU backend.

Semantics belong to PLY_ACCEPTANCE_V0_1.md. This module reads the triangle-template subset the
migration corpus requires, which is what replaces a `Mesh(filename=...)` call against an
unmaintained package.

Python opens files; the native core parses bytes. That is the core's own boundary and it is kept
rather than flattened, so the same parser serves a file, a memory-mapped region, an archive member
and a network download without a second entry point. It also means `read` accepts bytes you already
have, which `read_file` is a thin convenience over.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Tuple, Union

import torch

from ._common import translate_native_error

__all__ = ["read", "read_file", "probe", "probe_file", "PlyHeader",
           "FORMAT_ASCII", "FORMAT_BINARY_LITTLE_ENDIAN"]

FORMAT_ASCII = 0
FORMAT_BINARY_LITTLE_ENDIAN = 1

_FORMAT_NAMES = {FORMAT_ASCII: "ascii", FORMAT_BINARY_LITTLE_ENDIAN: "binary_little_endian"}


class PlyHeader(NamedTuple):
    """What `probe` reports without parsing the body."""

    format: int
    vertex_count: int
    face_count: int

    @property
    def format_name(self) -> str:
        return _FORMAT_NAMES.get(self.format, "unknown")


def _as_buffer(data: Union[bytes, bytearray, memoryview, torch.Tensor]) -> torch.Tensor:
    """Wrap the caller's bytes as a uint8 tensor without copying where possible."""
    if isinstance(data, torch.Tensor):
        if data.dtype != torch.uint8:
            raise TypeError("a tensor buffer must be uint8; received %s" % (data.dtype,))
        if data.dim() != 1:
            raise ValueError("a tensor buffer must be 1-D; received shape %s"
                             % (tuple(data.shape),))
        if not data.is_contiguous():
            raise ValueError("a tensor buffer must be contiguous")
        if data.device.type != "cpu":
            raise ValueError("a tensor buffer must be on the cpu device")
        return data
    if isinstance(data, (bytes, bytearray, memoryview)):
        # frombuffer borrows for bytearray and memoryview; bytes is immutable so torch copies it.
        return torch.frombuffer(bytearray(data), dtype=torch.uint8)
    raise TypeError(
        "data must be bytes, bytearray, memoryview, or a uint8 tensor; received %s"
        % (type(data).__name__,)
    )


def probe(data: Union[bytes, bytearray, memoryview, torch.Tensor]) -> PlyHeader:
    """Read only the header, reporting the format and element counts."""
    buffer = _as_buffer(data)
    try:
        format_code, vertex_count, face_count = torch.ops.gffx.ply_probe(buffer)
    except RuntimeError as error:
        raise translate_native_error(error) from None
    return PlyHeader(int(format_code), int(vertex_count), int(face_count))


def read(
    data: Union[bytes, bytearray, memoryview, torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse a PLY triangle template into ``(vertices[V,3], faces[F,3])``.

    ``dtype`` selects the vertex dtype and must be ``float32`` or ``float64``; faces are always
    ``int32``, the packed index dtype every gffx operation takes. A ``float32`` result is rounded
    once, at the point of storage, rather than twice through an intermediate.

    Supports ASCII and binary little-endian files whose vertex element carries ``x``, ``y`` and
    ``z`` in any position, with any number of other properties, and whose face element carries a
    triangular ``vertex_indices`` list. Anything outside that subset raises rather than being
    approximated, because a mesh that loads wrong produces plausible geometry and wrong results.

    Indices are checked for sign and int32 range but **not** against the vertex count; use
    `gffx.torch.mesh.validate` for that, which surveys the whole mesh and reports counts.
    """
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64; received %r" % (dtype,))
    buffer = _as_buffer(data)
    try:
        return torch.ops.gffx.ply_read(buffer, dtype == torch.float64)
    except RuntimeError as error:
        raise translate_native_error(error) from None


def probe_file(path: Union[str, os.PathLike]) -> PlyHeader:
    """Read a file's header without parsing its body."""
    with open(path, "rb") as handle:
        return probe(handle.read())


def read_file(
    path: Union[str, os.PathLike], dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read a PLY file into ``(vertices[V,3], faces[F,3])``.

    A thin convenience over `read`: Python opens the file and the native core parses the bytes.
    """
    with open(path, "rb") as handle:
        return read(handle.read(), dtype=dtype)
