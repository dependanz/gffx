"""Wire framing of DASHBOARD_CONTRACT_V0_1.md section 2.

Text frames carry one JSON envelope. Binary frames carry a 4-byte little-endian header length, the
JSON header, then the raw bytes of every array the header lists, in order, each little-endian and
C-contiguous. Nothing is base64-encoded, and nothing here imports a framework: a tensor, a NumPy
array, or any buffer-protocol object is reduced to ``(dtype, shape, bytes)`` at the edge and
reconstituted on the far side only when the reader asks for it.
"""

from __future__ import annotations

import json
import struct
from typing import Any

BINARY_KINDS = frozenset({"mesh", "points", "image"})

#: Array fields per binary kind, in the order the wire carries them; the first is mandatory.
ARRAY_FIELDS = {
    "mesh": ("vertices", "faces", "colors", "normals"),
    "points": ("positions", "colors", "radii"),
    "image": ("data",),
}

_ITEMSIZE = {
    "float32": 4, "float64": 8, "int32": 4, "int64": 8, "uint32": 4, "uint8": 1, "uint16": 2, "int16": 2,
}
_STRUCT_CODE = {
    "float32": "f", "float64": "d", "int32": "i", "int64": "q", "uint32": "I", "uint8": "B", "uint16": "H", "int16": "h",
}


class ProtocolError(RuntimeError):
    """A message the far side rejected, or a frame that does not follow section 2."""


class ArrayView:
    """One array as carried on the wire: dtype name, shape, and its bytes."""

    __slots__ = ("dtype", "shape", "data")

    def __init__(self, dtype: str, shape: tuple[int, ...], data: bytes | memoryview) -> None:
        if dtype not in _ITEMSIZE:
            raise ProtocolError("unsupported array dtype %r" % (dtype,))
        expected = _ITEMSIZE[dtype]
        for extent in shape:
            expected *= int(extent)
        if len(data) != expected:
            raise ProtocolError(
                "array of dtype %s and shape %r needs %d bytes, received %d" % (dtype, tuple(shape), expected, len(data)))
        self.dtype = dtype
        self.shape = tuple(int(extent) for extent in shape)
        self.data = data if isinstance(data, bytes) else bytes(data)

    def to_numpy(self):
        """The array as NumPy when NumPy is importable; NumPy is never required."""
        import numpy  # optional, resolved here so the base import stays dependency-free
        # A copy, so the caller owns a writable array rather than a view of immutable bytes.
        return numpy.frombuffer(self.data, dtype=self.dtype).reshape(self.shape).copy()

    def to_list(self) -> list:
        """The array as nested lists using only the standard library."""
        count = len(self.data) // _ITEMSIZE[self.dtype]
        flat = list(struct.unpack("<%d%s" % (count, _STRUCT_CODE[self.dtype]), self.data))
        return _nest(flat, self.shape)

    def meta(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape)}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArrayView) and (self.dtype, self.shape, self.data) == (other.dtype, other.shape, other.data)

    def __repr__(self) -> str:
        return "ArrayView(%s%r)" % (self.dtype, self.shape)


def _nest(flat: list, shape: tuple[int, ...]) -> list:
    if len(shape) <= 1:
        return flat
    inner = 1
    for extent in shape[1:]:
        inner *= extent
    return [_nest(flat[i * inner:(i + 1) * inner], shape[1:]) for i in range(shape[0])]


def as_array(obj: Any, *, dtype: str | None = None) -> ArrayView:
    """Reduce a tensor, NumPy array, buffer, or nested list to an ``ArrayView``.

    A framework tensor is moved to the host once and never modified or retained. ``dtype`` names a
    required wire dtype; an integer input is converted to it when the name is an integer type
    and the values fit, which is how int32 or int64 face indices become the contract's uint32.
    """
    if isinstance(obj, ArrayView):
        return obj
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):  # a framework tensor
        tensor = obj.detach().cpu().contiguous()
        try:
            obj = tensor.numpy()
        except Exception:  # NumPy absent: go through Python lists
            return _from_list(tensor.tolist(), str(tensor.dtype).split(".")[-1], dtype)
    if hasattr(obj, "__array__") and not hasattr(obj, "dtype"):
        obj = obj.__array__()
    if hasattr(obj, "dtype") and hasattr(obj, "shape") and hasattr(obj, "tobytes"):
        name = str(obj.dtype)
        if dtype is not None and dtype != name:
            obj = obj.astype(dtype)
            name = dtype
        if name not in _ITEMSIZE:
            raise ProtocolError("unsupported array dtype %r" % (name,))
        contiguous = obj if getattr(obj.flags, "c_contiguous", True) else obj.copy(order="C")
        return ArrayView(name, tuple(int(extent) for extent in contiguous.shape), contiguous.tobytes())
    if isinstance(obj, (bytes, bytearray, memoryview)):
        if dtype is None:
            raise ProtocolError("raw bytes need an explicit dtype")
        view = memoryview(obj)
        return ArrayView(dtype, (len(view) // _ITEMSIZE[dtype],), view.tobytes())
    if isinstance(obj, (list, tuple)):
        return _from_list(obj, None, dtype)
    raise ProtocolError("cannot reduce %s to an array" % (type(obj).__name__,))


def _from_list(nested: Any, source_dtype: str | None, dtype: str | None) -> ArrayView:
    shape: list[int] = []
    probe = nested
    while isinstance(probe, (list, tuple)):
        shape.append(len(probe))
        probe = probe[0] if probe else None
    flat: list = []

    def walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for element in item:
                walk(element)
        else:
            flat.append(item)

    walk(nested)
    if dtype is None:
        dtype = source_dtype or ("float32" if any(isinstance(v, float) for v in flat) else "int32")
    if dtype not in _ITEMSIZE:
        raise ProtocolError("unsupported array dtype %r" % (dtype,))
    return ArrayView(dtype, tuple(shape), struct.pack("<%d%s" % (len(flat), _STRUCT_CODE[dtype]), *flat))


def encode_text(envelope: dict[str, Any]) -> str:
    return json.dumps(envelope, separators=(",", ":"), allow_nan=True)


def decode_text(payload: str | bytes) -> dict[str, Any]:
    try:
        envelope = json.loads(payload)
    except ValueError as error:
        raise ProtocolError("text frame is not JSON: %s" % (error,)) from None
    if not isinstance(envelope, dict) or "t" not in envelope:
        raise ProtocolError("text frame is not an envelope with a type")
    return envelope


def encode_binary_parts(envelope: dict[str, Any], arrays: list[tuple[str, ArrayView]]) -> list[bytes]:
    """The parts of a binary frame, uncopied: header length, JSON header, then each array's bytes.

    The caller's thread pays only for the small header; whoever writes the socket joins the parts.
    """
    header = dict(envelope)
    header["arrays"] = [
        {"name": name, "dtype": view.dtype, "shape": list(view.shape), "nbytes": len(view.data)} for name, view in arrays
    ]
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    parts = [struct.pack("<I", len(header_bytes)), header_bytes]
    parts.extend(view.data for _, view in arrays)
    return parts


def encode_binary(envelope: dict[str, Any], arrays: list[tuple[str, ArrayView]]) -> bytes:
    """A binary frame: header length, JSON header naming the arrays in order, then their bytes."""
    return b"".join(encode_binary_parts(envelope, arrays))


def decode_binary(payload: bytes | memoryview) -> tuple[dict[str, Any], list[tuple[str, ArrayView]]]:
    view = memoryview(payload)
    if len(view) < 4:
        raise ProtocolError("binary frame shorter than its header length")
    (header_length,) = struct.unpack("<I", view[:4])
    if 4 + header_length > len(view):
        raise ProtocolError("binary frame header runs past the frame")
    try:
        header = json.loads(bytes(view[4:4 + header_length]).decode("utf-8"))
    except ValueError as error:
        raise ProtocolError("binary frame header is not JSON: %s" % (error,)) from None
    if not isinstance(header, dict) or "t" not in header or not isinstance(header.get("arrays"), list):
        raise ProtocolError("binary frame header is not an envelope listing arrays")
    arrays: list[tuple[str, ArrayView]] = []
    offset = 4 + header_length
    for entry in header["arrays"]:
        try:
            name, dtype, shape, nbytes = entry["name"], entry["dtype"], entry["shape"], int(entry["nbytes"])
        except (KeyError, TypeError, ValueError):
            raise ProtocolError("binary frame array entry is malformed: %r" % (entry,)) from None
        if offset + nbytes > len(view):
            raise ProtocolError("binary frame array %r runs past the frame" % (name,))
        arrays.append((name, ArrayView(dtype, tuple(shape), view[offset:offset + nbytes])))
        offset += nbytes
    if offset != len(view):
        raise ProtocolError("binary frame carries %d trailing bytes" % (len(view) - offset,))
    del header["arrays"]
    return header, arrays


def value_from_arrays(kind: str, header: dict[str, Any], arrays: list[tuple[str, ArrayView]]) -> dict[str, Any]:
    """The stored value of a binary kind: its arrays, their order, and the header's metadata."""
    if kind not in ARRAY_FIELDS:
        raise ProtocolError("kind %r carries no arrays" % (kind,))
    allowed = ARRAY_FIELDS[kind]
    names = [name for name, _ in arrays]
    if not names or names[0] != allowed[0]:
        raise ProtocolError("kind %r needs %r as its first array" % (kind, allowed[0]))
    for name in names:
        if name not in allowed:
            raise ProtocolError("kind %r does not carry an array named %r" % (kind, name))
    value: dict[str, Any] = {"array_order": names}
    for name, view in arrays:
        value[name] = view
    for key, item in header.items():
        if key not in ("t", "id", "path", "kind", "step", "bound") and key not in value:
            value[key] = item
    return value


def split_value(value: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, ArrayView]]]:
    """The inverse of ``value_from_arrays``: metadata for the header, arrays in wire order."""
    arrays = [(name, value[name]) for name in value["array_order"]]
    meta = {key: item for key, item in value.items() if key != "array_order" and not isinstance(item, ArrayView)}
    return meta, arrays


def json_safe(value: Any) -> Any:
    """A value with every ``ArrayView`` replaced by its metadata, for snapshots and pages."""
    if isinstance(value, ArrayView):
        return value.meta()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
