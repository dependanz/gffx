"""Minimal ctypes binding to the gffx C ABI, used to generate the showcase figures.

This exists rather than using the PyTorch adapter because the adapter requires PyTorch 2.10 and a
CPU-only 2.9.1 is what is installed here. Driving the shipped C ABI directly turned out to be the
better demonstration anyway: it shows the library is usable from anything that can call C, with no
framework in the loop, which is the portability claim the project actually makes.

It is deliberately not a general binding. It covers the operations the figures need and validates
nothing the library already validates, because a wrapper that re-checks arguments would be
asserting its own opinion of the contract rather than exercising the library's.
"""

import ctypes
import os

import numpy as np

ABI_VERSION = (1 << 16) | 0

DTYPE_FLOAT32 = 1
DTYPE_FLOAT64 = 2
DTYPE_INT32 = 3
DTYPE_BOOL = 5

DEVICE_CPU = 1

TENSOR_READ_ONLY = 1
TENSOR_OUTPUT = 2

FILTER_NEAREST, FILTER_BILINEAR = 1, 2
MIP_NEAREST, MIP_LINEAR = 1, 2
WRAP_REPEAT, WRAP_CLAMP, WRAP_MIRROR, WRAP_BORDER = 1, 2, 3, 4
CULL_NONE, CULL_BACK, CULL_FRONT = 1, 2, 3

_NUMPY_TO_GFFX = {
    np.dtype(np.float32): DTYPE_FLOAT32,
    np.dtype(np.float64): DTYPE_FLOAT64,
    np.dtype(np.int32): DTYPE_INT32,
    np.dtype(np.uint8): DTYPE_BOOL,
}


class TensorView(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("byte_offset", ctypes.c_uint64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("rank", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("device_type", ctypes.c_uint32),
        ("device_index", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class ExecutionContext(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_type", ctypes.c_uint32),
        ("device_index", ctypes.c_int32),
        ("stream", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class DiagnosticBuffer(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("data", ctypes.c_char_p),
        ("capacity_bytes", ctypes.c_uint64),
        ("required_bytes", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class GffxError(RuntimeError):
    pass


def load(library_path=None):
    """Load gffx_core. The default looks beside this file first, then in the CUDA build tree."""
    candidates = []
    if library_path:
        candidates.append(library_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "gffx_core.dll"))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates.append(os.path.join(root, "build", "cuda", "Debug", "gffx_core.dll"))
    candidates.append(os.path.join(root, "build", "phase4-red-1", "Debug", "gffx_core.dll"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return ctypes.CDLL(candidate)
    raise GffxError("gffx_core not found; tried:\n  " + "\n  ".join(candidates))


class Session(object):
    """Holds the loaded library plus the scratch a call needs, so figures stay readable."""

    def __init__(self, library=None):
        self.lib = library if library is not None else load()
        self._keepalive = []
        self.context = ExecutionContext()
        self.context.struct_size = ctypes.sizeof(ExecutionContext)
        self.context.abi_version = ABI_VERSION
        self.context.device_type = DEVICE_CPU
        self._message = ctypes.create_string_buffer(1024)
        self.diagnostic = DiagnosticBuffer()
        self.diagnostic.struct_size = ctypes.sizeof(DiagnosticBuffer)
        self.diagnostic.abi_version = ABI_VERSION
        self.diagnostic.data = ctypes.cast(self._message, ctypes.c_char_p)
        self.diagnostic.capacity_bytes = ctypes.sizeof(self._message)

    def view(self, array, output=False):
        """Wrap a contiguous numpy array. The array must outlive every call that reads it, so a
        reference is kept here rather than relying on the caller to hold one."""
        array = np.ascontiguousarray(array)
        self._keepalive.append(array)
        shape = (ctypes.c_int64 * array.ndim)(*array.shape)
        strides = (ctypes.c_int64 * array.ndim)(
            *[s // array.itemsize for s in array.strides])
        self._keepalive.extend([shape, strides])
        v = TensorView()
        v.struct_size = ctypes.sizeof(TensorView)
        v.abi_version = ABI_VERSION
        v.data = array.ctypes.data_as(ctypes.c_void_p)
        v.shape = shape
        v.strides = strides
        v.rank = array.ndim
        v.dtype = _NUMPY_TO_GFFX[array.dtype]
        v.device_type = DEVICE_CPU
        v.flags = TENSOR_OUTPUT if output else TENSOR_READ_ONLY
        return v

    def check(self, status, what):
        if status != 0:
            text = self._message.value.decode("utf-8", "replace")
            raise GffxError("%s failed with status %d: %s" % (what, status, text))
        self._message.value = b""

    # --- operations the figures use -------------------------------------------------------

    def texture_pyramid(self, texture, levels=0):
        """Returns (pyramid, offsets). Capacity is sized generously and trimmed by the offsets the
        library reports, so this never has to duplicate the level-chain arithmetic."""
        h, w, c = texture.shape
        capacity = 4 * h * w * c + 16
        chain = 1
        hh, ww = h, w
        while hh > 1 or ww > 1:
            hh = hh // 2 if hh > 1 else 1
            ww = ww // 2 if ww > 1 else 1
            chain += 1
        count = chain if levels == 0 else levels
        pyramid = np.zeros(capacity, dtype=texture.dtype)
        offsets = np.zeros(count + 1, dtype=np.int32)
        tv, pv, ov = self.view(texture), self.view(pyramid, True), self.view(offsets, True)
        self.check(self.lib.gffx_render_texture_pyramid(
            ctypes.byref(tv), ctypes.c_int64(levels), ctypes.byref(self.context),
            ctypes.byref(pv), ctypes.byref(ov), None, ctypes.byref(self.diagnostic)),
            "render.texture_pyramid")
        return pyramid, offsets

    def texture(self, pyramid, offsets, height, width, coordinates, channels,
                filter=FILTER_BILINEAR, mip_filter=MIP_NEAREST,
                wrap_u=WRAP_REPEAT, wrap_v=WRAP_REPEAT, border=None, lod=None,
                derivatives=None):
        n = coordinates.shape[0]
        samples = np.zeros((n, channels), dtype=pyramid.dtype)
        if border is None:
            border = np.zeros(channels, dtype=pyramid.dtype)
        pv, ov = self.view(pyramid), self.view(offsets)
        cv, bv = self.view(coordinates), self.view(border)
        sv = self.view(samples, True)
        lv = self.view(lod) if lod is not None else None
        dv = self.view(derivatives) if derivatives is not None else None
        self.check(self.lib.gffx_render_texture(
            ctypes.byref(pv), ctypes.byref(ov),
            ctypes.c_int64(height), ctypes.c_int64(width), ctypes.byref(cv),
            ctypes.byref(dv) if dv is not None else None,
            ctypes.byref(lv) if lv is not None else None,
            ctypes.c_uint32(filter), ctypes.c_uint32(mip_filter),
            ctypes.c_uint32(wrap_u), ctypes.c_uint32(wrap_v), ctypes.byref(bv),
            ctypes.byref(self.context), ctypes.byref(sv), None,
            ctypes.byref(self.diagnostic)), "render.texture")
        return samples

    # --- the geometry and rendering chain ---------------------------------------------------

    def transform_points(self, points, matrices, offsets):
        homogeneous = np.zeros((points.shape[0], 4), dtype=points.dtype)
        pv, mv, ov = self.view(points), self.view(matrices), self.view(offsets)
        hv = self.view(homogeneous, True)
        self.check(self.lib.gffx_transforms_transform_points(
            ctypes.byref(pv), ctypes.byref(mv), ctypes.byref(ov), ctypes.byref(self.context),
            ctypes.byref(hv), None, ctypes.byref(self.diagnostic)),
            "transforms.transform_points")
        return homogeneous

    def perspective_divide(self, homogeneous, eps=1e-9):
        ndc = np.zeros((homogeneous.shape[0], 3), dtype=homogeneous.dtype)
        valid = np.zeros(homogeneous.shape[0], dtype=np.uint8)
        hv = self.view(homogeneous)
        nv, vv = self.view(ndc, True), self.view(valid, True)
        self.check(self.lib.gffx_transforms_perspective_divide(
            ctypes.byref(hv), ctypes.c_double(eps), ctypes.byref(self.context),
            ctypes.byref(nv), ctypes.byref(vv), None, ctypes.byref(self.diagnostic)),
            "transforms.perspective_divide")
        return ndc, valid

    def rasterize(self, ndc, faces, vertex_offsets, face_offsets, height, width,
                  faces_per_pixel=1, blur_radius_px=0.0, cull_mode=CULL_NONE, eps=1e-12):
        batch = vertex_offsets.shape[0] - 1
        shape = (batch, height, width, faces_per_pixel)
        face_index = np.zeros(shape, dtype=np.int32)
        barycentric = np.zeros(shape + (3,), dtype=ndc.dtype)
        depth = np.zeros(shape, dtype=ndc.dtype)
        signed_distance = np.zeros(shape, dtype=ndc.dtype)
        nv, fv = self.view(ndc), self.view(faces)
        vo, fo = self.view(vertex_offsets), self.view(face_offsets)
        iv = self.view(face_index, True)
        bv = self.view(barycentric, True)
        dv = self.view(depth, True)
        sv = self.view(signed_distance, True)
        self.check(self.lib.gffx_render_rasterize(
            ctypes.byref(nv), ctypes.byref(fv), ctypes.byref(vo), ctypes.byref(fo),
            ctypes.c_int64(height), ctypes.c_int64(width), ctypes.c_int64(faces_per_pixel),
            ctypes.c_double(blur_radius_px), ctypes.c_uint32(cull_mode), ctypes.c_double(eps),
            ctypes.byref(self.context), ctypes.byref(iv), ctypes.byref(bv), ctypes.byref(dv),
            ctypes.byref(sv), None, ctypes.byref(self.diagnostic)), "render.rasterize")
        return face_index, barycentric, depth, signed_distance

    def interpolate(self, face_index, barycentric, face_attributes):
        channels = face_attributes.shape[2]
        attributes = np.zeros(face_index.shape + (channels,), dtype=face_attributes.dtype)
        iv, bv = self.view(face_index), self.view(barycentric)
        av = self.view(face_attributes)
        ov = self.view(attributes, True)
        self.check(self.lib.gffx_render_interpolate(
            ctypes.byref(iv), ctypes.byref(bv), ctypes.byref(av), ctypes.byref(self.context),
            ctypes.byref(ov), None, ctypes.byref(self.diagnostic)), "render.interpolate")
        return attributes

    def interpolate_backward(self, face_index, barycentric, face_attributes, grad_attributes):
        grad_barycentric = np.zeros_like(barycentric)
        grad_face_attributes = np.zeros_like(face_attributes)
        iv, bv = self.view(face_index), self.view(barycentric)
        av, gv = self.view(face_attributes), self.view(grad_attributes)
        gb = self.view(grad_barycentric, True)
        ga = self.view(grad_face_attributes, True)
        self.check(self.lib.gffx_render_interpolate_backward(
            ctypes.byref(iv), ctypes.byref(bv), ctypes.byref(av), ctypes.byref(gv),
            ctypes.byref(self.context), ctypes.byref(gb), ctypes.byref(ga), None,
            ctypes.byref(self.diagnostic)), "render.interpolate_backward")
        return grad_barycentric, grad_face_attributes

    def rasterize_backward(self, ndc, faces, height, width, face_index,
                           grad_barycentric, grad_depth, grad_signed_distance):
        grad_ndc = np.zeros_like(ndc)
        nv, fv = self.view(ndc), self.view(faces)
        iv = self.view(face_index)
        gb = self.view(grad_barycentric)
        gd = self.view(grad_depth)
        gs = self.view(grad_signed_distance)
        gn = self.view(grad_ndc, True)
        self.check(self.lib.gffx_render_rasterize_backward(
            ctypes.byref(nv), ctypes.byref(fv),
            ctypes.c_int64(height), ctypes.c_int64(width), ctypes.byref(iv),
            ctypes.byref(gb), ctypes.byref(gd), ctypes.byref(gs),
            ctypes.byref(self.context), ctypes.byref(gn), None,
            ctypes.byref(self.diagnostic)), "render.rasterize_backward")
        return grad_ndc
