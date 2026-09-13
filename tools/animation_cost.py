"""Bounded native per-frame animation cost; synthetic dimensions, never FLAME assets.

One device/dtype/mode per supervised process. Validation, ordered reductions and
CUDA completion are included in timings; fixture generation and allocation are not.
This is not a callback-safety or hard real-time claim.
"""
from __future__ import annotations
import argparse
import ctypes as C
import os
from pathlib import Path
import statistics
import time

from tools import material_loss_cost as supervisor
from tools import shading_cuda_closeout as artifact

ABI = 0x10000


class Context(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32), ("device_type", C.c_uint32),
                ("device_index", C.c_int32), ("stream", C.c_void_p), ("flags", C.c_uint32),
                ("reserved0", C.c_uint32), ("reserved", C.c_uint64 * 4)]


class Buffer(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32), ("data", C.c_void_p),
                ("capacity_bytes", C.c_uint64), ("alignment", C.c_uint64), ("device_type", C.c_uint32),
                ("device_index", C.c_int32), ("flags", C.c_uint32), ("reserved0", C.c_uint32),
                ("reserved", C.c_uint64 * 2)]


class View(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32), ("data", C.c_void_p),
                ("byte_offset", C.c_uint64), ("shape", C.POINTER(C.c_int64)), ("strides", C.POINTER(C.c_int64)),
                ("rank", C.c_uint32), ("dtype", C.c_uint32), ("device_type", C.c_uint32),
                ("device_index", C.c_int32), ("flags", C.c_uint32), ("reserved0", C.c_uint32),
                ("reserved", C.c_uint64 * 4)]


def profile(dtype, tiny=False):
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype outside declared profiles")
    v, k, j = (4, 2, 3) if tiny else (5023, 400, 5)
    unit = 4 if dtype == "float32" else 8
    # Template, basis, coefficients, posed vertices, transforms, skin weights,
    # skinned output, scratch and bounded verification temporaries. No gradients.
    allowance = 4096 + 2 * 1024**2
    total = (v * 3 + k * v * 3 + k + v * 3 + j * 16 + v * j + v * 3) * unit + allowance
    # Public validation caps each isfinite input at65536 elements. Conservatively
    # reserve eight dtype-sized/mask intermediates, plus small rig storage. This is
    # deliberately prospective, not a measured allocator peak or a native cost.
    basis = k * v * 3 * unit
    validation_allowance = 8 * (unit + 1) * min(65536, k * v * 3)
    public_allowance = validation_allowance + (12 * v * 3 + 20 * j * 16 + 4 * k) * unit
    fitting = total + public_allowance
    all_inputs = total + basis + (v * 3 + k + j * 16 + v * j) * unit
    cap = 64 * 1024**2
    admission = {
        "all_inputs": dict(minimum_live_bytes=all_inputs, basis_gradient_bytes=basis,
                           cuda_admitted=False if all_inputs > cap else None,
                           status="refused by lower bound" if all_inputs > cap else "not yet admitted; complete peak bound required"),
        "fitting_coefficients_pose": dict(minimum_live_bytes=fitting,
                           basis_gradient_bytes=0, cuda_admitted=fitting <= cap,
                           status="prospective bound; not measured",
                           validation_mask_allowance_bytes=validation_allowance,
                           validation_chunk_elements=65536,
                           omitted_gradients=["template", "corrective basis", "skin weights"]),
    }
    return dict(batch=1, vertices=v, deltas=k, joints=j, dtype=dtype,
                input_kind="synthetic constant tensors; no consumer model assets",
                tensor_preflight_bytes=total, auxiliary_allowance_bytes=allowance, gpu_cap_bytes=cap,
                gradient_admission=admission, warmups=1, repetitions=5, tiny=tiny)


def tensor_view(tensor, output=False):
    view = View()
    view.struct_size = C.sizeof(View)
    view.abi_version = ABI
    view.data = tensor.data_ptr()
    view.rank = tensor.ndim
    view._shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
    view._strides = (C.c_int64 * tensor.ndim)(*tensor.stride())
    view.shape, view.strides = view._shape, view._strides
    view.dtype = 1 if tensor.element_size() == 4 else 2
    view.device_type = 2 if tensor.is_cuda else 1
    view.device_index = tensor.device.index or 0
    view.flags = 2 if output else 1
    return view


def bind_native():
    import gffx
    root = Path(gffx.__file__).resolve().parent
    names = ("gffx_core.dll",) if os.name == "nt" else ("libgffx_core.so", "libgffx_core.dylib")
    candidates = [root / name for name in names if (root / name).is_file()]
    if len(candidates) != 1:
        raise ValueError("expected one installed native core beside gffx")
    library = C.CDLL(str(candidates[0]))
    for name in ("blend_shapes", "skin_lbs"):
        query = getattr(library, "gffx_mesh_" + name + "_workspace")
        query.argtypes = [C.c_int64, C.c_int64, C.c_int64, C.c_uint32, C.POINTER(Context),
                          C.POINTER(C.c_uint64), C.POINTER(C.c_uint64), C.c_void_p]
        query.restype = C.c_uint32
        function = getattr(library, "gffx_mesh_" + name)
        function.argtypes = [C.POINTER(View)] * 3 + [C.POINTER(Context), C.POINTER(View), C.POINTER(Buffer), C.c_void_p]
        function.restype = C.c_uint32
    return library, candidates[0]


def measure(args, row):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    p = profile(args.dtype, args.tiny)
    row["profile"] = p
    if args.device == "cuda" and p["tensor_preflight_bytes"] > p["gpu_cap_bytes"]:
        raise MemoryError("declared tensors exceed CUDA allocation cap")
    dtype = getattr(torch, args.dtype)
    v, k, j = p["vertices"], p["deltas"], p["joints"]
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA qualification requires a real adapter")
        torch.cuda.reset_peak_memory_stats()
    # Values are deliberately simple and shape-complete; data preparation is not timed.
    template = torch.zeros((v, 3), dtype=dtype, device=args.device)
    deltas = torch.full((k, v, 3), .001, dtype=dtype, device=args.device)
    weights = torch.ones((1, k), dtype=dtype, device=args.device)
    posed = torch.empty((1, v, 3), dtype=dtype, device=args.device)
    transforms = torch.eye(4, dtype=dtype, device=args.device).reshape(1, 1, 4, 4).repeat(1, j, 1, 1)
    skin_weights = torch.zeros((v, j), dtype=dtype, device=args.device)
    skin_weights[:, 0] = 1
    skinned = torch.empty_like(posed)
    context = Context()
    context.struct_size, context.abi_version = C.sizeof(Context), ABI
    context.device_type = 2 if args.device == "cuda" else 1
    context.flags = 1 if args.mode == "relaxed" else 0
    if args.device == "cuda":
        context.device_index = template.device.index
        context.stream = torch.cuda.current_stream().cuda_stream
    library, path = bind_native()
    row["native_core"] = dict(path=str(path), sha256=artifact.digest(path))
    row["context_flags"] = context.flags
    row["timings"] = {}
    for name, inputs, output, last in (("blend_shapes", (template, deltas, weights), posed, k),
                                       ("skin_lbs", (posed, transforms, skin_weights), skinned, j)):
        size, alignment = C.c_uint64(), C.c_uint64()
        status = getattr(library, "gffx_mesh_" + name + "_workspace")(1, v, last, 1 if dtype == torch.float32 else 2,
                   C.byref(context), C.byref(size), C.byref(alignment), None)
        if status:
            raise RuntimeError(f"{name} workspace status {status}")
        if size.value > 4096:
            raise MemoryError("scratch exceeds preflight allowance")
        scratch = torch.empty(size.value, dtype=torch.uint8, device=args.device)
        buffer = Buffer()
        buffer.struct_size, buffer.abi_version = C.sizeof(Buffer), ABI
        buffer.data, buffer.capacity_bytes, buffer.alignment = scratch.data_ptr(), size.value, max(8, alignment.value)
        buffer.device_type, buffer.device_index = context.device_type, context.device_index
        views = [tensor_view(value) for value in inputs]
        out = tensor_view(output, True)
        function = getattr(library, "gffx_mesh_" + name)
        samples = []
        for repetition in range(p["warmups"] + p["repetitions"]):
            supervisor.check_deadline(row["deadline"])
            if args.device == "cuda":
                torch.cuda.current_stream().synchronize()
            start = time.perf_counter_ns()
            status = function(*(C.byref(view) for view in views), C.byref(context), C.byref(out), C.byref(buffer), None)
            if args.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elapsed_us = (time.perf_counter_ns() - start) / 1000
            if status:
                raise RuntimeError(f"{name} status {status}")
            if repetition >= p["warmups"]:
                samples.append(elapsed_us)
        row["timings"][name] = dict(microseconds=samples, median_microseconds=statistics.median(samples),
                                     cumulative_microseconds=sum(samples),
                                     scratch_bytes=size.value, includes="ctypes invocation + native validation + operation + completed result; not the full PyTorch adapter")
    row["cumulative_measured_microseconds"] = sum(value["cumulative_microseconds"] for value in row["timings"].values())
    row["gradient_costs"] = "not measured; this profile executes native forward operations only"
    torch.testing.assert_close(skinned, posed, atol=0, rtol=0)
    torch.testing.assert_close(posed, torch.full_like(posed, k * .001), atol=1e-5 if dtype == torch.float32 else 1e-12, rtol=0)
    if args.device == "cuda":
        row["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
        if row["gpu_peak_bytes"] > p["gpu_cap_bytes"]:
            raise MemoryError("measured GPU allocation cap exceeded")


def measure_public_fitting(args, row):
    """Public default-reference chain, learning coefficients and pose only."""
    import gffx
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode != "reference":
        raise ValueError("public fitting uses the adapter's default reference mode; no public relaxed-mode control")
    p = profile(args.dtype, args.tiny)
    row["profile"] = p
    admission = p["gradient_admission"]["fitting_coefficients_pose"]
    if args.device == "cuda" and not admission["cuda_admitted"]:
        row["status"] = "not admitted"
        row["reason"] = "prospective public validation/intermediate bound exceeds the unchanged 64 MiB cap"
        return
    dtype = getattr(torch, args.dtype)
    v, k, j = p["vertices"], p["deltas"], p["joints"]
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA qualification requires a real adapter")
        torch.cuda.reset_peak_memory_stats()
    template = torch.full((v, 3), .1, dtype=dtype, device=args.device)
    template[:, 1], template[:, 2] = .2, .3
    deltas = torch.full((k, v, 3), .001, dtype=dtype, device=args.device)
    coefficients = torch.ones((1, k), dtype=dtype, device=args.device, requires_grad=True)
    pose = torch.full((1, j, 3), .02, dtype=dtype, device=args.device, requires_grad=True)
    skin_weights = torch.zeros((v, j), dtype=dtype, device=args.device)
    for joint in range(j):
        skin_weights[joint::j, joint] = 1
    parents = torch.arange(-1, j - 1, dtype=torch.int32, device=args.device)
    faces = torch.tensor([[0, 1, 2], [v - 3, v - 2, v - 1]], dtype=torch.int32, device=args.device)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=args.device)
    bary = template.new_tensor([[.25, .25, .5], [.5, .25, .25]])
    cotangent = template.new_tensor([[[1., 2., 3.], [3., 1., 2.]]])
    translation = template.new_zeros((1, j, 3, 1))
    last_row = template.new_tensor([0., 0., 0., 1.]).reshape(1, 1, 1, 4).repeat(1, j, 1, 1)
    adapter = gffx.torch
    samples = []
    for repetition in range(p["warmups"] + p["repetitions"]):
        supervisor.check_deadline(row["deadline"])
        if args.device == "cuda":
            torch.cuda.current_stream().synchronize()
        start = time.perf_counter_ns()
        posed = adapter.mesh.blend_shapes(template, deltas, coefficients)
        rotation = adapter.transforms.rodrigues(pose)
        local = torch.cat((torch.cat((rotation, translation), -1), last_row), -2)
        world = adapter.skeleton.forward_kinematics(local, parents)
        skin = adapter.mesh.skin_lbs(posed, world, skin_weights)
        output = adapter.mesh.evaluate_barycentric(skin, faces, indices, bary)
        gradients = torch.autograd.grad(output, (coefficients, pose), cotangent)
        if args.device == "cuda":
            torch.cuda.current_stream().synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1000
        if repetition >= p["warmups"]:
            samples.append(elapsed)
        assert bool(torch.isfinite(output).all())
        for gradient in gradients:
            assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().sum() > 0)
        assert template.grad is None and deltas.grad is None and skin_weights.grad is None
        del posed, rotation, local, world, skin, output, gradients
    row["timings"] = {"public_fitting_chain": dict(microseconds=samples,
        median_microseconds=statistics.median(samples), cumulative_microseconds=sum(samples),
        includes="public PyTorch validation, allocations, all five animation operations, coefficient/pose backward and completed result")}
    row["cumulative_measured_microseconds"] = sum(samples)
    row["gradient_costs"] = "coefficients and pose only; template, corrective basis and skin weights fixed"
    row["gradient_validation"] = "finite nonzero requested gradients; fixed inputs receive none"
    if args.device == "cuda":
        row["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
        if row["gpu_peak_bytes"] > p["gpu_cap_bytes"]:
            raise MemoryError("measured GPU allocation cap exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    for name in ("fixtures", "output", "wheel"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--mode", choices=("reference", "relaxed"), required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--public-fitting", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        if args.device == "cuda":
            supervisor.PROFILE["sampled_working_set_limit_bytes"] = 2 * 1024**3
        extra = ["--wheel", str(args.wheel), "--sha256", args.sha256, "--device", args.device,
                 "--dtype", args.dtype, "--mode", args.mode]
        if args.tiny:
            extra.append("--tiny")
        if args.public_fitting:
            extra.append("--public-fitting")
        return supervisor.supervise(args.fixtures, args.output, worker_module="tools.animation_cost", worker_args=extra)
    row = dict(status="working", worker_pid=os.getpid(), device=args.device, mode=args.mode,
               deadline=time.perf_counter() + 120)
    supervisor.write_report(args.output, row)
    try:
        row["wheel"] = artifact.identity(args.wheel, args.sha256)
        row["driver_sha256"] = artifact.digest(__file__)
        (measure_public_fitting if args.public_fitting else measure)(args, row)
        supervisor.check_deadline(row["deadline"])
        if row["status"] != "not admitted":
            row["status"] = "passed"
    except Exception as error:
        row.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        row.pop("deadline", None)
        supervisor.write_report(args.output, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
