"""Fresh-process context diagnostic, not an empty-operation acceptance waiver.

Run alone: another CUDA test can activate the primary context and erase the case.
An inactive-primary rejection records an unresolved adapter boundary, not success
of the public contract that empty operations are valid.
"""
import ctypes
import os

import pytest
import torch
import gffx.torch


def test_animation_empty_reference_does_not_silently_activate_a_context(record_property):
    if not torch.cuda.is_available():
        pytest.skip("requires a real CUDA adapter")
    driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    state = driver.cuDevicePrimaryCtxGetState
    state.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
    state.restype = ctypes.c_int
    device = 0
    flags, before, after, before_tensor = ctypes.c_uint(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    assert state(device, ctypes.byref(flags), ctypes.byref(before_tensor)) == 0
    record_property("primary_active_before_tensor_creation", before_tensor.value)
    record_property("torch_cuda_initialized_before_tensor_creation", torch.cuda.is_initialized())
    # The supported PyTorch factory may activate its own context even for zero
    # elements. That is caller ownership, not a gffx warmup or hidden activation.
    empty = torch.empty((0, 3), dtype=torch.float64, device="cuda:0")
    assert state(device, ctypes.byref(flags), ctypes.byref(before)) == 0
    record_property("primary_active_before", before.value)
    # Direct dispatch intentionally avoids Python isfinite(...).all(), whose tiny
    # reduction allocation could otherwise activate the context before this test.
    if before.value:
        result = torch.ops.gffx.rodrigues(empty)
        assert result.shape == (0, 3, 3) and result.numel() == 0
        record_property("empty_context_result", "accepted with existing primary owner")
    else:
        with pytest.raises(RuntimeError, match="context|primary|owner"):
            torch.ops.gffx.rodrigues(empty)
        record_property("empty_context_result", "unsupported gap: no active primary owner")
    assert state(device, ctypes.byref(flags), ctypes.byref(after)) == 0
    assert after.value == before.value
    record_property("primary_active_after", after.value)
