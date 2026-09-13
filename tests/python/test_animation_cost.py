"""Cost harness bounds and ABI layout; no benchmark or GPU claim."""
import ctypes
import pytest
from tools.animation_cost import Buffer, Context, View, profile


def test_animation_cost_profile_admits_only_declared_forward_storage():
    single, double = profile("float32"), profile("float64")
    assert (double["vertices"], double["deltas"], double["joints"], double["batch"]) == (5023, 400, 5, 1)
    allowance = double["auxiliary_allowance_bytes"]
    assert allowance == 4096 + 2 * 1024**2
    assert double["tensor_preflight_bytes"] == 2 * (single["tensor_preflight_bytes"] - allowance) + allowance
    assert double["tensor_preflight_bytes"] < double["gpu_cap_bytes"]
    assert double["repetitions"] == 5 and double["warmups"] == 1
    assert double["gradient_admission"]["all_inputs"]["minimum_live_bytes"] > double["gpu_cap_bytes"]
    assert double["gradient_admission"]["all_inputs"]["cuda_admitted"] is False
    assert double["gradient_admission"]["all_inputs"]["basis_gradient_bytes"] == 48_220_800
    assert single["gradient_admission"]["fitting_coefficients_pose"]["cuda_admitted"] is True
    assert double["gradient_admission"]["fitting_coefficients_pose"]["cuda_admitted"] is True
    assert double["gradient_admission"]["fitting_coefficients_pose"]["validation_chunk_elements"] == 65536
    assert double["gradient_admission"]["fitting_coefficients_pose"]["status"] == "prospective bound; not measured"
    assert profile("float64", tiny=True)["gradient_admission"]["fitting_coefficients_pose"]["cuda_admitted"] is True
    assert profile("float64", tiny=True)["vertices"] == 4
    with pytest.raises(ValueError, match="declared"):
        profile("float16")


def test_animation_cost_ctypes_layout_matches_public_64_bit_abi():
    assert ctypes.sizeof(Context) == 64
    assert ctypes.sizeof(Buffer) == 64
    assert ctypes.sizeof(View) == 96
    assert View.shape.offset == 24 and View.strides.offset == 32
    assert Context.stream.offset == 16
