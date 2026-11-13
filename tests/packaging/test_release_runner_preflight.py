"""Runner preflight refuses wrong hardware without accessing runner credentials."""
from __future__ import annotations

import pytest

from tools.release_runner_preflight import TOOLS, validate


def valid_report(platform="linux"):
    results = {
        name: {"available": True, "exit_code": 0, "output": "tool version"}
        for name in TOOLS
    }
    results["nvcc"]["output"] = "Cuda compilation tools, release 12.9, V12.9.86"
    results["nvidia_smi"]["output"] = "GPU 0: NVIDIA GeForce RTX (UUID: GPU-placeholder)"
    return {"os": platform, "architecture": "x86_64", "python": "3.10.14",
            "kernel_release": "6.8.0-generic" if platform == "linux" else "10.0.26100",
            "tools": results}


@pytest.mark.parametrize("system", ["windows", "linux"])
def test_real_nvidia_runner_preflight_accepts_complete_synthetic_host(system):
    assert validate(valid_report(system), system) == []


@pytest.mark.parametrize("problem", [
    "wrong_os", "wrong_arch", "old_python", "wrong_cuda", "missing_gpu", "missing_nvcc",
    "missing_sanitizer", "missing_cmake", "missing_git",
])
def test_runner_preflight_refuses_missing_qualification_prerequisites(problem):
    report = valid_report("linux")
    if problem == "wrong_os":
        report["os"] = "darwin"
    elif problem == "wrong_arch":
        report["architecture"] = "aarch64"
    elif problem == "old_python":
        report["python"] = "3.9.21"
    elif problem == "wrong_cuda":
        report["tools"]["nvcc"]["output"] = "Cuda compilation tools, release 13.0"
    elif problem == "missing_gpu":
        report["tools"]["nvidia_smi"]["output"] = "No devices were found"
    else:
        name = {"missing_sanitizer": "compute_sanitizer"}.get(problem, problem.removeprefix("missing_"))
        report["tools"][name]["available"] = False
    assert validate(report, "linux")


@pytest.mark.parametrize("target", ["macos", "other", ""])
def test_runner_preflight_cannot_claim_unsupported_cuda_os(target):
    with pytest.raises(ValueError):
        validate(valid_report(), target)


def test_native_linux_rc_gate_accepts_real_linux_kernel():
    assert validate(valid_report("linux"), "linux", require_native_linux=True) == []


@pytest.mark.parametrize("kernel", [
    "6.18.58.1-microsoft-standard-WSL2", "5.15.153.1-microsoft-standard-WSL2",
    "", None,
])
def test_native_linux_rc_gate_rejects_wsl_and_unreported_kernels(kernel):
    report = valid_report("linux")
    report["kernel_release"] = kernel
    errors = validate(report, "linux", require_native_linux=True)
    assert any("native Linux kernel" in message for message in errors)
    assert validate(report, "linux") == []  # Local development WSL remains supported.


def test_native_linux_gate_never_claims_windows_admission():
    with pytest.raises(ValueError, match="Linux target"):
        validate(valid_report("windows"), "windows", require_native_linux=True)
