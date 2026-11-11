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
    return {"os": platform, "architecture": "x86_64", "python": "3.10.14", "tools": results}


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
