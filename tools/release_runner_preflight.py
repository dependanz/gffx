#!/usr/bin/env python3
"""Preflight a trusted NVIDIA hardware host before enrolling it as a GFFX RC runner.

This checks hardware and toolchain *presence*, not numerical correctness or
real release-wheel qualification. It performs no builds, downloads or installs
and never reads GitHub registration credentials.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys


TOOLS = {
    "nvidia_smi": ("nvidia-smi", "-L"),
    "nvcc": ("nvcc", "--version"),
    "compute_sanitizer": ("compute-sanitizer", "--version"),
    "cmake": ("cmake", "--version"),
    "git": ("git", "--version"),
}


def probe(command: tuple[str, ...]) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "exit_code": None, "output": ""}
    try:
        result = subprocess.run([executable, *command[1:]], capture_output=True,
                                text=True, timeout=25, check=False)
    except (subprocess.SubprocessError, OSError):
        return {"available": False, "exit_code": None, "output": ""}
    return {
        "available": result.returncode == 0,
        "exit_code": result.returncode,
        "output": (result.stdout + "\n" + result.stderr).strip()[:4000],
    }


def inspect() -> dict:
    return {
        "os": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "kernel_release": platform.release(),
        "python": platform.python_version(),
        "tools": {name: probe(command) for name, command in TOOLS.items()},
    }


def validate(report: dict, os_target: str, *, require_native_linux: bool = False) -> list[str]:
    errors = []
    target = {"windows": "windows", "linux": "linux"}.get(os_target)
    if target is None:
        raise ValueError("os_target must be windows or linux")
    if report.get("os") != target:
        errors.append(f"host must run {target}; received {report.get('os')}")
    if require_native_linux:
        if target != "linux":
            raise ValueError("require_native_linux is valid only for a Linux target")
        kernel = str(report.get("kernel_release") or "")
        if not kernel or "microsoft" in kernel.lower() or "wsl" in kernel.lower():
            errors.append("release candidate Linux NVIDIA qualification requires a native Linux kernel, not WSL2")
    if report.get("architecture") not in {"x86_64", "amd64"}:
        errors.append("CUDA release host must use x86-64")
    version = report.get("python", "")
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None or int(match.group(1)) != 3 or not 10 <= int(match.group(2)) <= 14:
        errors.append("runner Python must be an supported CPython 3.10-3.14")
    tools = report.get("tools") or {}
    for name in TOOLS:
        if not tools.get(name, {}).get("available"):
            errors.append(f"{name} must be installed and return a successful probe")
    nvcc = tools.get("nvcc", {}).get("output", "")
    if tools.get("nvcc", {}).get("available") and re.search(r"\brelease\s+12(?:\.\d+)?\b", nvcc) is None:
        errors.append("CUDA 12 nvcc is required for the current release candidate")
    devices = tools.get("nvidia_smi", {}).get("output", "")
    if tools.get("nvidia_smi", {}).get("available") and not re.search(r"\bGPU\s+0\s*:", devices):
        errors.append("NVIDIA GPU 0 must be visible to the local driver")
    # We do not run GPU kernels here: only the exact-wheel RC lane proves those.
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["windows", "linux"], required=True)
    parser.add_argument("--json", action="store_true", help="include host/tool output in result")
    parser.add_argument("--require-native-linux", action="store_true",
                        help="refuse WSL2 and unknown kernel provenance for production Linux RC")
    args = parser.parse_args()
    result = inspect()
    issues = validate(result, args.target, require_native_linux=args.require_native_linux)
    summary = {
        "status": "pass" if not issues else "blocked",
        "target": args.target,
        "issues": issues,
        "note": "Toolchain discovery only; release CUDA kernel qualification is separate.",
    }
    if args.json:
        summary["host"] = result
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
