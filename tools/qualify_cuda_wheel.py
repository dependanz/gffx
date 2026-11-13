#!/usr/bin/env python3
"""Qualify an installed, provenance-pinned GFFX CUDA wheel on a real GPU.

Run from a clean *installed-wheel* virtual environment with CUDA-enabled Torch,
pytest and a driver-visible NVIDIA GPU. The native/driver test lane is separate.
The source tree supplies tests only, never the Python package under test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import sys
from typing import Any


REQUIRED_ORTHOGRAPHIC = (
    "OrthographicTests::test_cpu_cuda_bytes",
    "OrthographicTests::test_render_cpu_cuda_bytes",
)
MINIMUM_PASSED = 285


class QualificationError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_wheel_provenance(wheel: Path, provenance_path: Path) -> str:
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise QualificationError("exact wheel is missing")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise QualificationError("missing or malformed exact-wheel provenance") from error
    declared = provenance.get("artifacts")
    if provenance.get("mode") != "artifacts" or not isinstance(declared, list):
        raise QualificationError("wheel was not accompanied by build artifact provenance")
    matches = [entry for entry in declared if isinstance(entry, dict)
               and entry.get("name") == wheel.name]
    if len(matches) != 1:
        raise QualificationError("exact wheel is missing or duplicated in provenance")
    actual = sha256(wheel)
    if actual != matches[0].get("sha256") or wheel.stat().st_size != matches[0].get("size_bytes"):
        raise QualificationError("wheel bytes differ from original CI provenance")
    return actual


def check_complete(outcomes: list[dict[str, str]], exit_code: int) -> dict[str, Any]:
    passed = [r["nodeid"] for r in outcomes if r["outcome"] == "passed"]
    failed = [r["nodeid"] for r in outcomes if r["outcome"] == "failed"]
    skipped = [r["nodeid"] for r in outcomes if r["outcome"] == "skipped"]
    covered = {
        suffix: any(nodeid.endswith(suffix) for nodeid in passed)
        for suffix in REQUIRED_ORTHOGRAPHIC
    }
    result = {
        "passed": len(passed), "failed": len(failed), "skipped": len(skipped),
        "exit_code": exit_code, "orthographic_cuda_coverage": covered,
        "failed_nodeids": failed, "skipped_nodeids": skipped,
    }
    if (exit_code != 0 or failed or skipped or len(passed) < MINIMUM_PASSED
            or not all(covered.values())):
        raise QualificationError(
            f"incomplete installed-wheel CUDA test coverage: "
            f"{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped, "
            f"exit={exit_code}, orthographic={covered}"
        )
    return result


def source_version(source: Path) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', (source / "pyproject.toml").read_text(
        encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise QualificationError("source project has no static version")
    return match.group(1)


def qualify(wheel: Path, provenance: Path, output_dir: Path, source: Path) -> dict:
    wheel = wheel.resolve()
    wheel_hash = verify_wheel_provenance(wheel, provenance)
    if output_dir.exists():
        raise QualificationError("qualification output directory already exists")
    # The camera's two explicitly GPU-owned tests select their device at import
    # time. Set both worker flags before pytest collection, not afterward.
    os.environ["GFFIT_TEST_DEVICE"] = "cuda"
    os.environ["GFFX_KNOT_DEVICE"] = "cuda"
    os.environ["PYTHONNOUSERSITE"] = "1"
    for key in ("GFFX_CUDA_PLUGIN_PATH", "GFFX_TEST_SOURCE_TREE", "PYTHONPATH"):
        os.environ.pop(key, None)

    import gffx
    import gffx.cuda
    import torch
    import pytest
    from importlib import metadata

    version = source_version(source)
    if metadata.version("gffx") != version or gffx.__version__ != version:
        raise QualificationError("installed wheel version differs from source candidate")
    if not Path(gffx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise QualificationError("GFFX resolved to the source tree, not installed wheel")
    if torch.__version__ != "2.10.0+cu126" or torch.version.cuda != "12.6":
        raise QualificationError("CUDA framework must be the pinned Torch 2.10.0+cu126")
    if not torch.cuda.is_available():
        raise QualificationError("CUDA device unavailable; refusing a CPU-only pass")
    capabilities = gffx.cuda.capabilities()
    if capabilities.get("status") != "loaded":
        raise QualificationError("GFFX installed CUDA provider was not loaded")
    provider_name = "gffx_cuda12.dll" if sys.platform == "win32" else "libgffx_cuda12.so"
    if not (Path(gffx.__file__).parent / provider_name).is_file():
        raise QualificationError("installed wheel does not contain required CUDA provider")

    suite = runpy.run_path(str(source / "tools" / "stable_validation.py"))
    tests = source / "tests"
    sys.path[:0] = [str(tests / name) for name in ("python", "pytorch", "packaging")]
    selected = [str(tests / name) for name in suite["GPU"]]
    selected.extend(str(tests / "python" / "test_image_cuda.py") + "::" + name
                    for name in suite["GPU_IMAGES"])

    class Recorder:
        def __init__(self):
            self.outcomes: list[dict[str, str]] = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.skipped or (
                report.when == "setup" and report.failed
            ):
                self.outcomes.append({
                    "nodeid": report.nodeid, "outcome": report.outcome,
                })

    output_dir.mkdir(parents=True)
    recorder = Recorder()
    exit_code = int(pytest.main([
        "-q", "-rs", "--import-mode=importlib", "-o", "junit_family=xunit1",
        "--basetemp=" + str(output_dir / "temp"),
        "--junitxml=" + str(output_dir / "tests.xml"), *selected,
    ], plugins=[recorder]))
    torch.cuda.synchronize()

    summary = {
        "schema": "gffx-installed-cuda-wheel-v1",
        "version": version, "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "wheel": wheel.name, "wheel_sha256": wheel_hash,
        "selected_suites": suite["GPU"],
        "selected_image_nodes": suite["GPU_IMAGES"],
        "worker_device": os.environ["GFFIT_TEST_DEVICE"],
    }
    outcome = check_complete
    try:
        summary.update(outcome(recorder.outcomes, exit_code))
    except QualificationError:
        summary.update({
            "passed": sum(r["outcome"] == "passed" for r in recorder.outcomes),
            "failed": sum(r["outcome"] == "failed" for r in recorder.outcomes),
            "skipped": sum(r["outcome"] == "skipped" for r in recorder.outcomes),
            "exit_code": exit_code,
        })
        (output_dir / "result.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    (output_dir / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = qualify(args.wheel, args.provenance, args.output_dir, args.source)
    except (QualificationError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"CUDA WHEEL QUALIFICATION BLOCKED: {error}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    print("INSTALLED CUDA WHEEL QUALIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
