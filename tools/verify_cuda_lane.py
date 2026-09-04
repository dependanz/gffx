#!/usr/bin/env python3
"""Verify that a CUDA hardware run actually exercised the device, and record what it ran on.

This module intentionally uses only the Python standard library.

The failure this exists to prevent is not a red test.  It is a green one.  Every device-gated
fixture in GFFX declines to run when no usable device is present, and a lane that quietly loses
its GPU - a driver update, a runner relabelled onto the wrong machine, a build configured without
``GFFX_ENABLE_CUDA`` - would otherwise report a complete pass over the tests that remained.  That
is the same shape as the adapter fixtures that were skipping unnoticed because the installed
PyTorch had no CUDA build: nothing failed, so nothing was investigated.  So the lane asserts the
device tests are present by name and that the inventory has not shrunk, rather than trusting the
process exit status.

The recorded device provenance is not decoration either.  Bitwise CPU/CUDA agreement is a claim
about one driver on one architecture; ``log2f`` is not required by IEEE 754 to be correctly
rounded, so the claim is measured rather than guaranteed and has to name what measured it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET


# Registered only when the build both enables CUDA and is permitted to touch a real device.
# These carry every bit-identity claim GFFX makes; without them the lane proves nothing a
# CPU-only runner does not already prove.
DEVICE_GATED_TESTS = (
    "cuda.texture_device_parity",
    "cuda.backward_device_parity",
    "cuda.plugin.real_device_probe",
    "cuda.plugin.default_discovery",
)

# Registered whenever CUDA is enabled, because they need the real plugin binary rather than the
# synthetic one.  A build that silently fell back to the CPU configuration loses these too.
PLUGIN_BUILD_TESTS = (
    "cuda.binary_plugin_isolation",
    "cuda.operation_dispatch",
)

# Measured on the reference host 2026-09-03.  A floor, not an equality: adding fixtures must not
# fail the lane, but removing them must.
MINIMUM_TOTAL_TESTS = 40


class LaneError(RuntimeError):
    """The lane did not establish what it claims to establish."""


def _count(suite: ET.Element, name: str) -> int:
    value = suite.get(name)
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError as error:
        raise LaneError(f"CTest JUnit result has a non-integer {name!r} count") from error


def parse_junit(path: Path) -> dict[str, Any]:
    try:
        suite = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise LaneError("CTest JUnit result is missing or malformed") from error
    if suite.tag != "testsuite":
        raise LaneError("CTest JUnit result does not contain one testsuite root")

    total = _count(suite, "tests")
    if total == 0:
        raise LaneError("CTest JUnit result reports zero executed tests")

    incomplete = {
        name: _count(suite, name) for name in ("failures", "errors", "disabled", "skipped")
    }
    if any(incomplete.values()):
        raise LaneError("CTest JUnit result is not a complete pass: " + repr(incomplete))

    cases = suite.findall("testcase")
    if len(cases) != total:
        raise LaneError("CTest JUnit test count does not match its testcase records")
    if any(suite.findall(f".//{name}") for name in ("failure", "error", "skipped")):
        raise LaneError("CTest JUnit testcase records are not a complete pass")

    return {"total": total, "names": sorted(case.get("name", "") for case in cases)}


def check_inventory(result: dict[str, Any]) -> None:
    names = set(result["names"])
    missing = [name for name in DEVICE_GATED_TESTS + PLUGIN_BUILD_TESTS if name not in names]
    if missing:
        raise LaneError(
            "the run did not include the device-gated fixtures, so it did not touch a GPU: "
            + ", ".join(missing)
        )
    if result["total"] < MINIMUM_TOTAL_TESTS:
        raise LaneError(
            f"the run executed {result['total']} tests, below the recorded floor of "
            f"{MINIMUM_TOTAL_TESTS}; the inventory shrank rather than the tests passing"
        )


def _run_text(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
            [executable, *command[1:]],
            capture_output=True,
            check=True,
            timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return completed.stdout.decode("utf-8", errors="replace").strip()


def collect_provenance() -> dict[str, Any]:
    """Name the hardware and toolchain the measurement was taken on.

    Absence is recorded as null rather than omitted, so a report that lost its provenance is
    distinguishable from one that never had any.
    """
    devices = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ]
    )
    nvcc = _run_text(["nvcc", "--version"])
    release = None
    if nvcc:
        for line in nvcc.splitlines():
            if "release" in line:
                release = line.strip()
                break
    return {
        "devices": devices.splitlines() if devices else None,
        "nvcc_release": release,
        "platform": sys.platform,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("junit", type=Path, help="CTest --output-junit result to verify")
    parser.add_argument("--report", type=Path, help="write the JSON lane report here")
    arguments = parser.parse_args(argv)

    provenance = collect_provenance()
    try:
        result = parse_junit(arguments.junit)
        check_inventory(result)
    except LaneError as error:
        report = {"status": "failed", "reason": str(error), "provenance": provenance}
        status = 1
    else:
        report = {
            "status": "passed",
            "tests_executed": result["total"],
            "device_gated_tests": list(DEVICE_GATED_TESTS),
            "provenance": provenance,
        }
        status = 0

    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.report:
        arguments.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
