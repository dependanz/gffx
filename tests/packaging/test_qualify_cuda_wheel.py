"""Fail-closed CUDA wheel qualification policy, testable without any GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.qualify_cuda_wheel import (
    MINIMUM_PASSED, QualificationError, REQUIRED_ORTHOGRAPHIC,
    check_complete, source_version, verify_wheel_provenance,
)


def _passing() -> list[dict[str, str]]:
    names = [*REQUIRED_ORTHOGRAPHIC, *(
        f"test_other_gpu_case_{i}" for i in range(MINIMUM_PASSED - len(REQUIRED_ORTHOGRAPHIC))
    )]
    return [{"nodeid": f"tests/python/test_file.py::{name}", "outcome": "passed"}
            for name in names]


def test_complete_exact_cuda_cohort_is_accepted():
    actual = check_complete(_passing(), exit_code=0)
    assert actual["passed"] == MINIMUM_PASSED
    assert actual["failed"] == actual["skipped"] == 0
    assert all(actual["orthographic_cuda_coverage"].values())


@pytest.mark.parametrize("defect", [
    "too_few", "missing_camera", "missing_render", "one_skip",
    "one_failure", "nonzero_exit",
])
def test_cpu_only_and_incomplete_cuda_cohorts_fail_closed(defect):
    results = _passing()
    status = 0
    if defect == "too_few":
        results.pop()
    elif defect == "missing_camera":
        results[0]["nodeid"] = "test_other_camera"
    elif defect == "missing_render":
        results[1]["nodeid"] = "test_other_render"
    elif defect == "one_skip":
        results[0]["outcome"] = "skipped"
    elif defect == "one_failure":
        results[2]["outcome"] = "failed"
    else:
        status = 1
    with pytest.raises(QualificationError, match="incomplete installed-wheel"):
        check_complete(results, exit_code=status)


def test_release_version_is_read_from_authoritative_source(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "gffx"\nversion = "0.2.2"\n', encoding="utf-8")
    assert source_version(tmp_path) == "0.2.2"


def _provenance(tmp_path: Path):
    wheel = tmp_path / "gffx-0.2.2-cp310-abi3-manylinux_2_28_x86_64.whl"
    wheel.write_bytes(b"synthetic candidate wheel bytes")
    receipt = tmp_path / "provenance.json"
    record = {
        "name": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "size_bytes": wheel.stat().st_size,
    }
    receipt.write_text(json.dumps({"mode": "artifacts", "artifacts": [record]}),
                       encoding="utf-8")
    return wheel, receipt


def test_exact_wheel_sha_and_length_must_match_original_build_receipt(tmp_path):
    wheel, receipt = _provenance(tmp_path)
    assert verify_wheel_provenance(wheel, receipt) == hashlib.sha256(
        wheel.read_bytes()).hexdigest()
    wheel.write_bytes(wheel.read_bytes() + b"tampering")
    with pytest.raises(QualificationError, match="wheel bytes differ"):
        verify_wheel_provenance(wheel, receipt)


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "different_name", "wrong_mode"])
def test_unverifiable_or_ambiguous_wheel_receipts_are_rejected(tmp_path, tamper):
    wheel, receipt = _provenance(tmp_path)
    if tamper == "missing":
        receipt.unlink()
    else:
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if tamper == "duplicate":
            data["artifacts"].append(data["artifacts"][0].copy())
        elif tamper == "different_name":
            data["artifacts"][0]["name"] = "another.whl"
        else:
            data["mode"] = "installed"
        receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(QualificationError):
        verify_wheel_provenance(wheel, receipt)
