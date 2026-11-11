"""A publishing handoff must reject stale runs, wrong tags and absent GPUs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.rc_candidate_receipt import receipt_for
from tools.verify_multiplatform_publication import (
    PublicationError, preflight, stable_version, verify_run_data, verify_device_report,
)


SHA = "b" * 40
RUN = "1234567"


def _bundle(tmp_path):
    source = Path(__file__).with_name("test_release_bundle_gate.py")
    spec = importlib.util.spec_from_file_location("gffx_publisher_fixtures", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._bundle(tmp_path)


def _run():
    return {
        "id": int(RUN), "name": "package-foundation-rc",
        "event": "workflow_dispatch", "status": "completed", "conclusion": "success",
        "repository": {"full_name": "dependanz/gffx"},
    }


def _device(path, platform):
    path.write_text(json.dumps({
        "status": "passed",
        "tests_executed": 57,
        "provenance": {
            "platform": platform,
            "devices": ["NVIDIA device, driver, capability"],
            "nvcc_release": "Cuda compilation tools, release 12.9",
        },
    }), encoding="utf-8")


def _args(tmp_path):
    dist, project = _bundle(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(receipt_for(dist, project, SHA, RUN)), encoding="utf-8")
    windows = tmp_path / "windows.json"
    linux = tmp_path / "linux.json"
    _device(windows, "win32")
    _device(linux, "linux")
    return dict(
        tag="v0.2.2", run_id=RUN, git_head=SHA, project=project,
        dist=dist, receipt=receipt, windows_report=windows,
        linux_report=linux, run_data=_run(),
    )


def test_exact_complete_source_run_and_gpu_reports_are_accepted(tmp_path):
    args = _args(tmp_path)
    result = preflight(**args)
    assert result["version"] == "0.2.2"
    assert result["source_commit"] == SHA
    assert result["wheel_and_sdist_count"] == 5


@pytest.mark.parametrize("tag", ["main", "v0.2.1", "v0.2.2rc1", "v0.2.2.dev1", "v1.2"])
def test_publisher_rejects_legacy_non_stable_or_moving_tags(tag):
    with pytest.raises(PublicationError):
        stable_version(tag)


@pytest.mark.parametrize("field,invalid", [
    ("id", 9), ("name", "windows-functional"), ("event", "pull_request"),
    ("status", "in_progress"), ("conclusion", "failure"),
    ("repository", {"full_name": "untrusted/fork"}),
])
def test_publisher_rejects_other_or_failing_rc_runs(field, invalid):
    result = _run()
    result[field] = invalid
    with pytest.raises(PublicationError):
        verify_run_data(result, RUN)


@pytest.mark.parametrize("scenario", ["no_gpu", "wrong_os", "no_nvcc", "skipped", "too_few"])
def test_publisher_rejects_inadequate_hardware_evidence(tmp_path, scenario):
    path = tmp_path / "gpu.json"
    _device(path, "linux")
    data = json.loads(path.read_text())
    if scenario == "no_gpu":
        data["provenance"]["devices"] = None
    elif scenario == "wrong_os":
        data["provenance"]["platform"] = "win32"
    elif scenario == "no_nvcc":
        data["provenance"]["nvcc_release"] = None
    elif scenario == "skipped":
        data["status"] = "failed"
    else:
        data["tests_executed"] = 12
    path.write_text(json.dumps(data))
    with pytest.raises(PublicationError):
        verify_device_report(path, "linux")


def test_publisher_detects_tag_and_checked_out_version_mismatch(tmp_path):
    args = _args(tmp_path)
    args["tag"] = "v0.2.3"
    with pytest.raises(PublicationError, match="tag does not match"):
        preflight(**args)


def test_publisher_detects_substituted_commit_and_run(tmp_path):
    args = _args(tmp_path)
    args["git_head"] = "c" * 40
    with pytest.raises(ValueError, match="differs"):
        preflight(**args)
    args = _args(tmp_path / "second")
    args["run_id"] = "999999"
    with pytest.raises(PublicationError):
        preflight(**args)
