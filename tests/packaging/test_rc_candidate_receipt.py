"""An RC receipt must bind immutable source, workflow, and exact distribution bytes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.rc_candidate_receipt import ReceiptError, receipt_for, check_receipt


COMMIT = "a" * 40
RUN = 123456789


def _bundle(tmp_path: Path):
    # Reuse the independent cross-platform wheel/sdist fixtures. This module
    # is loaded as a source file to avoid assumptions about pytest package paths.
    source = Path(__file__).with_name("test_release_bundle_gate.py")
    spec = importlib.util.spec_from_file_location("gffx_bundle_fixture", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._bundle(tmp_path)


def test_receipt_binds_exact_complete_bundle_and_run(tmp_path):
    dist, project = _bundle(tmp_path)
    expected = receipt_for(dist, project, COMMIT, RUN)
    assert expected["version"] == "0.2.2"
    assert expected["rc_run_id"] == RUN
    assert len(expected["artifacts"]) == 5
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(expected), encoding="utf-8")
    assert check_receipt(dist, project, COMMIT, RUN, receipt) == expected


@pytest.mark.parametrize("revision,run", [
    ("dev", RUN), ("A" * 40, RUN), ("0" * 39, RUN), (COMMIT, 0),
    (COMMIT, -1), (COMMIT, "01"), (COMMIT, "not-an-id"),
])
def test_moving_revision_or_invalid_run_refused(tmp_path, revision, run):
    dist, project = _bundle(tmp_path)
    with pytest.raises(ReceiptError):
        receipt_for(dist, project, revision, run)


def test_other_source_commit_is_rejected(tmp_path):
    dist, project = _bundle(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(receipt_for(dist, project, COMMIT, RUN)), encoding="utf-8")
    with pytest.raises(ReceiptError, match="differs"):
        check_receipt(dist, project, "b" * 40, RUN, receipt)


def test_other_rc_run_id_is_rejected(tmp_path):
    dist, project = _bundle(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(receipt_for(dist, project, COMMIT, RUN)), encoding="utf-8")
    with pytest.raises(ReceiptError, match="differs"):
        check_receipt(dist, project, COMMIT, RUN + 1, receipt)


def test_modified_distribution_bytes_cannot_reuse_receipt(tmp_path):
    dist, project = _bundle(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(receipt_for(dist, project, COMMIT, RUN)), encoding="utf-8")
    binary = next(dist.glob("*win_amd64.whl"))
    binary.write_bytes(binary.read_bytes() + b"\nbyte-modified")
    with pytest.raises(ReceiptError, match="differs"):
        check_receipt(dist, project, COMMIT, RUN, receipt)


def test_receipt_cannot_hide_an_extra_artifact(tmp_path):
    dist, project = _bundle(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(receipt_for(dist, project, COMMIT, RUN)), encoding="utf-8")
    (dist / "unexpected.whl").write_bytes(b"new")
    with pytest.raises(ValueError, match="unexpected"):
        check_receipt(dist, project, COMMIT, RUN, receipt)


def test_extra_or_deleted_receipt_field_is_refused(tmp_path):
    dist, project = _bundle(tmp_path)
    receipt = tmp_path / "receipt.json"
    data = receipt_for(dist, project, COMMIT, RUN)
    data["unexpected_field"] = True
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReceiptError, match="differs"):
        check_receipt(dist, project, COMMIT, RUN, receipt)
