"""Candidate release artifact assembly must reject missing, stale or altered wheels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from tools.ci_matrix import PLATFORMS
from tools.assemble_candidate_bundle import assemble, AssemblyError


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inputs(tmp_path: Path):
    source = tmp_path / "pyproject.toml"
    source.write_text('[project]\nname = "gffx"\nversion = "0.2.2"\n', encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for platform, row in PLATFORMS.items():
        folder = artifacts / ("wheel-" + row["artifact"])
        folder.mkdir()
        files = {f"gffx-0.2.2-{row['wheel_tag']}.whl": b"wheel-" + platform.encode()}
        if platform == "linux-x64":
            files["gffx-0.2.2.tar.gz"] = b"source"
        for filename, payload in files.items():
            (folder / filename).write_bytes(payload)
        (folder / "provenance.json").write_text(json.dumps({
            "mode": "artifacts",
            "artifacts": [
                {"name": name, "size_bytes": len(payload), "sha256": _sha(payload)}
                for name, payload in files.items()
            ],
        }), encoding="utf-8")
    return artifacts, source


def test_assembles_four_wheels_and_one_source(tmp_path):
    artifacts, pyproject = _inputs(tmp_path)
    out = tmp_path / "out"
    result = assemble(artifacts, out, pyproject)
    assert result["version"] == "0.2.2"
    assert len(result["artifacts"]) == 5
    assert len(list(out.iterdir())) == 5


def test_missing_platform_blocks_without_output(tmp_path):
    artifacts, pyproject = _inputs(tmp_path)
    folder = artifacts / "wheel-macosx_14_0_arm64"
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
    out = tmp_path / "out"
    with pytest.raises(AssemblyError, match="folders differ"):
        assemble(artifacts, out, pyproject)
    assert not out.exists()


def test_unrecorded_binary_replacement_blocks(tmp_path):
    artifacts, pyproject = _inputs(tmp_path)
    candidate = next((artifacts / "wheel-win_amd64").glob("*.whl"))
    candidate.write_bytes(b"tampered")
    with pytest.raises(AssemblyError, match="artifact bytes differ"):
        assemble(artifacts, tmp_path / "out", pyproject)


def test_extra_unknown_artifact_blocks(tmp_path):
    artifacts, pyproject = _inputs(tmp_path)
    (artifacts / "wheel-win_amd64" / "unexpected.whl").write_bytes(b"extra")
    with pytest.raises(AssemblyError, match="files differ"):
        assemble(artifacts, tmp_path / "out", pyproject)


def test_incorrect_version_blocks(tmp_path):
    artifacts, pyproject = _inputs(tmp_path)
    pyproject.write_text('[project]\nname = "gffx"\nversion = "0.2.3"\n', encoding="utf-8")
    with pytest.raises(AssemblyError, match="files differ"):
        assemble(artifacts, tmp_path / "out", pyproject)


def test_existing_output_cannot_be_reused(tmp_path):
    artifacts, pyproject = _inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(AssemblyError, match="new path"):
        assemble(artifacts, out, pyproject)
