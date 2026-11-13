"""Native-only Windows CUDA resource generation must fail closed.

The output verifier is exercised with deliberately missing files on every host.
Windows-specific CMake source contracts ensure Python discovery is independent
of optional CPython and PyTorch adapters.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "cmake" / "GffxEmbedPtxResource.py"
VERIFIER = ROOT / "cmake" / "GffxVerifyPtxResource.cmake"


def _validate(header: Path, payload: Path, rc: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "cmake",
            f"-DGFFX_PTX_RESOURCE_HEADER={header}",
            f"-DGFFX_PTX_RESOURCE_PAYLOAD={payload}",
            f"-DGFFX_PTX_RESOURCE_SCRIPT={rc}",
            "-P",
            str(VERIFIER),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _generate(root: Path):
    root.mkdir(parents=True)
    ptx = root / "kernels.ptx"
    ptx.write_bytes(b".version 8.5\\n.target sm_75\\n" + bytes(range(1, 120)))
    header = root / "gffx_cuda_ptx.h"
    payload = root / "gffx_cuda_ptx.bin"
    rc = root / "gffx_cuda_ptx.rc"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--input", str(ptx),
            "--payload", str(payload),
            "--rc", str(rc),
            "--header", str(header),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return header, payload, rc


def test_windows_native_cuda_build_discovers_python_independently():
    cmake = (ROOT / "cmake" / "GffxCuda.cmake").read_text(encoding="utf-8")
    source = cmake.split("if(WIN32)", 1)[1].split("else()", 1)[0]
    assert "find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)" in source
    assert 'COMMAND "${Python_EXECUTABLE}"' in source
    assert "GffxVerifyPtxResource.cmake" in source
    assert "GFFX_PTX_RESOURCE_HEADER=" in source


def test_generator_cli_and_postcondition_with_whitespace_path(tmp_path):
    header, payload, rc = _generate(tmp_path / "generated resource folder")
    assert header.exists() and payload.exists() and rc.exists()
    ok = _validate(header, payload, rc)
    assert ok.returncode == 0, ok.stdout + ok.stderr


@pytest.mark.parametrize("missing", ["header", "payload", "rc"])
def test_visual_studio_missing_output_is_a_hard_failure(tmp_path, missing):
    header, payload, rc = _generate(tmp_path / "generated")
    target = {"header": header, "payload": payload, "rc": rc}[missing]
    target.unlink()
    bad = _validate(header, payload, rc)
    assert bad.returncode != 0
    assert "did not create" in bad.stdout + bad.stderr


@pytest.mark.parametrize("empty", ["header", "payload", "rc"])
def test_visual_studio_empty_output_is_a_hard_failure(tmp_path, empty):
    header, payload, rc = _generate(tmp_path / "generated")
    target = {"header": header, "payload": payload, "rc": rc}[empty]
    target.write_bytes(b"")
    bad = _validate(header, payload, rc)
    assert bad.returncode != 0
    assert "empty output" in bad.stdout + bad.stderr


@pytest.mark.parametrize("corrupt", ["header", "rc"])
def test_missing_resource_marker_is_a_hard_failure(tmp_path, corrupt):
    header, payload, rc = _generate(tmp_path / "generated")
    target = {"header": header, "rc": rc}[corrupt]
    target.write_text("generated but invalid\\n", encoding="utf-8")
    bad = _validate(header, payload, rc)
    assert bad.returncode != 0
    assert "lacks its" in bad.stdout + bad.stderr
