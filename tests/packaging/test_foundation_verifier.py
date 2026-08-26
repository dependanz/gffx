"""Phase 1 Step 11 contracts for the package-foundation verifier."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = REPO_ROOT / "tools" / "verify_foundation.py"


def load_verifier():
    assert VERIFIER_PATH.is_file(), "tools/verify_foundation.py has not been implemented"
    spec = importlib.util.spec_from_file_location("gffx_foundation_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wheel(path: Path, *, requires_dist: str | None = None, cuda_plugin: bool = False) -> None:
    metadata = (
        "Metadata-Version: 2.2\n"
        "Name: gffx\n"
        "Version: 0.2.0.dev0\n"
        "Requires-Python: >=3.10\n"
    )
    if requires_dist is not None:
        metadata += f"Requires-Dist: {requires_dist}\n"

    names = {
        "gffx/__init__.py": b"",
        "gffx/_version.py": b"",
        "gffx/_capabilities.py": b"",
        "gffx/cuda/__init__.py": b"",
        "gffx/torch/__init__.py": b"",
        "gffx/_core.pyd": b"native",
        "gffx/gffx_core.dll": b"native",
        "gffx-0.2.0.dev0.dist-info/METADATA": metadata.encode(),
        "gffx-0.2.0.dev0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nTag: cp310-abi3-win_amd64\n"
        ),
        "gffx-0.2.0.dev0.dist-info/licenses/LICENSE": b"MIT",
        "gffx-0.2.0.dev0.dist-info/RECORD": b"",
    }
    if cuda_plugin:
        names["gffx/gffx_cuda12.dll"] = b"optional provider"

    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in names.items():
            archive.writestr(name, payload)


def write_sdist(path: Path, *, generated_path: str | None = None) -> None:
    names = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "src/gffx/__init__.py",
        "docs/INSTALLATION.md",
        "docs/BUILDING.md",
        "docs/DEPENDENCIES.md",
        "docs/SUPPORT_STATUS.md",
        "tools/verify_foundation.py",
        "tests/packaging/test_foundation_verifier.py",
    ]
    if generated_path is not None:
        names.append(generated_path)

    with tarfile.open(path, "w:gz") as archive:
        for relative in names:
            payload = b"foundation fixture"
            info = tarfile.TarInfo(f"gffx-0.2.0.dev0/{relative}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_verifier_declares_every_step_11_gate():
    verifier = load_verifier()
    assert verifier.GATE_NAMES == (
        "source_hygiene",
        "base_dependency_scan",
        "python_contracts",
        "native_contracts",
        "artifact_build",
        "artifact_contents",
        "compiler_free_install",
        "optional_component_failures",
        "explicit_adapter_load",
        "uninstall_cleanup",
        "source_prerequisite_failures",
    )


def test_ctest_summary_parser_records_the_completed_test_count():
    verifier = load_verifier()
    output = "100% tests passed, 0 tests failed out of 18"
    assert verifier._parse_ctest_pass_count(output) == 18


def test_current_source_inventory_is_clean_and_complete():
    verifier = load_verifier()
    paths = verifier.collect_source_paths(REPO_ROOT)

    for required in (
        Path("README.md"),
        Path("docs/INSTALLATION.md"),
        Path("tests/packaging/test_foundation_verifier.py"),
        Path("tools/verify_foundation.py"),
    ):
        assert required in paths
    assert verifier.scan_source_hygiene(REPO_ROOT, paths) == []


def test_hygiene_scan_detects_generated_prototype_and_secret_inputs_without_echoing_secrets(
    tmp_path: Path,
):
    verifier = load_verifier()
    secret = "ghp_" + "abcdefghijklmnopqrstuvwxyz" + "1234567890"
    files = {
        Path("src/gffx/ops/stale.py"): "prototype",
        Path("src/gffx/__pycache__/stale.pyc"): "generated",
        Path(".env"): f"GITHUB_TOKEN={secret}",
    }
    for relative, contents in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")

    findings = verifier.scan_source_hygiene(tmp_path, sorted(files))
    codes = {finding.code for finding in findings}
    assert {"LEGACY_PROTOTYPE", "GENERATED_PATH", "SENSITIVE_PATH"} <= codes
    assert secret not in repr(findings)


def test_base_dependency_scan_rejects_an_undeclared_import(tmp_path: Path):
    verifier = load_verifier()
    assert verifier.scan_base_dependencies(REPO_ROOT) == []

    package = tmp_path / "src" / "gffx"
    (package / "cuda").mkdir(parents=True)
    for path in (package / "__init__.py", package / "_version.py", package / "cuda" / "__init__.py"):
        path.write_text("", encoding="utf-8")
    (package / "_capabilities.py").write_text("import numpy\n", encoding="utf-8")

    findings = verifier.scan_base_dependencies(tmp_path)
    assert [(finding.code, finding.path) for finding in findings] == [
        ("UNDECLARED_BASE_IMPORT", "src/gffx/_capabilities.py")
    ]


def test_wheel_inspection_accepts_the_base_contract_and_rejects_runtime_leaks(tmp_path: Path):
    verifier = load_verifier()
    good = tmp_path / "gffx-0.2.0.dev0-cp310-abi3-win_amd64.whl"
    write_wheel(good)
    report = verifier.inspect_wheel(good)
    assert report["runtime_dependencies"] == 0
    assert report["abi_tag"] == "abi3"

    dependency_leak = tmp_path / "dependency-leak.whl"
    write_wheel(dependency_leak, requires_dist="numpy")
    with pytest.raises(verifier.VerificationError, match="Requires-Dist"):
        verifier.inspect_wheel(dependency_leak)

    cuda_leak = tmp_path / "cuda-leak.whl"
    write_wheel(cuda_leak, cuda_plugin=True)
    with pytest.raises(verifier.VerificationError, match="CUDA provider"):
        verifier.inspect_wheel(cuda_leak)


def test_sdist_inspection_accepts_source_and_rejects_generated_output(tmp_path: Path):
    verifier = load_verifier()
    good = tmp_path / "gffx-0.2.0.dev0.tar.gz"
    write_sdist(good)
    assert verifier.inspect_sdist(good)["members"] == 10

    bad = tmp_path / "generated.tar.gz"
    write_sdist(bad, generated_path="build/compiler-output.obj")
    with pytest.raises(verifier.VerificationError, match="generated path"):
        verifier.inspect_sdist(bad)


def test_ignore_policy_covers_local_credentials_and_keys():
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in (".env", ".env.*", "*.pem", "*.key"):
        assert pattern in ignore
