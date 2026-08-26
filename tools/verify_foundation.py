#!/usr/bin/env python3
"""Run GFFX Phase 1 package-foundation verification from clean inputs.

This module intentionally uses only the Python standard library. Development tools are invoked as
subprocesses so they remain build/test inputs and never become GFFX runtime dependencies.
"""

from __future__ import annotations

import argparse
import ast
import email.parser
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import venv
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, NamedTuple, Sequence


GATE_NAMES = (
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

BASE_MODULES = (
    Path("src/gffx/__init__.py"),
    Path("src/gffx/_version.py"),
    Path("src/gffx/_capabilities.py"),
    Path("src/gffx/cuda/__init__.py"),
)
LEGACY_PACKAGE_ROOTS = {"context", "io", "linalg", "obj", "ops", "random", "ray"}
GENERATED_PARTS = {
    ".eggs",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
}
GENERATED_SUFFIXES = {
    ".a",
    ".dll",
    ".dylib",
    ".egg",
    ".exp",
    ".lib",
    ".o",
    ".obj",
    ".pdb",
    ".pyc",
    ".pyo",
    ".so",
    ".whl",
}
SENSITIVE_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
TEXT_SUFFIXES = {
    "",
    ".bat",
    ".c",
    ".cmake",
    ".cpp",
    ".cu",
    ".h",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
)
REQUIRED_WHEEL_FILES = {
    "gffx/__init__.py",
    "gffx/_version.py",
    "gffx/_capabilities.py",
    "gffx/cuda/__init__.py",
    "gffx/torch/__init__.py",
}
REQUIRED_SDIST_FILES = {
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "src/gffx/__init__.py",
    "docs/INSTALLATION.md",
    "docs/BUILDING.md",
    "docs/DEPENDENCIES.md",
    "docs/SUPPORT_STATUS.md",
    "tools/verify_foundation.py",
    "tests/packaging/test_foundation_verifier.py",
}


class VerificationError(RuntimeError):
    """A package-foundation contract was violated."""


class Finding(NamedTuple):
    code: str
    path: str
    detail: str


class CommandResult(NamedTuple):
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_command(command: Sequence[str]) -> str:
    return " ".join('"%s"' % value if " " in value else value for value in command)


def run_command(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    expect_success: bool = True,
    expected_text: Sequence[str] = (),
) -> CommandResult:
    normalized = tuple(os.fspath(value) for value in command)
    print("+", _display_command(normalized), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        normalized,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    duration = time.perf_counter() - started
    result = CommandResult(
        normalized,
        completed.returncode,
        completed.stdout,
        completed.stderr,
        duration,
    )
    combined = completed.stdout + "\n" + completed.stderr
    success = completed.returncode == 0
    if success != expect_success:
        expectation = "succeed" if expect_success else "fail"
        tail = "\n".join(combined.splitlines()[-80:])
        raise VerificationError(
            f"command was expected to {expectation} but returned {completed.returncode}: "
            f"{_display_command(normalized)}\n{tail}"
        )
    missing = [needle for needle in expected_text if needle.lower() not in combined.lower()]
    if missing:
        raise VerificationError(
            "command output did not contain expected diagnostic text " + repr(missing)
        )
    print(f"  return={completed.returncode} duration={duration:.2f}s", flush=True)
    return result


def collect_source_paths(root: Path) -> list[Path]:
    root = root.resolve()
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise VerificationError(
            "source inventory requires a Git worktree: "
            + completed.stderr.decode(errors="replace").strip()
        )
    paths = []
    for value in completed.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if not value:
            continue
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise VerificationError(f"unsafe source inventory path: {value!r}")
        if (root / relative).is_file():
            paths.append(relative)
    return sorted(set(paths), key=lambda item: item.as_posix())


def _is_sensitive_path(relative: Path) -> bool:
    lower_name = relative.name.lower()
    return (
        lower_name in SENSITIVE_NAMES
        or lower_name.startswith(".env.")
        or relative.suffix.lower() in SENSITIVE_SUFFIXES
    )


def _has_generated_component(relative: Path) -> bool:
    lowered = {part.lower() for part in relative.parts}
    return bool(lowered & GENERATED_PARTS) or any(
        part.lower().endswith(".egg-info") for part in relative.parts
    )


def _is_legacy_prototype(relative: Path) -> bool:
    parts = relative.as_posix().split("/")
    return len(parts) >= 3 and parts[:2] == ["src", "gffx"] and parts[2] in LEGACY_PACKAGE_ROOTS


def scan_source_hygiene(root: Path, paths: Iterable[Path]) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for relative in sorted(set(paths), key=lambda item: item.as_posix()):
        display = relative.as_posix()
        target = root / relative
        if relative.is_absolute() or ".." in relative.parts:
            findings.append(Finding("UNSAFE_PATH", display, "path escapes the source root"))
            continue
        if _is_sensitive_path(relative):
            findings.append(Finding("SENSITIVE_PATH", display, "credential/key filename is forbidden"))
        if _has_generated_component(relative) or relative.suffix.lower() in GENERATED_SUFFIXES:
            findings.append(Finding("GENERATED_PATH", display, "generated/cache artifact is forbidden"))
        if _is_legacy_prototype(relative):
            findings.append(Finding("LEGACY_PROTOTYPE", display, "prototype namespace is not admitted"))

        if (
            target.is_file()
            and target.stat().st_size <= 2 * 1024 * 1024
            and target.suffix.lower() in TEXT_SUFFIXES
        ):
            payload = target.read_bytes()
            if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
                findings.append(
                    Finding("SENSITIVE_CONTENT", display, "credential-like content is forbidden")
                )
    return sorted(set(findings))


def scan_base_dependencies(root: Path) -> list[Finding]:
    root = root.resolve()
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    findings: list[Finding] = []
    for relative in BASE_MODULES:
        path = root / relative
        if not path.is_file():
            findings.append(Finding("MISSING_BASE_MODULE", relative.as_posix(), "required file absent"))
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative.as_posix())
        except SyntaxError as error:
            findings.append(Finding("INVALID_PYTHON", relative.as_posix(), str(error)))
            continue
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".", 1)[0])
        for name in sorted(imported - stdlib - {"gffx"}):
            findings.append(
                Finding(
                    "UNDECLARED_BASE_IMPORT",
                    relative.as_posix(),
                    f"base module imports third-party root {name!r}",
                )
            )
    return findings


def _safe_archive_names(names: Iterable[str], *, label: str) -> list[str]:
    normalized: list[str] = []
    for name in names:
        if "\\" in name:
            raise VerificationError(f"{label} contains a backslash path: {name!r}")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise VerificationError(f"{label} contains an unsafe path: {name!r}")
        normalized.append(path.as_posix())
    return normalized


def inspect_wheel(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with zipfile.ZipFile(path) as archive:
        names = _safe_archive_names(archive.namelist(), label="wheel")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise VerificationError("wheel must contain exactly one METADATA and one WHEEL file")
        metadata_text = archive.read(metadata_names[0]).decode("utf-8")
        wheel_text = archive.read(wheel_names[0]).decode("utf-8")

    metadata = email.parser.Parser().parsestr(metadata_text)
    if metadata.get("Name") != "gffx" or metadata.get("Version") != "0.2.0.dev0":
        raise VerificationError("wheel identity is not gffx 0.2.0.dev0")
    if metadata.get("Requires-Python") != ">=3.10":
        raise VerificationError("wheel Requires-Python must be exactly >=3.10")
    requires = metadata.get_all("Requires-Dist", [])
    if requires:
        raise VerificationError("wheel contains forbidden Requires-Dist runtime dependencies")

    missing = sorted(REQUIRED_WHEEL_FILES - set(names))
    if missing:
        raise VerificationError("wheel is missing required package files: " + repr(missing))
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        raise VerificationError("wheel does not package the MIT LICENSE file")
    if not any(re.search(r"^gffx/_core(?:\.abi3)?\.(?:pyd|so)$", name) for name in names):
        raise VerificationError("wheel does not contain the CPython Limited-API core loader")
    if not any(
        name in {"gffx/gffx_core.dll", "gffx/libgffx_core.so", "gffx/libgffx_core.dylib"}
        for name in names
    ):
        raise VerificationError("wheel does not contain the framework-neutral native core")
    if any("gffx_cuda12" in name.lower() for name in names):
        raise VerificationError("base wheel contains the optional CUDA provider")
    if any(re.search(r"^gffx/_torch(?:\.abi3)?\.(?:pyd|so)$", name) for name in names):
        raise VerificationError("base wheel contains the optional PyTorch adapter")
    generated = [
        name
        for name in names
        if "__pycache__" in PurePosixPath(name).parts
        or PurePosixPath(name).suffix.lower() in {".a", ".exp", ".lib", ".obj", ".pyc"}
    ]
    if generated:
        raise VerificationError("wheel contains generated/build-only payload: " + repr(generated))

    tags = [line.split(":", 1)[1].strip() for line in wheel_text.splitlines() if line.startswith("Tag:")]
    abi_tags = {tag.split("-")[1] for tag in tags if len(tag.split("-")) == 3}
    if "abi3" not in abi_tags:
        raise VerificationError("wheel does not declare an abi3 compatibility tag")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "members": len(names),
        "runtime_dependencies": 0,
        "abi_tag": "abi3",
        "tags": tags,
    }


def inspect_sdist(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = _safe_archive_names((member.name for member in members), label="sdist")
    roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if len(roots) != 1:
        raise VerificationError("sdist must have one top-level source directory")
    root = next(iter(roots))
    relative_names = {
        PurePosixPath(*PurePosixPath(name).parts[1:]).as_posix() for name in names
    }
    missing = sorted(REQUIRED_SDIST_FILES - relative_names)
    if missing:
        raise VerificationError("sdist is missing required source files: " + repr(missing))
    for name in sorted(relative_names):
        relative = Path(name)
        if _has_generated_component(relative) or relative.suffix.lower() in GENERATED_SUFFIXES:
            raise VerificationError(f"sdist contains generated path: {name}")
        if _is_sensitive_path(relative):
            raise VerificationError(f"sdist contains sensitive path: {name}")
        if _is_legacy_prototype(relative):
            raise VerificationError(f"sdist contains unsupported prototype path: {name}")
    return {
        "path": str(path),
        "root": root,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "members": len(names),
    }


def create_source_snapshot(source_root: Path, destination: Path) -> list[Path]:
    paths = collect_source_paths(source_root)
    findings = scan_source_hygiene(source_root, paths)
    if findings:
        raise VerificationError("source hygiene failed: " + repr(findings))
    destination.mkdir(parents=True, exist_ok=False)
    for relative in paths:
        source = source_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return paths


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _clean_python_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "GFFX_CUDA_PLUGIN_PATH"):
        environment.pop(name, None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def run_python_contracts(source_root: Path, python: Path) -> dict[str, Any]:
    result = run_command(
        [python, "-m", "pytest", "-q", "tests/python", "tests/pytorch", "tests/packaging"],
        cwd=source_root,
    )
    summary = next(
        (line.strip() for line in reversed(result.stdout.splitlines()) if " passed" in line),
        "pytest completed",
    )
    return {"summary": summary, "duration_seconds": result.duration_seconds}


def _ctest_junit_count(suite: ET.Element, name: str, *, required: bool = False) -> int:
    value = suite.get(name)
    if value is None:
        if required:
            raise VerificationError(f"native CTest JUnit result is missing {name!r}")
        return 0
    try:
        count = int(value)
    except ValueError as error:
        raise VerificationError(
            f"native CTest JUnit result has a non-integer {name!r} count"
        ) from error
    if count < 0:
        raise VerificationError(f"native CTest JUnit result has a negative {name!r} count")
    return count


def _parse_ctest_junit_pass_count(path: Path) -> int:
    try:
        suite = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise VerificationError("native CTest JUnit result is missing or malformed") from error
    if suite.tag != "testsuite":
        raise VerificationError("native CTest JUnit result does not contain one testsuite root")

    tests = _ctest_junit_count(suite, "tests", required=True)
    if tests == 0:
        raise VerificationError("native CTest JUnit result reports zero executed tests")

    incomplete = {
        name: _ctest_junit_count(suite, name)
        for name in ("failures", "errors", "disabled", "skipped")
    }
    if any(incomplete.values()):
        raise VerificationError(
            "native CTest JUnit result did not report a complete pass: " + repr(incomplete)
        )

    cases = suite.findall("testcase")
    if len(cases) != tests:
        raise VerificationError(
            "native CTest JUnit test count does not match its testcase records"
        )
    if any(suite.findall(f".//{name}") for name in ("failure", "error", "skipped")):
        raise VerificationError("native CTest JUnit testcase records are not a complete pass")
    return tests


def run_native_contracts(snapshot: Path, work: Path) -> dict[str, Any]:
    build = work / "native"
    junit = work / "native-ctest.xml"
    configure = [
        "cmake",
        "-S",
        snapshot,
        "-B",
        build,
        "-DBUILD_TESTING=ON",
        "-DGFFX_BUILD_PYTHON=OFF",
        "-DGFFX_BUILD_PYTORCH=OFF",
        "-DGFFX_ENABLE_CUDA=OFF",
    ]
    if shutil.which("ninja"):
        configure.extend(["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"])
    run_command(configure, cwd=work)
    run_command(["cmake", "--build", build, "--config", "Release", "--parallel"], cwd=work)
    result = run_command(
        [
            "ctest",
            "--test-dir",
            build,
            "-C",
            "Release",
            "--output-on-failure",
            "--output-junit",
            junit,
        ],
        cwd=work,
    )
    return {
        "tests_passed": _parse_ctest_junit_pass_count(junit),
        "duration_seconds": result.duration_seconds,
    }


def build_artifacts(snapshot: Path, work: Path, python: Path) -> tuple[Path, Path, dict[str, Any]]:
    output = work / "artifacts"
    output.mkdir(parents=True, exist_ok=False)
    result = run_command(
        [python, "-m", "build", "--wheel", "--sdist", "--outdir", output, snapshot],
        cwd=work,
    )
    wheels = list(output.glob("*.whl"))
    sdists = list(output.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise VerificationError(
            f"artifact build produced {len(wheels)} wheels and {len(sdists)} sdists"
        )
    return wheels[0], sdists[0], {"duration_seconds": result.duration_seconds}


def _write_compiler_sentinel(work: Path) -> tuple[Path, Path]:
    marker = work / "compiler-was-invoked"
    if os.name == "nt":
        script = work / "forbidden-compiler.cmd"
        script.write_text(
            f"@echo off\r\necho invoked>\"{marker}\"\r\nexit /b 99\r\n", encoding="utf-8"
        )
    else:
        script = work / "forbidden-compiler"
        script.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n", encoding="utf-8")
        script.chmod(0o755)
    return script, marker


def verify_install_optional_and_uninstall(
    wheel: Path, work: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    environment_path = work / "install-environment"
    venv.EnvBuilder(with_pip=True, clear=False).create(environment_path)
    python = _venv_python(environment_path)
    environment = _clean_python_environment()
    sentinel, marker = _write_compiler_sentinel(work)
    environment["CC"] = str(sentinel)
    environment["CXX"] = str(sentinel)

    install = run_command(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--only-binary",
            ":all:",
            wheel,
        ],
        cwd=work,
        env=environment,
    )
    if marker.exists():
        raise VerificationError("installing the wheel invoked a compiler sentinel")

    smoke = textwrap.dedent(
        """
        import importlib.metadata
        import json
        import sys
        import gffx

        forbidden = ('torch', 'numpy', 'jax', 'tensorflow', 'cupy', 'matplotlib', 'PIL', 'trimesh')
        leaked = sorted(name for name in forbidden if name in sys.modules)
        assert not leaked, leaked
        assert 'gffx._core' not in sys.modules
        report = gffx.capabilities()
        assert report['native_core']['available'], report
        assert report['native_core']['abi_version'] == '1.0', report
        assert report['native_core']['limited_api'] == '0x030A0000', report
        assert report['gpu_probed'] is False, report
        distribution = importlib.metadata.distribution('gffx')
        assert not list(distribution.requires or []), distribution.requires
        files = [str(distribution.locate_file(item)) for item in distribution.files or []]
        print(json.dumps({'files': files, 'version': gffx.__version__}))
        """
    )
    smoke_result = run_command([python, "-c", smoke], cwd=work, env=environment)
    inventory = json.loads(smoke_result.stdout.splitlines()[-1])

    absent_components = textwrap.dedent(
        """
        import gffx
        report = gffx.cuda.capabilities()
        assert report['probe_attempted'] is True, report
        assert report['gpu_probed'] is True, report
        assert report['status'] == 'not found', report
        assert report['result_flags']['optional_provider_absent'] is True, report
        assert report['result_flags']['partial_failure'] is False, report
        assert report['records'], report
        try:
            gffx.torch
        except ImportError as error:
            message = str(error)
            assert 'PyTorch' in message and 'pip install torch' in message and 'base package' in message
        else:
            raise AssertionError('adapter unexpectedly loaded without PyTorch')
        print(report['status'])
        """
    )
    absent_result = run_command([python, "-c", absent_components], cwd=work, env=environment)

    corrupt_plugin = work / ("corrupt-gffx-cuda.dll" if os.name == "nt" else "corrupt-gffx-cuda.so")
    corrupt_plugin.write_bytes(b"not a loadable provider")
    corrupt_environment = dict(environment)
    corrupt_environment["GFFX_CUDA_PLUGIN_PATH"] = str(corrupt_plugin.resolve())
    corrupt_probe = textwrap.dedent(
        """
        import gffx
        report = gffx.cuda.capabilities()
        assert report['probe_attempted'] is True, report
        assert report['gpu_probed'] is True, report
        assert report['records'], report
        assert report['status'].startswith('load failed'), report
        assert report['result_flags']['optional_provider_absent'] is False, report
        assert report['result_flags']['partial_failure'] is True, report
        print(report['status'])
        """
    )
    corrupt_result = run_command([python, "-c", corrupt_probe], cwd=work, env=corrupt_environment)

    run_command([python, "-m", "pip", "uninstall", "--yes", "gffx"], cwd=work, env=environment)
    leftovers = [path for path in inventory["files"] if Path(path).exists()]
    if leftovers:
        raise VerificationError("uninstall left recorded distribution files: " + repr(leftovers))
    post_uninstall = textwrap.dedent(
        """
        import importlib.util
        import pathlib
        import site
        assert importlib.util.find_spec('gffx') is None
        leftovers = []
        for root in site.getsitepackages():
            leftovers.extend(str(path) for path in pathlib.Path(root).glob('gffx*'))
        assert not leftovers, leftovers
        print('uninstall clean')
        """
    )
    uninstall_result = run_command([python, "-c", post_uninstall], cwd=work, env=environment)
    return (
        {
            "compiler_invoked": False,
            "installed_version": inventory["version"],
            "recorded_files": len(inventory["files"]),
            "duration_seconds": install.duration_seconds + smoke_result.duration_seconds,
        },
        {
            "absent_status": absent_result.stdout.strip().splitlines()[-1],
            "corrupt_status": corrupt_result.stdout.strip().splitlines()[-1],
        },
        {"leftovers": 0, "duration_seconds": uninstall_result.duration_seconds},
    )


def verify_explicit_adapter(adapter_python: Path, work: Path) -> dict[str, Any]:
    adapter_python = adapter_python.resolve()
    if not adapter_python.is_file():
        raise VerificationError(f"adapter Python does not exist: {adapter_python}")
    environment = _clean_python_environment()
    code = textwrap.dedent(
        """
        import re
        import sys
        import gffx
        assert 'torch' not in sys.modules
        assert 'gffx._torch' not in sys.modules
        adapter = gffx.torch
        import torch
        match = re.match(r'^(\\d+)\\.(\\d+)', torch.__version__)
        assert match and tuple(map(int, match.groups())) >= (2, 10), torch.__version__
        assert adapter is gffx.torch
        assert 'gffx._torch' in sys.modules
        assert hasattr(torch.ops.gffx_internal, '_foundation_probe')
        print(torch.__version__)
        """
    )
    result = run_command([adapter_python, "-c", code], cwd=work, env=environment)
    return {"pytorch_version": result.stdout.strip().splitlines()[-1]}


def verify_source_prerequisite_failures(
    snapshot: Path, work: Path, no_framework_python: Path
) -> dict[str, Any]:
    cmake = shutil.which("cmake")
    if not cmake:
        raise VerificationError("cmake is required for source prerequisite verification")

    common = [cmake, "-S", snapshot, "-DBUILD_TESTING=OFF"]
    missing_compiler = work / "missing-compiler"
    run_command(
        [
            *common,
            "-G",
            "Ninja",
            "-B",
            work / "prerequisite-missing-compiler",
            f"-DCMAKE_MAKE_PROGRAM={cmake}",
            f"-DCMAKE_C_COMPILER={missing_compiler}",
        ],
        cwd=work,
        expect_success=False,
        expected_text=("CMAKE_C_COMPILER", "not a full path"),
    )

    run_command(
        [
            *common,
            "-B",
            work / "prerequisite-missing-pytorch",
            "-DGFFX_BUILD_PYTHON=ON",
            "-DGFFX_BUILD_PYTORCH=ON",
            f"-DPython_EXECUTABLE={no_framework_python}",
        ],
        cwd=work,
        expect_success=False,
        expected_text=("requires PyTorch 2.10 or newer",),
    )

    run_command(
        [
            *common,
            "-B",
            work / "prerequisite-missing-cuda",
            "-DGFFX_ENABLE_CUDA=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE",
        ],
        cwd=work,
        expect_success=False,
        expected_text=("CUDAToolkit",),
    )
    return {"expected_failures": 3}


def _record_gate(report: dict[str, Any], name: str, action) -> Any:
    print(f"\n== {name} ==", flush=True)
    started = time.perf_counter()
    try:
        details = action()
    except Exception as error:
        report["gates"][name] = {
            "status": "failed",
            "duration_seconds": round(time.perf_counter() - started, 6),
            "error": str(error),
        }
        raise
    report["gates"][name] = {
        "status": "passed",
        "duration_seconds": round(time.perf_counter() - started, 6),
        "details": details,
    }
    return details


def verify_development_environment(python: Path, source_root: Path) -> None:
    """Fail before creating evidence when the declared verification tools are absent."""
    completed = subprocess.run(
        [os.fspath(python), "-I", "-c", "import build, pytest"],
        cwd=source_root,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise VerificationError(
            "package-foundation verification requires the declared development tool group; "
            "run `python -m pip install --group development` in an isolated environment "
            "and invoke the verifier with that Python executable"
        )


def run_foundation_verification(
    *, source_root: Path, work: Path, python: Path, adapter_python: Path
) -> dict[str, Any]:
    source_root = source_root.resolve()
    work = work.resolve()
    python = python.resolve()
    adapter_python = adapter_python.resolve()
    verify_development_environment(python, source_root)
    if work.exists():
        raise VerificationError(f"work directory already exists; choose a clean path: {work}")
    work.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(source_root),
        "python": str(python),
        "adapter_python": str(adapter_python),
        "gates": {},
    }
    report_path = work / "report.json"

    try:
        paths = _record_gate(
            report,
            "source_hygiene",
            lambda: _verify_and_snapshot(source_root, work),
        )
        snapshot = Path(paths["snapshot"])
        _record_gate(
            report,
            "base_dependency_scan",
            lambda: _require_no_findings(scan_base_dependencies(source_root)),
        )
        _record_gate(
            report,
            "python_contracts",
            lambda: run_python_contracts(source_root, python),
        )
        _record_gate(
            report,
            "native_contracts",
            lambda: run_native_contracts(snapshot, work),
        )
        artifact_result = _record_gate(
            report,
            "artifact_build",
            lambda: _build_artifact_report(snapshot, work, python),
        )
        wheel = Path(artifact_result["wheel"])
        sdist = Path(artifact_result["sdist"])
        _record_gate(
            report,
            "artifact_contents",
            lambda: {"wheel": inspect_wheel(wheel), "sdist": inspect_sdist(sdist)},
        )

        install_bundle: dict[str, Any] = {}

        def install_action():
            install, optional, uninstall = verify_install_optional_and_uninstall(wheel, work)
            install_bundle.update(install=install, optional=optional, uninstall=uninstall)
            return install

        _record_gate(report, "compiler_free_install", install_action)
        _record_gate(
            report,
            "optional_component_failures",
            lambda: install_bundle["optional"],
        )
        _record_gate(
            report,
            "explicit_adapter_load",
            lambda: verify_explicit_adapter(adapter_python, work),
        )
        _record_gate(report, "uninstall_cleanup", lambda: install_bundle["uninstall"])
        no_framework_python = _venv_python(work / "install-environment")
        _record_gate(
            report,
            "source_prerequisite_failures",
            lambda: verify_source_prerequisite_failures(snapshot, work, no_framework_python),
        )
    except Exception:
        report["status"] = "failed"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise

    missing = [name for name in GATE_NAMES if report["gates"].get(name, {}).get("status") != "passed"]
    if missing:
        raise VerificationError("verification did not pass every required gate: " + repr(missing))
    report["status"] = "passed"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _verify_and_snapshot(source_root: Path, work: Path) -> dict[str, Any]:
    paths = collect_source_paths(source_root)
    findings = scan_source_hygiene(source_root, paths)
    _require_no_findings(findings)
    snapshot = work / "source-snapshot"
    copied = create_source_snapshot(source_root, snapshot)
    return {"files": len(copied), "snapshot": str(snapshot)}


def _require_no_findings(findings: Sequence[Finding]) -> dict[str, Any]:
    if findings:
        raise VerificationError("verification findings: " + repr(list(findings)))
    return {"findings": 0}


def _build_artifact_report(snapshot: Path, work: Path, python: Path) -> dict[str, Any]:
    wheel, sdist, details = build_artifacts(snapshot, work, python)
    return {"wheel": str(wheel), "sdist": str(sdist), **details}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="GFFX Git worktree to verify (default: repository containing this script)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="new, non-existing directory for generated verification evidence",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python with the declared test and packaging groups installed",
    )
    parser.add_argument(
        "--adapter-python",
        type=Path,
        required=True,
        help="Python environment containing a PyTorch 2.10+ ready GFFX adapter wheel",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = make_parser().parse_args(argv)
    try:
        report = run_foundation_verification(
            source_root=arguments.source_root,
            work=arguments.work_dir,
            python=arguments.python,
            adapter_python=arguments.adapter_python,
        )
    except VerificationError as error:
        print(f"FOUNDATION VERIFICATION FAILED: {error}", file=sys.stderr)
        return 1
    print("\nFOUNDATION VERIFICATION PASSED")
    print(json.dumps({"gates": len(report["gates"]), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
