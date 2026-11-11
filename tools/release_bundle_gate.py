#!/usr/bin/env python3
"""Refuse an incomplete or mismatched stable-release distribution bundle.

Only inspected, exact-version wheels and one sdist may enter the publication directory.
This is an artifact inventory gate, not a substitute for wheel installation, functional,
or real-NVIDIA-device validation in the release workflow.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import tarfile
import tomllib
import zipfile

if __package__:
    from .ci_matrix import PLATFORMS
else:
    from ci_matrix import PLATFORMS


# CUDA is an optional-at-runtime provider, but is part of these release artifacts.
# Linux ARM64 and macOS are intentionally CPU-only until qualified separately.
CUDA_PROVIDERS = {
    "windows-x64": "gffx/gffx_cuda12.dll",
    "linux-x64": "gffx/libgffx_cuda12.so",
}

PLATFORM_BINARIES = {
    "windows-x64": {"gffx/gffx_core.dll", "gffx/_core.abi3.pyd", "gffx/_torch.abi3.pyd"},
    "linux-x64": {"gffx/libgffx_core.so", "gffx/_core.abi3.so", "gffx/_torch.abi3.so"},
    "linux-arm64": {"gffx/libgffx_core.so", "gffx/_core.abi3.so", "gffx/_torch.abi3.so"},
    "macos-arm64": {"gffx/libgffx_core.dylib", "gffx/_core.abi3.so", "gffx/_torch.abi3.so"},
}


class ReleaseBundleError(ValueError):
    """The exact release bundle does not satisfy the public platform contract."""


def _digest(path: Path) -> dict[str, str | int]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _check_wheel(path: Path, version: str, platform: str, dependencies_declared: bool) -> list[str]:
    try:
        with zipfile.ZipFile(path) as wheel:
            names = set(wheel.namelist())
            root = f"gffx-{version}.dist-info"
            metadata = Parser().parsestr(wheel.read(f"{root}/METADATA").decode("utf-8"))
            wheel_metadata = Parser().parsestr(wheel.read(f"{root}/WHEEL").decode("utf-8"))
    except (KeyError, zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise ReleaseBundleError(f"invalid wheel {path.name}: {error}") from error

    if metadata.get("Name", "").lower() != "gffx" or metadata.get("Version") != version:
        raise ReleaseBundleError(f"package identity mismatch in {path.name}")
    if not metadata.get("Requires-Python"):
        raise ReleaseBundleError(f"missing Requires-Python in {path.name}")
    requires = metadata.get_all("Requires-Dist", [])
    if dependencies_declared and not requires:
        raise ReleaseBundleError(f"missing project runtime dependencies in {path.name}")
    if not dependencies_declared and requires:
        raise ReleaseBundleError(f"unexpected project runtime dependencies in {path.name}")
    tag = PLATFORMS[platform]["wheel_tag"]
    if tag not in wheel_metadata.get_all("Tag", []):
        raise ReleaseBundleError(f"wheel tag mismatch for {platform}: expected {tag}")

    expected = {"gffx/__init__.py", "gffx/examples/cornell_box.py"} | PLATFORM_BINARIES[platform]
    if platform in CUDA_PROVIDERS:
        expected.add(CUDA_PROVIDERS[platform])
    missing = sorted(expected - names)
    if missing:
        raise ReleaseBundleError(f"{path.name} is missing release payload: {missing}")
    if platform not in CUDA_PROVIDERS and any("gffx_cuda12" in name for name in names):
        raise ReleaseBundleError(f"{platform} unexpectedly bundles unqualified CUDA provider")
    return requires


def _check_sdist(path: Path, version: str) -> None:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            names = set(archive.getnames())
            expected = {f"gffx-{version}/pyproject.toml", f"gffx-{version}/src/gffx/__init__.py"}
            if not expected <= names:
                raise ReleaseBundleError(f"sdist missing source files: {sorted(expected - names)}")
            source = archive.extractfile(f"gffx-{version}/pyproject.toml")
            if source is None:
                raise ReleaseBundleError("sdist pyproject.toml is not a regular file")
            project = tomllib.loads(source.read().decode("utf-8"))["project"]
    except (tarfile.TarError, UnicodeDecodeError, KeyError) as error:
        raise ReleaseBundleError(f"invalid sdist {path.name}: {error}") from error
    if project.get("name") != "gffx" or project.get("version") != version:
        raise ReleaseBundleError("sdist version does not match wheel release")


def verify(dist: Path, pyproject: Path) -> dict:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    if project.get("name") != "gffx":
        raise ReleaseBundleError("source package must be gffx")
    version = project["version"]
    if not version or any(char not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!+_-" for char in version):
        raise ReleaseBundleError("invalid source version")
    if not dist.is_dir():
        raise ReleaseBundleError(f"distribution directory does not exist: {dist}")

    expected_wheels = {
        platform: f"gffx-{version}-{properties['wheel_tag']}.whl"
        for platform, properties in PLATFORMS.items()
    }
    expected_files = {*expected_wheels.values(), f"gffx-{version}.tar.gz"}
    actual_files = {path.name for path in dist.iterdir() if path.is_file()}
    extra = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    if extra or missing:
        raise ReleaseBundleError(f"release artifacts incomplete: missing={missing}, unexpected={extra}")
    if any(not (dist / name).is_file() for name in expected_files):
        raise ReleaseBundleError("some release artifacts are not regular files")

    dependencies = None
    for platform, filename in expected_wheels.items():
        current = _check_wheel(dist / filename, version, platform, bool(project.get("dependencies")))
        if dependencies is None:
            dependencies = current
        elif current != dependencies:
            raise ReleaseBundleError(f"inconsistent runtime dependencies for {platform}")
    _check_sdist(dist / f"gffx-{version}.tar.gz", version)
    return {
        "version": version,
        "platforms": sorted(expected_wheels),
        "cuda_providers": sorted(CUDA_PROVIDERS),
        "artifacts": [_digest(dist / filename) for filename in sorted(expected_files)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--pyproject", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml")
    args = parser.parse_args()
    try:
        report = verify(args.dist, args.pyproject)
    except (ReleaseBundleError, OSError, ValueError) as error:
        parser.exit(1, f"RELEASE BLOCKED: {error}\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
