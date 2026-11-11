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
import re
from pathlib import Path
import tarfile
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 build/test environments
    import tomli as tomllib
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


def wheel_name_matches(filename: str, version: str, wheel_tag: str) -> bool:
    """Accept PEP 425 compressed platform tags, including auditwheel's multiple tags.

    Require the exact gffx version, cp310-abi3, and the canonical platform tag.
    A Linux wheel tagged for both 2.27 and 2.28 is compatible with the 2.28 floor;
    that is not a missing wheel or a reason to rename the audited artifact.
    """
    prefix = f"gffx-{version}-cp310-abi3-"
    if not filename.startswith(prefix) or not filename.endswith(".whl"):
        return False
    suffix = filename[len(prefix):-4]
    platform = wheel_tag.removeprefix("cp310-abi3-")
    return platform in suffix.split(".")




def _digest(path: Path) -> dict[str, str | int]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _normalize_dependency(value: str) -> tuple[str, tuple[str, ...]]:
    """Normalize this release line's exact simple (unmarked, no extras) dependencies."""
    canonical = value.replace(" ", "")
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)((?:[<>=!~].*)?)", canonical)
    if match is None:
        raise ReleaseBundleError(f"unsupported runtime dependency syntax: {value!r}")
    conditions = [term for term in match.group(2).split(",") if term]
    if any(re.fullmatch(r"(?:===|~=|==|!=|<=|>=|<|>)[A-Za-z0-9_.+*-]+", term) is None
           for term in conditions):
        raise ReleaseBundleError(f"unsupported dependency specifier syntax: {value!r}")
    return match.group(1).lower().replace("_", "-"), tuple(sorted(conditions))


def _check_wheel(path: Path, version: str, platform: str, project: dict) -> list[str]:
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
    if metadata.get("Requires-Python") != project["requires-python"]:
        raise ReleaseBundleError(f"Requires-Python differs from source in {path.name}")
    requires = metadata.get_all("Requires-Dist", [])
    if sorted(map(_normalize_dependency, requires)) != sorted(map(_normalize_dependency, project.get("dependencies", []))):
        raise ReleaseBundleError(f"runtime dependencies differ from source in {path.name}")
    tag = PLATFORMS[platform]["wheel_tag"]
    if tag not in wheel_metadata.get_all("Tag", []):
        raise ReleaseBundleError(f"wheel tag mismatch for {platform}: expected {tag}")
    suffix = path.name.removeprefix(f"gffx-{version}-cp310-abi3-").removesuffix(".whl")
    declared = set(wheel_metadata.get_all("Tag", []))
    filename_tags = {"cp310-abi3-" + platform_tag for platform_tag in suffix.split(".")}
    if declared != filename_tags:
        raise ReleaseBundleError(f"wheel filename tags disagree with WHEEL metadata for {path.name}")

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

    # auditwheel may add a compatible, higher-priority manylinux tag to the filename.
    # Match the required platform among compressed tags rather than demanding a
    # single hard-coded spelling of a perfectly valid repaired wheel.
    wheels = sorted(path for path in dist.iterdir() if path.is_file() and path.suffix == ".whl")
    expected_wheels: dict[str, Path] = {}
    for platform, properties in PLATFORMS.items():
        matching = [path for path in wheels if
                    wheel_name_matches(path.name, version, properties["wheel_tag"])]
        if len(matching) != 1:
            raise ReleaseBundleError(
                f"release artifacts incomplete for {platform}: found {len(matching)} matching wheels"
            )
        expected_wheels[platform] = matching[0]
    sdist_name = f"gffx-{version}.tar.gz"
    actual_files = {path.name for path in dist.iterdir() if path.is_file()}
    expected_files = {path.name for path in expected_wheels.values()} | {sdist_name}
    extra = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    if (extra or missing or len(set(expected_wheels.values())) != len(PLATFORMS)
            or any(not path.is_file() for path in dist.iterdir())):
        raise ReleaseBundleError(f"release artifacts incomplete: missing={missing}, unexpected={extra}")
    if not (dist / sdist_name).is_file():
        raise ReleaseBundleError("missing source distribution")

    dependencies = None
    for platform, wheel in expected_wheels.items():
        current = _check_wheel(wheel, version, platform, project)
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
