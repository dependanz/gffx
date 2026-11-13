"""No incomplete or inconsistently labeled stable distribution can be published."""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from tools.ci_matrix import PLATFORMS
from tools.release_bundle_gate import ReleaseBundleError, verify


def _write_wheel(path: Path, platform: str, version: str, *, cuda: bool = True, mismatched: bool = False):
    suffix = ".pyd" if platform == "windows-x64" else ".so"
    core = "gffx_core.dll" if platform == "windows-x64" else (
        "libgffx_core.dylib" if platform == "macos-arm64" else "libgffx_core.so"
    )
    tag = PLATFORMS[platform]["wheel_tag"]
    with zipfile.ZipFile(path, "w") as file:
        file.writestr("gffx/__init__.py", "")
        file.writestr("gffx/examples/cornell_box.py", "")
        file.writestr("gffx/" + core, "binary")
        extension_prefix = "" if platform == "windows-x64" else ".abi3"
        file.writestr("gffx/_core" + extension_prefix + suffix, "binary")
        file.writestr("gffx/_torch" + extension_prefix + suffix, "binary")
        if cuda and platform in ("windows-x64", "linux-x64"):
            file.writestr("gffx/" + ("gffx_cuda12.dll" if platform == "windows-x64" else "libgffx_cuda12.so"), "plugin")
        info = f"gffx-{version}.dist-info/"
        file.writestr(info + "METADATA", f"Name: gffx\nVersion: {'0.0.0' if mismatched else version}\nRequires-Python: >=3.10\nRequires-Dist: torch>=2.10.0,<=2.14.1\n")
        filename_platform_tags = path.name.removeprefix(f"gffx-{version}-cp310-abi3-").removesuffix(".whl").split(".")
        file.writestr(info + "WHEEL", "Wheel-Version: 1.0\n" + "".join(f"Tag: cp310-abi3-{platform_tag}\n" for platform_tag in filename_platform_tags))


def _bundle(root: Path, version: str = "0.2.2") -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "gffx"\nversion = "{version}"\ndependencies = ["torch>=2.10.0,<=2.14.1"]\nrequires-python = ">=3.10"\n')
    dist = root / "dist"
    dist.mkdir()
    for platform, info in PLATFORMS.items():
        _write_wheel(dist / f"gffx-{version}-{info['wheel_tag']}.whl", platform, version)
    archive = dist / f"gffx-{version}.tar.gz"
    with tarfile.open(archive, "w:gz") as file:
        for filename, content in [("pyproject.toml", pyproject.read_bytes()), ("src/gffx/__init__.py", b"")]:
            entry = tarfile.TarInfo(f"gffx-{version}/{filename}")
            entry.size = len(content)
            file.addfile(entry, io.BytesIO(content))
    return dist, pyproject


def test_complete_bundle_is_accepted(tmp_path):
    dist, pyproject = _bundle(tmp_path)
    result = verify(dist, pyproject)
    assert result["version"] == "0.2.2"
    assert len(result["platforms"]) == len(PLATFORMS)
    assert len(result["artifacts"]) == len(PLATFORMS) + 1


@pytest.mark.parametrize("missing_platform", list(PLATFORMS))
def test_missing_wheel_fails_closed(tmp_path, missing_platform):
    dist, pyproject = _bundle(tmp_path)
    (dist / f"gffx-0.2.2-{PLATFORMS[missing_platform]['wheel_tag']}.whl").unlink()
    with pytest.raises(ReleaseBundleError, match="incomplete"):
        verify(dist, pyproject)


@pytest.mark.parametrize("platform", ["windows-x64", "linux-x64"])
def test_missing_cuda_provider_blocks_x64_release(tmp_path, platform):
    dist, pyproject = _bundle(tmp_path)
    _write_wheel(dist / f"gffx-0.2.2-{PLATFORMS[platform]['wheel_tag']}.whl", platform, "0.2.2", cuda=False)
    with pytest.raises(ReleaseBundleError, match="cuda12"):
        verify(dist, pyproject)


def test_metadata_version_mismatch_fails(tmp_path):
    dist, pyproject = _bundle(tmp_path)
    _write_wheel(dist / f"gffx-0.2.2-{PLATFORMS['macos-arm64']['wheel_tag']}.whl", "macos-arm64", "0.2.2", mismatched=True)
    with pytest.raises(ReleaseBundleError, match="identity mismatch"):
        verify(dist, pyproject)


def test_unexpected_artifact_fails(tmp_path):
    dist, pyproject = _bundle(tmp_path)
    (dist / "gffx-0.1.4-py3-none-any.whl").touch()
    with pytest.raises(ReleaseBundleError, match="unexpected"):
        verify(dist, pyproject)


def test_repaired_composite_manylinux_tags_are_accepted(tmp_path):
    dist, pyproject = _bundle(tmp_path)
    old = dist / "gffx-0.2.2-cp310-abi3-manylinux_2_28_x86_64.whl"
    repaired = dist / "gffx-0.2.2-cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    old.unlink()
    _write_wheel(repaired, "linux-x64", "0.2.2")
    result = verify(dist, pyproject)
    assert any(row["filename"] == repaired.name for row in result["artifacts"])


def test_duplicate_compatible_manylinux_tags_fail_closed(tmp_path):
    dist, pyproject = _bundle(tmp_path)
    old = dist / "gffx-0.2.2-cp310-abi3-manylinux_2_28_x86_64.whl"
    new = dist / "gffx-0.2.2-cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    new.write_bytes(old.read_bytes())
    with pytest.raises(ReleaseBundleError, match="2 matching wheels"):
        verify(dist, pyproject)


def test_missing_sdist_fails(tmp_path):
    dist, pyproject = _bundle(tmp_path)
    (dist / "gffx-0.2.2.tar.gz").unlink()
    with pytest.raises(ReleaseBundleError, match="incomplete"):
        verify(dist, pyproject)


@pytest.mark.parametrize("extension", ["_core.pyd", "_torch.pyd"])
def test_windows_release_refuses_missing_native_extensions(tmp_path, extension):
    """Both real Windows .pyd extension payloads remain mandatory for release."""
    dist, pyproject = _bundle(tmp_path)
    wheel = dist / f"gffx-0.2.2-{PLATFORMS['windows-x64']['wheel_tag']}.whl"
    with zipfile.ZipFile(wheel) as source:
        contents = [(name, source.read(name)) for name in source.namelist()]
    with zipfile.ZipFile(wheel, "w") as target:
        for name, data in contents:
            if name != f"gffx/{extension}":
                target.writestr(name, data)
    with pytest.raises(ReleaseBundleError, match="missing release payload") as exc:
        verify(dist, pyproject)
    assert extension in str(exc.value)


def test_windows_release_rejects_misnamed_abi3_pyd_extensions(tmp_path):
    """An incorrect .abi3.pyd spelling cannot disguise absent importable .pyd files."""
    dist, pyproject = _bundle(tmp_path)
    wheel = dist / f"gffx-0.2.2-{PLATFORMS['windows-x64']['wheel_tag']}.whl"
    with zipfile.ZipFile(wheel) as source:
        contents = [(name, source.read(name)) for name in source.namelist()]
    with zipfile.ZipFile(wheel, "w") as target:
        for name, data in contents:
            if name in {"gffx/_core.pyd", "gffx/_torch.pyd"}:
                name = name.removesuffix(".pyd") + ".abi3.pyd"
            target.writestr(name, data)
    with pytest.raises(ReleaseBundleError, match="missing release payload"):
        verify(dist, pyproject)
