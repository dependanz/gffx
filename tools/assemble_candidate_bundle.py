#!/usr/bin/env python3
"""Assemble exactly one provenance-checked candidate wheel per supported platform.

Inputs are four separate GitHub Actions artifacts, not a wheelhouse merged blindly
from mutable filenames. This tool never builds, signs, uploads or publishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

if __package__:
    from .ci_matrix import PLATFORMS
else:
    from ci_matrix import PLATFORMS


class AssemblyError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(artifacts: Path, out: Path, pyproject: Path) -> dict:
    version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    if not artifacts.is_dir() or out.exists():
        raise AssemblyError("artifact input must exist and output must be a new path")
    expected_dirs = {"wheel-" + row["artifact"]: (platform, row)
                     for platform, row in PLATFORMS.items()}
    actual_dirs = {path.name for path in artifacts.iterdir() if path.is_dir()}
    if actual_dirs != set(expected_dirs):
        raise AssemblyError(f"platform artifact folders differ: expected={sorted(expected_dirs)} actual={sorted(actual_dirs)}")
    if any(not path.is_dir() for path in artifacts.iterdir()):
        raise AssemblyError("unexpected file at artifact input root")

    validated: list[Path] = []
    records = []
    for folder_name, (platform, row) in expected_dirs.items():
        folder = artifacts / folder_name
        wheel_name = f"gffx-{version}-{row['wheel_tag']}.whl"
        sdist_name = f"gffx-{version}.tar.gz"
        expected_names = {wheel_name, "provenance.json"}
        if platform == "linux-x64":
            expected_names.add(sdist_name)
        items = list(folder.iterdir())
        if {p.name for p in items} != expected_names or not all(p.is_file() for p in items):
            raise AssemblyError(f"{platform}: files differ: expected={sorted(expected_names)} actual={sorted(p.name for p in items)}")
        try:
            provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
            declared = provenance["artifacts"]
        except (KeyError, ValueError, OSError) as error:
            raise AssemblyError(f"{platform}: invalid recorded build provenance") from error
        if provenance.get("mode") != "artifacts" or not isinstance(declared, list):
            raise AssemblyError(f"{platform}: missing artifact provenance")
        if {r.get("name") for r in declared} != expected_names - {"provenance.json"} or len(declared) != len(expected_names) - 1:
            raise AssemblyError(f"{platform}: provenance file inventory differs")
        for item in declared:
            path = folder / item["name"]
            if item.get("sha256") != sha256(path) or item.get("size_bytes") != path.stat().st_size:
                raise AssemblyError(f"{platform}: artifact bytes differ from recorded provenance")
            validated.append(path)
            records.append({"platform": platform, "name": path.name, "sha256": item["sha256"]})
    if len(validated) != len(PLATFORMS) + 1 or len({path.name for path in validated}) != len(validated):
        raise AssemblyError("release assembly is incomplete or contains duplicate distribution names")
    out.mkdir(parents=True, exist_ok=False)
    for path in validated:
        shutil.copyfile(path, out / path.name)
    return {"version": version, "artifacts": sorted(records, key=lambda item: item["name"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pyproject", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml")
    args = parser.parse_args()
    try:
        result = assemble(args.artifacts, args.out, args.pyproject)
    except (AssemblyError, OSError, KeyError, ValueError) as error:
        parser.exit(1, f"CANDIDATE BLOCKED: {error}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
