#!/usr/bin/env python3
"""Canonical Phase 1 packaging matrices and workflow drift checks.

This module intentionally uses only the Python standard library.  The matrix is executable
project policy: workflows consume it, tests pin its cardinality, and maintainers can validate
workflow wiring without adding a YAML parser to GFFX's dependency-light base package.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


ACTION_PINS = {
    "actions/checkout": (
        "v7.0.1",
        "3d3c42e5aac5ba805825da76410c181273ba90b1",
    ),
    "actions/setup-python": (
        "v7.0.0",
        "5fda3b95a4ea91299a34e894583c3862153e4b97",
    ),
    "actions/upload-artifact": (
        "v7.0.1",
        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    ),
    "actions/download-artifact": (
        "v8.0.0",
        "70fc10c6e5e1ce46ad2ea6f2b72d43f7d47b13c3",
    ),
    "pypa/cibuildwheel": (
        "v4.2.0",
        "1828c10ab37f080699c7b81cea34097c684a7074",
    ),
}

SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
SUPPORTED_PYTORCH = ("2.10.0", "2.11.0", "2.12.1", "2.13.0")

# Insertion order is normative and is used by release-candidate artifact planning.
PLATFORMS: Mapping[str, Mapping[str, str]] = {
    "windows-x64": {
        "runner": "windows-2022",
        "artifact": "win_amd64",
        "cibw_arch": "AMD64",
        "wheel_tag": "cp310-abi3-win_amd64",
    },
    "linux-x64": {
        "runner": "ubuntu-24.04",
        "artifact": "manylinux_2_28_x86_64",
        "cibw_arch": "x86_64",
        "wheel_tag": "cp310-abi3-manylinux_2_28_x86_64",
    },
    "linux-arm64": {
        "runner": "ubuntu-24.04-arm",
        "artifact": "manylinux_2_28_aarch64",
        "cibw_arch": "aarch64",
        "wheel_tag": "cp310-abi3-manylinux_2_28_aarch64",
    },
    "macos-arm64": {
        "runner": "macos-26",
        "artifact": "macosx_14_0_arm64",
        "cibw_arch": "arm64",
        "wheel_tag": "cp310-abi3-macosx_14_0_arm64",
    },
}

WORKFLOW_NAMES = {
    "package-foundation-pr.yml",
    "package-foundation-nightly.yml",
    "package-foundation-rc.yml",
}

_USES_PATTERN = re.compile(
    r"^\s*-?\s*uses:\s+([^@\s]+)@([0-9a-f]{40})\s+#\s+(v[^\s]+)\s*$",
    re.MULTILINE,
)


def _entry(
    *,
    name: str,
    platform: str,
    python: str,
    pytorch: str,
    lane: str,
    blocking: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "platform": platform,
        "python": python,
        "pytorch": pytorch,
        "lane": lane,
        "blocking": blocking,
    }
    result.update(PLATFORMS[platform])
    return result


def pull_request_matrix() -> list[dict[str, Any]]:
    """Return the seven adopted representative and boundary lanes."""
    specifications = (
        ("linux-x64-floor-floor", "linux-x64", "3.10", "2.10.0", "boundary"),
        ("linux-x64-floor-current", "linux-x64", "3.10", "2.13.0", "boundary"),
        ("linux-x64-current-floor", "linux-x64", "3.14", "2.10.0", "boundary"),
        ("linux-x64-current-current", "linux-x64", "3.14", "2.13.0", "boundary"),
        ("windows-x64-current-current", "windows-x64", "3.14", "2.13.0", "platform"),
        ("linux-arm64-current-current", "linux-arm64", "3.14", "2.13.0", "platform"),
        ("macos-arm64-current-current", "macos-arm64", "3.14", "2.13.0", "platform"),
    )
    return [
        _entry(
            name=name,
            platform=platform,
            python=python,
            pytorch=pytorch,
            lane=lane,
        )
        for name, platform, python, pytorch, lane in specifications
    ]


def nightly_matrix(iso_week: int | None = None) -> list[dict[str, Any]]:
    """Return 36 blocking nightly lanes, rotating the intermediate PyTorch line weekly."""
    week = date.today().isocalendar().week if iso_week is None else iso_week
    if week < 1 or week > 53:
        raise ValueError("ISO week must be between 1 and 53")
    intermediate = "2.11.0" if week % 2 else "2.12.1"
    result: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        for python in SUPPORTED_PYTHONS:
            result.append(
                _entry(
                    name=f"current-{platform}-py{python}-torch2.13.0",
                    platform=platform,
                    python=python,
                    pytorch="2.13.0",
                    lane="current",
                )
            )
        for python in (SUPPORTED_PYTHONS[0], SUPPORTED_PYTHONS[-1]):
            result.append(
                _entry(
                    name=f"floor-{platform}-py{python}-torch2.10.0",
                    platform=platform,
                    python=python,
                    pytorch="2.10.0",
                    lane="floor",
                )
            )
            result.append(
                _entry(
                    name=f"intermediate-{platform}-py{python}-torch{intermediate}",
                    platform=platform,
                    python=python,
                    pytorch=intermediate,
                    lane="intermediate",
                )
            )
    return result


def release_candidate_build_matrix() -> list[dict[str, Any]]:
    """Return one minimum-version adapter build per supported CPU artifact."""
    result = []
    for platform, values in PLATFORMS.items():
        entry: dict[str, Any] = {
            "name": platform,
            "platform": platform,
            "python": SUPPORTED_PYTHONS[0],
            "pytorch": SUPPORTED_PYTORCH[0],
            "build_sdist": platform == "linux-x64",
        }
        entry.update(values)
        result.append(entry)
    return result


def release_candidate_test_matrix() -> list[dict[str, Any]]:
    """Return the exhaustive 4 platform x 5 Python x 4 PyTorch artifact-test matrix."""
    result = []
    for platform, values in PLATFORMS.items():
        for python in SUPPORTED_PYTHONS:
            for pytorch in SUPPORTED_PYTORCH:
                entry: dict[str, Any] = {
                    "name": f"{platform}-py{python}-torch{pytorch}",
                    "platform": platform,
                    "python": python,
                    "pytorch": pytorch,
                }
                entry.update(values)
                result.append(entry)
    return result


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, ensure_ascii=False)


def _render_include(
    entries: Sequence[Mapping[str, Any]],
    *,
    label: str,
    fields: Sequence[str],
    indent: int = 10,
) -> str:
    prefix = " " * indent
    child = " " * (indent + 2)
    value_prefix = " " * (indent + 4)
    lines = [f"{prefix}# BEGIN GFFX GENERATED {label}", f"{prefix}include:"]
    for entry in entries:
        first, *remaining = fields
        lines.append(f"{child}- {first}: {_yaml_scalar(entry[first])}")
        for field in remaining:
            lines.append(f"{value_prefix}{field}: {_yaml_scalar(entry[field])}")
    lines.append(f"{prefix}# END GFFX GENERATED {label}")
    return "\n".join(lines)


def render_pull_request_block() -> str:
    return _render_include(
        pull_request_matrix(),
        label="PR MATRIX",
        fields=("name", "platform", "runner", "python", "pytorch", "artifact", "cibw_arch"),
    )


def render_release_build_block() -> str:
    return _render_include(
        release_candidate_build_matrix(),
        label="RC BUILD MATRIX",
        fields=(
            "name", "platform", "runner", "python", "pytorch", "artifact", "cibw_arch",
            "build_sdist",
        ),
    )


def render_release_test_block() -> str:
    prefix = " " * 10
    child = " " * 12
    lines = [f"{prefix}# BEGIN GFFX GENERATED RC TEST MATRIX", f"{prefix}platform:"]
    lines.extend(f"{child}- {_yaml_scalar(value)}" for value in PLATFORMS)
    lines.append(f"{prefix}python:")
    lines.extend(f"{child}- {_yaml_scalar(value)}" for value in SUPPORTED_PYTHONS)
    lines.append(f"{prefix}pytorch:")
    lines.extend(f"{child}- {_yaml_scalar(value)}" for value in SUPPORTED_PYTORCH)
    platform_entries = [dict(platform=key, **values) for key, values in PLATFORMS.items()]
    include = _render_include(
        platform_entries,
        label="RC TEST PLATFORM METADATA",
        fields=("platform", "runner", "artifact", "cibw_arch"),
    )
    lines.extend(include.splitlines())
    lines.append(f"{prefix}# END GFFX GENERATED RC TEST MATRIX")
    return "\n".join(lines)


def _extract_generated(text: str, label: str) -> str | None:
    begin = " " * 10 + f"# BEGIN GFFX GENERATED {label}"
    end = " " * 10 + f"# END GFFX GENERATED {label}"
    start = text.find(begin)
    if start < 0:
        return None
    finish = text.find(end, start)
    if finish < 0:
        return None
    return text[start : finish + len(end)]


def validate_workflows(root: Path) -> list[str]:
    """Return deterministic workflow drift and security errors; an empty list is success."""
    root = Path(root)
    directory = root / ".github" / "workflows"
    errors: list[str] = []
    actual_names = {path.name for path in directory.glob("*.yml")}
    if actual_names != WORKFLOW_NAMES:
        errors.append(
            "workflow files differ: expected %s, found %s"
            % (sorted(WORKFLOW_NAMES), sorted(actual_names))
        )
    texts = {
        name: (directory / name).read_text(encoding="utf-8")
        for name in WORKFLOW_NAMES
        if (directory / name).is_file()
    }
    combined = "\n".join(texts.values())
    lowered = combined.lower()

    for forbidden in (
        "pull_request_target", "@main", "@master", "@latest", "ubuntu-latest",
        "windows-latest", "macos-latest", "id-token: write", "contents: write", "twine",
    ):
        if forbidden in lowered:
            errors.append(f"forbidden workflow token: {forbidden}")

    if "permissions:\n  contents: read" not in combined:
        errors.append("top-level permissions must contain only contents: read")
    if "persist-credentials: false" not in combined:
        errors.append("checkout must disable credential persistence")

    references = _USES_PATTERN.findall(combined)
    for owner, sha, version in references:
        expected = ACTION_PINS.get(owner)
        if expected is None:
            errors.append(f"unknown action reference: {owner}")
        elif (version, sha) != expected:
            errors.append(f"action pin drift: {owner}@{sha} # {version}")
    for line in combined.splitlines():
        if "uses:" in line and _USES_PATTERN.match(line) is None:
            errors.append(f"mutable or malformed action reference: {line.strip()}")

    pr = texts.get("package-foundation-pr.yml", "")
    if _extract_generated(pr, "PR MATRIX") != render_pull_request_block():
        errors.append("pull-request matrix block differs from canonical seven lanes")
    for token in ("tools/verify_foundation.py", "package-foundation-required"):
        if token not in pr:
            errors.append(f"pull-request workflow missing {token}")

    nightly = texts.get("package-foundation-nightly.yml", "")
    for token in (
        "tools/ci_matrix.py emit-nightly --github-output matrix",
        "fromJSON(needs.matrix.outputs.matrix)",
        'python-version: "3.15"',
        "continue-on-error: true",
        "tools/verify_foundation.py",
    ):
        if token not in nightly:
            errors.append(f"nightly workflow missing {token}")

    rc = texts.get("package-foundation-rc.yml", "")
    if _extract_generated(rc, "RC BUILD MATRIX") != render_release_build_block():
        errors.append("release-candidate build matrix differs from canonical four artifacts")
    if _extract_generated(rc, "RC TEST MATRIX") != render_release_test_block():
        errors.append("release-candidate test matrix differs from canonical 80 environments")
    for token in (
        "needs: build-artifacts", "actions/download-artifact@", "tools/ci_smoke.py installed",
        "tools/ci_smoke.py uninstalled", "python -m pip uninstall --yes gffx",
        "CIBW_BUILD_FRONTEND", "--no-build-isolation", "GFFX_BUILD_PYTORCH=ON",
    ):
        if token not in rc:
            errors.append(f"release-candidate workflow missing {token}")
    artifact_tests = rc.split("artifact-tests:", maxsplit=1)
    if len(artifact_tests) == 2 and "python -m pip wheel ." in artifact_tests[1]:
        errors.append("release-candidate artifact tests rebuild the source tree")
    return errors


def _write_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        raise RuntimeError("--github-output requires the GITHUB_OUTPUT environment variable")
    with Path(output_path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{name}={value}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit-nightly", help="emit the rotating nightly JSON matrix")
    emit.add_argument("--iso-week", type=int)
    emit.add_argument("--github-output", metavar="NAME")
    check = subparsers.add_parser("check", help="validate workflow files against this policy")
    check.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "emit-nightly":
        payload = json.dumps({"include": nightly_matrix(args.iso_week)}, separators=(",", ":"))
        if args.github_output:
            _write_github_output(args.github_output, payload)
        else:
            print(payload)
        return 0
    errors = validate_workflows(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("workflow contracts: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
