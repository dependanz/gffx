#!/usr/bin/env python3
"""Probe exact GFFX CI artifacts and write verbose, secret-safe provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import site
import sys
import sysconfig
from typing import Any, Sequence


_CI_ENVIRONMENT_KEYS = (
    "CI",
    "GITHUB_ACTION",
    "GITHUB_ACTOR",
    "GITHUB_EVENT_NAME",
    "GITHUB_JOB",
    "GITHUB_REF",
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_NUMBER",
    "GITHUB_SHA",
    "RUNNER_ARCH",
    "RUNNER_NAME",
    "RUNNER_OS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    return {
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "abi_flags": getattr(sys, "abiflags", ""),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
        },
        "tooling": {
            name: _distribution_version(name)
            for name in ("pip", "setuptools", "wheel")
        },
        "ci": {
            key: os.environ[key]
            for key in _CI_ENVIRONMENT_KEYS
            if key in os.environ
        },
    }


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "name": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _write_report(output: Path | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"provenance: {output.resolve()}")


def _installed(args: argparse.Namespace) -> int:
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise RuntimeError(f"wheel does not exist: {wheel}")

    optional_before = {
        name: name in sys.modules
        for name in ("torch", "gffx._torch", "gffx.cuda")
    }
    import gffx

    optional_after_base_import = {
        name: name in sys.modules
        for name in ("torch", "gffx._torch", "gffx.cuda")
    }
    if any(optional_after_base_import.values()):
        raise RuntimeError(
            "base import eagerly loaded an optional component: "
            + repr(optional_after_base_import)
        )

    capabilities = gffx.capabilities()
    if capabilities["gpu_probed"] is not False:
        raise RuntimeError("base capabilities unexpectedly probed a GPU")
    native = capabilities["native_core"]
    if not native["available"] or native["abi_version"] != "1.0":
        raise RuntimeError(f"native ABI is unavailable or incompatible: {native!r}")
    if native["limited_api"] != "0x030A0000":
        raise RuntimeError(f"unexpected CPython Limited API floor: {native!r}")

    distribution = metadata.distribution("gffx")
    requirements = list(distribution.requires or ())
    if requirements:
        raise RuntimeError(f"GFFX wheel acquired runtime dependencies: {requirements!r}")

    adapter = gffx.torch
    import torch

    if adapter is not gffx.torch:
        raise RuntimeError("gffx.torch did not remain a stable lazy module attribute")
    if "gffx._torch" not in sys.modules:
        raise RuntimeError("the private PyTorch adapter binary did not load")
    if not hasattr(torch.ops.gffx_internal, "_foundation_probe"):
        raise RuntimeError("gffx_internal::_foundation_probe was not registered")

    report = {
        "schema": "gffx-ci-provenance-v1",
        "mode": "installed",
        "artifact": _artifact(wheel),
        "distribution": {
            "name": distribution.metadata["Name"],
            "version": distribution.version,
            "requires_python": distribution.metadata.get("Requires-Python"),
            "requires_dist": requirements,
            "files": len(list(distribution.files or ())),
        },
        "base_import": {
            "optional_modules_before": optional_before,
            "optional_modules_after": optional_after_base_import,
            "gpu_probed": capabilities["gpu_probed"],
            "native_core": native,
        },
        "pytorch_adapter": {
            "version": torch.__version__,
            "private_module_loaded": "gffx._torch" in sys.modules,
            "foundation_probe_registered": hasattr(
                torch.ops.gffx_internal, "_foundation_probe"
            ),
        },
        "environment": _environment(),
    }
    _write_report(args.output, report)
    return 0


def _uninstalled(args: argparse.Namespace) -> int:
    spec = importlib.util.find_spec("gffx")
    distributions = [
        {
            "name": distribution.metadata.get("Name"),
            "version": distribution.version,
            "path": str(distribution.locate_file("")),
        }
        for distribution in metadata.distributions()
        if (distribution.metadata.get("Name") or "").lower() == "gffx"
    ]
    roots = {
        Path(path).resolve()
        for path in (*site.getsitepackages(), site.getusersitepackages())
        if path
    }
    leftovers = sorted(
        str(candidate)
        for root in roots
        if root.is_dir()
        for candidate in root.glob("gffx*")
    )
    if spec is not None or distributions or leftovers:
        raise RuntimeError(
            "GFFX uninstall left importable or distribution files: "
            f"spec={spec!r}, distributions={distributions!r}, leftovers={leftovers!r}"
        )
    report = {
        "schema": "gffx-ci-provenance-v1",
        "mode": "uninstalled",
        "import_spec": None,
        "distributions": distributions,
        "leftovers": leftovers,
        "environment": _environment(),
    }
    _write_report(args.output, report)
    return 0


def _artifacts(args: argparse.Namespace) -> int:
    directory = args.directory.resolve()
    if not directory.is_dir():
        raise RuntimeError(f"artifact directory does not exist: {directory}")
    output = args.output.resolve() if args.output else None
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and (output is None or path.resolve() != output)
    ]
    if not files:
        raise RuntimeError(f"artifact directory is empty: {directory}")
    report = {
        "schema": "gffx-ci-provenance-v1",
        "mode": "artifacts",
        "artifacts": [_artifact(path) for path in files],
        "environment": _environment(),
    }
    _write_report(args.output, report)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    installed = subparsers.add_parser("installed", help="probe an installed exact wheel")
    installed.add_argument("--wheel", type=Path, required=True)
    installed.add_argument("--output", type=Path)
    installed.set_defaults(function=_installed)

    uninstalled = subparsers.add_parser("uninstalled", help="prove uninstall cleanup")
    uninstalled.add_argument("--output", type=Path)
    uninstalled.set_defaults(function=_uninstalled)

    artifacts = subparsers.add_parser("artifacts", help="hash build outputs and record provenance")
    artifacts.add_argument("--directory", type=Path, required=True)
    artifacts.add_argument("--output", type=Path)
    artifacts.set_defaults(function=_artifacts)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.function(args))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
