#!/usr/bin/env python3
"""Fail-closed preflight for manually approved multi-platform PyPI publishing.

Run only in the publisher's read-only preflight job. The OIDC publishing job
must never checkout or execute code from a candidate source revision.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

if __package__:
    from .rc_candidate_receipt import check_receipt
else:
    from rc_candidate_receipt import check_receipt


REPOSITORY = "dependanz/gffx"
OWNER = "dependanz"
RC_WORKFLOW = "package-foundation-rc"


class PublicationError(ValueError):
    pass


def stable_version(tag: str) -> str:
    if not re.fullmatch(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag):
        raise PublicationError("stable publisher requires an exact vMAJOR.MINOR.PATCH tag")
    version = tag[1:]
    if tuple(int(part) for part in version.split(".")) < (0, 2, 2):
        raise PublicationError("publisher must not republish legacy 0.2.1 or earlier")
    return version


def verify_run_data(data: dict, run_id: int | str) -> None:
    if data.get("name") != RC_WORKFLOW or data.get("event") != "workflow_dispatch":
        raise PublicationError("receipt must originate from manually dispatched release-candidate workflow")
    if data.get("status") != "completed" or data.get("conclusion") != "success":
        raise PublicationError("release-candidate workflow has not fully passed")
    if data.get("repository", {}).get("full_name") != REPOSITORY:
        raise PublicationError("release-candidate workflow belongs to a different repository")
    if str(data.get("id")) != str(run_id):
        raise PublicationError("release-candidate run ID does not match requested run")


def read_run_from_github(run_id: int | str) -> dict:
    if not re.fullmatch(r"[1-9][0-9]*", str(run_id)):
        raise PublicationError("run ID must be a positive canonical decimal integer")
    command = [
        "gh", "api", f"repos/{REPOSITORY}/actions/runs/{run_id}",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=True, timeout=30)
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        raise PublicationError("could not independently confirm the source Actions run") from error


def verify_device_report(path: Path, expected_platform: str) -> None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PublicationError("missing or invalid NVIDIA device qualification report") from error
    provenance = report.get("provenance", {})
    platform = provenance.get("platform", "")
    if report.get("status") != "passed" or report.get("tests_executed", 0) < 57:
        raise PublicationError(f"{expected_platform} NVIDIA device tests did not pass")
    if not platform.startswith(expected_platform) or not provenance.get("devices"):
        raise PublicationError(f"{expected_platform} device provenance is missing or mismatched")
    if not provenance.get("nvcc_release"):
        raise PublicationError("CUDA toolkit version is missing from NVIDIA evidence")


def preflight(
    *, tag: str, run_id: str, git_head: str,
    dist: Path, receipt: Path, project: Path,
    windows_report: Path, linux_report: Path,
    run_data: dict,
) -> dict:
    version = stable_version(tag)
    if re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
        raise PublicationError("tag checkout is not an exact 40-character source commit")
    try:
        pyproject = tomllib.loads(project.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as error:
        raise PublicationError("missing or malformed project version metadata") from error
    if pyproject["project"]["name"] != "gffx" or pyproject["project"]["version"] != version:
        raise PublicationError("stable tag does not match project name and version")
    verify_run_data(run_data, run_id)
    checked = check_receipt(dist, project, git_head, run_id, receipt)
    verify_device_report(windows_report, "win")
    verify_device_report(linux_report, "linux")
    return {
        "status": "approved-for-human-publishing-review",
        "version": version,
        "source_commit": git_head,
        "rc_run_id": int(run_id),
        "wheel_and_sdist_count": len(checked["artifacts"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--windows-report", type=Path, required=True)
    parser.add_argument("--linux-report", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml")
    args = parser.parse_args(argv)
    try:
        if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
            raise PublicationError("unexpected GitHub repository")
        if os.environ.get("GITHUB_ACTOR") != OWNER:
            raise PublicationError("publisher must be dispatched by the repository owner")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report = preflight(
            tag=args.tag, run_id=args.run_id, git_head=sha,
            dist=args.dist, receipt=args.receipt,
            project=args.project, windows_report=args.windows_report,
            linux_report=args.linux_report,
            run_data=read_run_from_github(args.run_id),
        )
    except (PublicationError, ValueError, OSError, KeyError) as error:
        parser.exit(1, f"PUBLICATION BLOCKED: {error}\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
