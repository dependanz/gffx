#!/usr/bin/env python3
"""Bind a complete RC distribution bundle to an exact source SHA and Actions run ID.

The receipt is an integrity claim, not a cryptographic signature. A future
publisher must additionally check that the originating GitHub Actions run
completed successfully and that owner approval was granted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

if __package__:
    from .release_bundle_gate import verify as verify_bundle
else:
    from release_bundle_gate import verify as verify_bundle


class ReceiptError(ValueError):
    pass


def _identity(source_sha: str, run_id: int | str) -> tuple[str, int]:
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ReceiptError("source must be an exact lowercase 40-hex-character commit")
    try:
        number = int(run_id)
    except (TypeError, ValueError) as error:
        raise ReceiptError("RC workflow run ID must be an integer") from error
    if number <= 0 or str(number) != str(run_id):
        raise ReceiptError("RC workflow run ID must be a positive canonical integer")
    return source_sha, number


def receipt_for(dist: Path, project_file: Path, source_sha: str, run_id: int | str) -> dict:
    sha, run = _identity(source_sha, run_id)
    approved = verify_bundle(dist, project_file)
    return {
        "schema": "gffx-rc-receipt-v1",
        "source_commit": sha,
        "rc_run_id": run,
        "name": "gffx",
        "version": approved["version"],
        "platforms": approved["platforms"],
        "cuda_providers": approved["cuda_providers"],
        "artifacts": approved["artifacts"],
    }


def check_receipt(
    dist: Path,
    project_file: Path,
    source_sha: str,
    run_id: int | str,
    receipt_path: Path,
) -> dict:
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReceiptError("missing or malformed candidate receipt") from error
    expected = receipt_for(dist, project_file, source_sha, run_id)
    if payload != expected:
        raise ReceiptError(
            "RC receipt differs from the exact candidate SHA, run, version, "
            "platform set, or distribution hashes"
        )
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("create", "verify"))
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--pyproject", type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "create":
            data = receipt_for(args.dist, args.pyproject, args.source_sha, args.run_id)
            if args.receipt.exists():
                raise ReceiptError("refusing to overwrite an existing receipt")
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n",
                                    encoding="utf-8")
        else:
            data = check_receipt(args.dist, args.pyproject, args.source_sha,
                                 args.run_id, args.receipt)
    except (ReceiptError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"RELEASE RECEIPT BLOCKED: {error}\n")
    print(json.dumps({"status": "accepted", "version": data["version"],
                      "rc_run_id": data["rc_run_id"],
                      "artifacts": len(data["artifacts"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
