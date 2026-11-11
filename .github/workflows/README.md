# Packaging-foundation workflows

These workflows implement only GFFX's adopted Phase 1 packaging and adapter-load cadence. They do
not publish a package and do not claim operation correctness, gradients, compilation/export
support, CUDA kernels, streaming performance, or release readiness. The executable matrix policy
lives in `tools/ci_matrix.py`.

## Cadences

- `package-foundation-pr.yml` runs seven representative platform/version boundary lanes. Every
  lane invokes `tools/verify_foundation.py`; `package-foundation-required` is the stable branch-rule
  check name.
- `package-foundation-nightly.yml` asks `tools/ci_matrix.py` for 36 blocking lanes. PyTorch 2.11
  runs in odd ISO weeks and 2.12 in even ISO weeks. A separate Linux x64 Python 3.15/PyTorch-nightly
  preview is deliberately nonblocking.
- `package-foundation-rc.yml` builds four adapter-enabled `cp310-abi3` wheels once against the
  PyTorch 2.10 CPU floor, hashes them, then installs those exact downloaded artifacts in 100 clean
  platform/Python/PyTorch environments. It also builds one sdist beside the Linux x64 wheel. There
  is no source rebuild in an artifact-test job and no package-index upload step.

All third-party actions are pinned to full commit SHAs with their release tag in a comment. The
workflows use fixed runner labels, read-only repository permissions, and checkout with credential
persistence disabled. Build and smoke provenance is retained as workflow artifacts for 14 days.

## Protected stable publication

The new publish-multiplatform.yml is manually dispatched by the repository owner
with an immutable stable tag and successful package-foundation-rc run ID.
A read-only preflight verifies every wheel/sdist hash, its source/RC receipt,
and the Windows and Linux NVIDIA device reports. The separate OIDC publishing
job uploads only approved bytes, without executing candidate source code.

Before using it, configure GitHub's pypi environment with required owner-review
protection, and register publish-multiplatform.yml with that environment as a
Trusted Publisher for the gffx project on PyPI. These account settings cannot
be applied by a PR alone; no automatic tag/release/branch trigger is enabled.

The RC remains blocked until protected real NVIDIA runners pass the hardware
gates and all 100 supported platform/Python/PyTorch environments are green.

## Local policy checks

Run the matrix and workflow contracts without adding a YAML dependency:

```text
python tools/ci_matrix.py check
python -m pytest -q tests/packaging/test_ci_contract.py
```

The generated matrix blocks are owned by `tools/ci_matrix.py`. Change the policy module and its
tests first, then regenerate or deliberately update the marked blocks; the check rejects drift.

These definitions are source configuration until they execute on GitHub-hosted runners. A local
green contract proves their matrix wiring and static safety rules, not that the hosted jobs pass.
