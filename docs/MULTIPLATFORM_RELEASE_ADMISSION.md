# Cross-platform stable release admission (proposed for 0.2.2 onward)

## Publication invariant

A stable GFFX release is an **atomic platform release**, not a successful upload from one builder.
A release event must not upload to PyPI unless every platform in the canonical matrix
('tools/ci_matrix.py') has an exact, tested wheel, and the matching source archive is
present. The matrix is the executable source of truth; adding a platform increases the
required artifact set rather than silently leaving publication permissive.

As of this proposal, the CPU-wheel matrix is:

- Windows x86-64, CPython 3.10+ ABI3
- Linux x86-64, manylinux 2.28, CPython 3.10+ ABI3
- Linux ARM64, manylinux 2.28, CPython 3.10+ ABI3
- macOS ARM64, macOS 14+, CPython 3.10+ ABI3

macOS Intel is not yet in the canonical matrix and must be explicitly added and qualified
before being advertised as supported. "Unix" is not a blanket compatibility guarantee:
Linux distributions meeting the manylinux floor are the initial Unix targets. Other
Unix platforms need their own compatibility matrix and test hardware.

## Accelerator policy

The Windows x86-64 and Linux x86-64 release wheels must contain the lazy optional
GFFX CUDA provider. A GPU or driver is **not** required for installation or CPU use;
the provider remains opt-in at runtime. Inclusion is necessary but insufficient: both
platforms also need actual NVIDIA-device functional and numerical tests on the exact
candidate binaries, including operation failures, driver compatibility, and GPU
fallback behavior.

Linux ARM64 and macOS ARM64 are CPU admission targets until specifically qualified.
Do not claim CUDA on macOS. Apple Silicon GPU acceleration requires a distinct
Metal/MPS implementation or an explicitly tested Torch/MPS operation path; installing
a CUDA package would not supply it.

## Required release pipeline

1. Select an exact release source commit and version, not an independently moving branch.
2. Build each platform artifact and an sdist reproducibly from that same commit.
3. Install each wheel in a clean target environment, execute native/adapter functionality,
   exercise gradients and the packaged Cornell-box example, and verify uninstall.
4. Run a representative supported Python/PyTorch compatibility matrix. Do not infer
   support for PyTorch versions that were never tested.
5. Qualify the CUDA-enabled Windows and Linux release binaries on real NVIDIA hosts.
6. Assemble the complete exact-artifact set into a clean publication-only directory.
7. Run 'python tools/release_bundle_gate.py --dist <publication-directory>' to reject
   missing platforms, missing x64 CUDA providers, missing sdist, mismatched tags,
   version/metadata disagreement, inconsistent dependencies, and unknown files.
8. Require owner release approval and upload those *same* artifacts in one isolated
   OIDC Trusted Publishing step. Never rebuild between validation and publication.
9. Confirm PyPI exposes every intended wheel, and perform public-index installation
   smoke tests in disposable target-platform environments.

The inventory gate in this initial change does **not** connect to PyPI publishing.
It must be made a non-bypassable prerequisite of a future version-neutral publication
workflow together with the actual test results. In particular, the old
'.github/workflows/publish-alpha.yml' is hard-coded to version 0.2.1 and a Windows-only
release artifact; it must not be reused as-is for 0.2.2.

## Current blockers to address before a stable cross-platform release

- Repair stale '0.2.0.dev0' expectations in the native capability tests and foundation
  verifier **without weakening the underlying ABI/API invariants**.
- Reconcile 'tools/ci_smoke.py' no-dependency expectation with the published 0.2.1
  PyTorch dependency policy.
- Restore green real cross-platform foundation and release-candidate workflows.
- Produce a Linux x86-64 wheel with the *lazy* CUDA provider and qualify it on NVIDIA
  hardware; preserve successful no-GPU CPU imports.
- Add the version-neutral, exact-commit release pipeline and enforce the bundle gate.
- Review macOS Intel as a possible additional required wheel before claiming broad
  Intel+Apple Silicon Mac coverage.

Until these conditions are satisfied, the existence of a CI matrix or a local wheel is
not evidence that a cross-platform GFFX release is ready.
