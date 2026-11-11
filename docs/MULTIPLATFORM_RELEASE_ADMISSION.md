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

## Release-candidate progress and remaining gates (November 11, 2026)

The exact installed CPU/PyTorch wheels have already passed on Windows x64,
Linux x64, Linux ARM64 and macOS ARM64, including Cornell-box rendering,
display/PNG pixel parity and uninstall. The Linux CUDA wheel pilot has passed
NVCC compilation, auditwheel repair, ABI inspection and provider-payload checks.
The pilot does **not** establish real NVIDIA device correctness.

The RC workflow now selects an **exact 40-character source commit SHA**, rather
than a moving branch. The Windows x64 CUDA wheel is built on the approved
self-hosted NVIDIA runner, behind the 'cuda-hardware' protection environment,
and must pass native device tests and compute-sanitizer before upload. Linux
x64 uses a CUDA 12.9 manylinux 2.28 build image pinned to its validated
SHA-256 digest; ARM64 and macOS CPU builds remain GPU-independent.

The candidate now **fails closed** until a protected Linux x64 NVIDIA runner
installs its exact candidate wheel, loads the bundled CUDA provider and passes
the native GPU test inventory and sanitizer checks. The RC then requires the
complete four-wheel + source archive bundle and the compatibility test matrix.
There is no PyPI publishing step in this workflow.

## Owner-controlled publication workflow

The proposed publish-multiplatform.yml workflow is owner-only and manually
triggered with a stable tag and RC run ID. Its unprivileged preflight verifies
the matching version and source SHA, a fully successful RC run, the SHA-256
receipt for all candidate artifacts, and Windows/Linux NVIDIA device reports.

A separate pypi-environment-gated OIDC job uploads only those exact approved
artifacts, without checking out or executing candidate source code. Deployment
still requires protecting the pypi environment with required owner review,
registering this new workflow as a PyPI Trusted Publisher, and protecting main
and release tags. Nothing has been uploaded or approved by this PR.

Remaining work before releasing 0.2.2:

- Register/provision the protected Windows and Linux CUDA runners, including
  drivers, NVCC, CMake and compute-sanitizer; execute the CUDA-inclusive RC
- Verify the claimed CUDA driver/toolkit compatibility floor. The Linux pilot
  uses a CUDA 12.9 toolkit and does not establish compatibility with earlier
  drivers solely because GFFX accepts toolkit 12.0 for local builds
- Run and retain every supported Python/PyTorch artifact test, including
  qualification of the declared PyTorch 2.14.1 upper boundary
- Configure and qualify the new version-neutral trusted publisher, retiring
  the hard-coded 0.2.1-only publication workflow after review
- Set release version 0.2.2 and update version-specific metadata and tests;
  confirm public-index installation on all supported systems and Colab
- Qualify macOS Intel separately before advertising it as supported; Metal/MPS
  implementation is independent of NVIDIA CUDA

Until these conditions are satisfied, the existence of a CI matrix or a local wheel is
not evidence that a cross-platform GFFX release is ready.
