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
the provider remains opt-in at runtime. Inclusion is necessary but insufficient: Windows x64 uses real NVIDIA-device
qualification against the candidate wheel as a release gate. Linux x64 uses
the hosted manylinux build, CPU/adapter functional tests and optional CUDA
provider payload checks as release gates; real Linux CUDA operation testing has
been exercised separately under WSL2. Independent native-Linux NVIDIA testing
is additional assurance to be developed after 0.2.2, not a publication blocker.
For every tested path we retain exact source, wheel hashes, device and limitations.

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
5. Qualify the CUDA-enabled Windows release wheel on a real NVIDIA host.
   Preserve the Linux WSL2 CUDA results as supplemental evidence, not native-Linux certification.
   Independent Linux hardware and separate native-Mac hardware testing are future assurance.
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

Runner setup and owner-controlled registration procedures are in
[RELEASE_RUNNERS_0_2_2.md](RELEASE_RUNNERS_0_2_2.md).

The candidate **does not** require an independent native-Linux NVIDIA runner for
0.2.2. It still requires the protected Windows GPU wheel-build/device gate,
the exact Linux x64 CUDA-provider wheel payload and hosted Linux tests, the
complete four-wheel + source archive bundle, and the 100-environment
installed-wheel compatibility matrix. The manually dispatched candidate
produces a revision/run-bound receipt and never publishes to PyPI.
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

- Supervise and qualify the existing protected Windows NVIDIA runner, including
  CUDA 12.x NVCC, CMake, Compute Sanitizer, and sufficient disk space; execute the RC.
  Dedicated native-Linux NVIDIA and native-Mac hardware lanes are post-0.2.2 assurance.
- Document the supported CUDA driver/toolkit compatibility floor conservatively.
  The Linux build uses CUDA 12.9 PTX; the tested WSL2 configuration is evidence,
  not a blanket guarantee for earlier native Linux drivers
- Run and retain every supported Python/PyTorch artifact test, including
  qualification of the declared PyTorch 2.14.1 upper boundary
- Configure and qualify the new version-neutral trusted publisher, retiring
  the hard-coded 0.2.1-only publication workflow after review
- Set release version 0.2.2 and update version-specific metadata and tests;
  confirm public-index installation on all supported systems and Colab
- Maintain hosted macOS ARM64 CPU wheel testing; independent native-Mac hardware
  testing and macOS Intel qualification can follow later. Metal/MPS acceleration
  is separate from NVIDIA CUDA

Until these conditions are satisfied, the existence of a CI matrix or a local wheel is
not evidence that a cross-platform GFFX release is ready.
