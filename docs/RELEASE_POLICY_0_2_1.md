# gffx 0.2.1 Windows release policy

## Scope and runtime policy

Windows x64, ordinary-GIL Python 3.10–3.14. The stable eager CPU geometry,
animation, camera and image operation list remains the bounded0.2.0 list;
scene/material/deferred rendering and CUDA retain experimental scope. M1 stays
complete. No broad platform/GPU, compilation/export, higher-derivative or
performance admission follows from this release.

One cp310-abi3 wheel contains the independent C11 CPU core, lazy Stable-ABI Torch
adapter and Driver-API CUDA provider. It declares `torch>=2.10.0,<=2.14.1`.
Fresh environments select current2.14.1; already installed compatible local
CPU/CUDA variants are preserved. The adapter is still built against CPU2.10
Stable ABI. The exact admitted endpoints are2.10.0 and2.14.1; intermediate
versions are not automatically claimed qualified by metadata compatibility.
No framework or NVIDIA DLL is vendored. Bare import remains lazy.

The default Windows PyPI Torch is CPU-only. Fresh GPU installation adds the
official cu126 index in one command; standard pip has no hardware selector.
Automatic scene setup prefers CUDA only when both framework and provider are
usable. Explicit operations never fall back. The same22-line example is included
in the wheel for `python -m gffx.examples.cornell_box`.

New shadowed CPU composition defaults to512MiB; unshadowed CPU stays64MiB.
Explicit caller bounds still refuse before allocation. Shadows/hard visibility
remain approximate/discrete; direct lighting and ambient fill are not GI.

## Exact validation and upstream state

Require fourteen no-failure/no-selected-skip exact-wheel profiles: five CPU
Python minors plus Python3.14 CPU/GPU oncu126 for each of2.10.0 and2.14.1.
Include the retained323CPU/285GPU operation cohorts and separate scene,
gradient,device,budget,PNG tests and actual512px examples. Fresh default PyPI
dependency resolution/rendering, existing CUDA retention, official CUDA-index
resolution, native12-case analytical checks, upgrade/uninstall, strict metadata,
native imports,PTX andRECORD checks are independent concrete gates.

Torch2.14.1 is the current upstream release at inspection. Retained2.10 is
upstream-unmaintained and affected by GHSA-rrmf-rvhw-rf47 (TorchScript; first fixed
in2.13). gffx has no TorchScript call and compilation/export remains excluded;
this retained compatibility result is not an upstream-security promise.
Publish the advisory and the new default; do not silently pin fresh users to2.10.

## Explicit release sign-off for 0.2.1

No fixed calendar wait. Preserve cumulative costs and failures; one source-bound
matrix can serve as fresh evidence without duplicate label-only executions.
Final evidence must be real and within24hours. A changed artifact is rechecked.
Local `--check` establishes technical preparation, never publication authority.

After separate exact transaction approval, new source CI must pass at the exact
reviewed commit: one builder and ten CPU minor/framework jobs. The publisher
queries actual CI rather than trusting a fabricated future manifest result.
The actual published-release event must come from dependanz on dependanz/gffx,
normal tagv0.2.1 at the reviewed commit with one exact wheel. Ten CPU jobs test
that exact release asset before OIDC upload. `--candidate`/`--check` cannot bypass
the actual published-release event or source CI. All old artifacts/policies and
the indefinitely paused campaign/heartbeat remain preserved.
