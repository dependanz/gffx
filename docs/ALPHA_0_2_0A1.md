# Experimental Windows CPU alpha: 0.2.0a1

This prerelease provides the framework-neutral CPU core, CPython stable-ABI
bridge and optional PyTorch adapter in one Windows x64 wheel. The base import
remains framework-free. PyTorch is installed and selected separately. No Torch
or CUDA DLL is vendored, and no gffx CUDA provider is included.

The release build uses CPU-only PyTorch 2.10.0. The build guard rejects a
CUDA-enabled framework environment; the artifact audit rejects unwanted CUDA
imports or bundled framework DLLs. CPU operations are then checked with both
CPU-only and CUDA-enabled PyTorch. Build-time choice does not require users to
replace an existing CUDA-enabled framework installation.

## Candidate validation profiles

These results use the same repaired candidate with SHA-256
`d132d7d2c852feb61eeddfd0eb99082ab7f9ae52f5913220e064d227e0267042`
(2,036,559 bytes). These are pre-publication validation results; package-index availability is
verified separately by the release transaction.

| Platform | CPython | PyTorch | Execution | Candidate result |
|---|---|---|---|---|
| Windows x64 | 3.10.11 | 2.10.0+cpu | CPU graphics/gradients | 232 passed, 1 skipped, 10 deselected; 9 subtests passed |
| Windows x64 | 3.14.6 | 2.10.0+cpu | CPU graphics/gradients | 232 passed, 1 skipped, 10 deselected; 9 subtests passed |
| Windows x64 | 3.14.6 | 2.10.0+cu126 | CPU graphics/gradients | 233 passed, 10 deselected; 9 subtests passed |

Fresh isolated environments install the exact wheel and first verify its base
import and native capabilities with no PyTorch installed. The CPU-only profiles
then install CPU Torch; the CUDA-enabled profile selects the existing framework
distribution read-only through a path entry, with the new wheel taking precedence.
Import-path and framework-version assertions pass, as does `pip check` in each
profile. Tests use one CPU thread, pytest 9.1.1 and the frozen gffit 0.0.14 consumer.
The skipped case needs a CUDA device to verify mixed-device rejection.

Ten cases are explicitly deselected: CUDA-provider cases outside this artifact
and the separately retained filtered opaque-mask exact-equality case. A separate
probe on this alpha reproduces the latter: maximum mask error
`1.1102230246251565e-16`, gradient `0.24999999999999994` versus exact `0.25`.
Neither its numerical implementation nor its strict test was changed. The scoped
pass count does not claim that test passes. One test warning about scalar
conversion of a gradient tensor is retained.

The `cp310-abi3` tag expresses the CPython ABI floor, not a complete support
matrix. Only recorded exact combinations gain evidence. Other Python/framework
minors, free-threaded Python, PyPy, Linux/macOS, and GPU execution retain their
own experimental/unverified status. The broader four-platform release-candidate
matrix remains separate from this first scoped alpha.

## Evidence and limits

Checks cover clean wheel installation, lazy base import without PyTorch, static
capabilities, native dependencies, CPU public operation/gradient fixtures and
small gffit integration/checkpoint controls. They do not rerun photograph fits,
licensed models, large-scale qualification or the reference campaign.

The earlier dev0 candidate had an unwanted `torch_cuda.dll` dependency and a
retained 4x4 opaque-mask exact-equality discrepancy of 1.1102230246251565e-16.
Those original artifacts and failures remain historical evidence; current alpha
results identify their own exact artifact. The filtered renderer remains
independently unqualified and its campaign is paused indefinitely.

M1 remains complete for the adopted development-readiness scope. The B1
photograph likeness remains unacceptable. This experimental package does not
turn those scientific limitations into an acceptance claim.

## Reproduction and publication

See [BUILDING.md](BUILDING.md) for the guarded CPU build and
[INSTALLATION.md](INSTALLATION.md) for the user's installation steps. The build
report records the selected interpreter/framework, commands, native DLL imports,
artifact SHA-256 and wheel RECORD verification. Public release notes identify
the final tested artifact; generated logs, environments and binaries stay outside
source Git. PyPI publication requires a separate final approved transaction.
[Publishing procedure](PUBLISHING_ALPHA.md) describes the exact-wheel verifier,
draft prerelease staging and scoped Trusted Publisher configuration. Strict Twine
metadata/readme validation passes; the verifier passes its approved-artifact case
and rejects wrong tags, missing/extra artifacts, altered bytes and source-version
mismatches.
