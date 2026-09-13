# Support status

GFFX `0.2.0.dev0` is pre-release. "Built," "loaded," and "measured internally" describe evidence;
they do not mean a combination is publicly supported. A support claim requires the full
release-candidate matrix, published artifacts, operation conformance, a maintenance policy, and
release evidence defined by the project plan. This document is the user-facing summary and stays
conservative wherever evidence and targets differ.

| Surface | Current state | Classification |
|---|---|---|
| `import gffx` and static capabilities | Verified on bounded Windows/Linux/macOS lanes | Internal foundation evidence |
| CPU `cp310-abi3` packaging on Windows x64, Linux x64/ARM64, macOS ARM64 | Four internal wheels built and smoke-tested | Internal foundation evidence |
| Nineteen native families | Thirteen foundation families, five deformation/skeleton/landmark families and one image-transfer family with decode/encode modes; public compositions are counted separately | Working-source inventory; the new transfer family awaits qualification, not support by counting |
| First-order native derivatives | Foundation derivatives plus all five animation operations; `mesh.build_edge_topology` publishes none by design | Measured fixture cohorts, not all-input/platform proof |
| CUDA provider on Windows x64 | Foundation evidence retained; all five animation operations pass the source7 CPU/CUDA forward/backward byte-comparison cohort | Internal evidence on **one** host only |
| CPU/CUDA bit-identity across hosts, drivers, or architectures | Measured on a single RTX 5090 (driver 595.79, Toolkit 12.9.41, compute capability 12.0) | Target, not support |
| CUDA hardware lane | Current source7/native7 local run passes 51 CTests with no skips; previous CI results retain their own revision | New M1 artifacts are not requalified by historical CI |
| CUDA release artifacts | Not built as release candidates | Target, not support |
| Linux GPU lane | Not provisioned | Target, not support |
| PyTorch Stable-ABI adapter | Current local CPU/CUDA execution uses Windows x64, CPython 3.13 and PyTorch 2.10.0+cu128 | Internal evidence on one combination |
| Camera, scene and shading compositions | Local M1 cohorts recorded separately from the native primitive inventory | Experimental; see authoritative acceptance records |
| Deferred renderer | Small CPU geometry/material/gradient and resource checks pass; full-image and CUDA matrix remain incomplete | Experimental, not complete M1 acceptance |
| Image pipeline | CPU reference/gradient/recovery checks pass; CUDA warp repaired, blur reduction repair under exact-artifact qualification | Experimental; remaining device/resource matrix is open |
| Readiness for starting gffit and Compositron | Generic gffx operations and synthetic integration fixtures; qualification still partial | Consumer implementation, real-model fitting and document transactions are outside gffx M1 and are not claimed |
| CPython 3.10-3.14 × PyTorch 2.10-2.13 full matrix | Adapter *load* covered by hosted lanes; adapter *operation* measured on one combination | Target, not support |
| PyTorch 2.9 | Not supported; the adapter refuses to load below 2.10 | Unsupported |
| JAX CPU/CUDA adapters | Not implemented | Target, not support |
| Spatial acceleration and real-time profiles | Brute-force implementations; selected bounded timings exist, not broad performance guarantees | Sustained real-time and edge profiles remain unqualified |
| Published PyPI distribution | None for `0.2.0.dev0` | Unsupported |

## What the evidence does and does not cover

The configured hosted foundation matrix is CPU-only. Its historical load/packaging results
belong to the revisions and artifacts that actually ran, not today's dirty M1 checkout. The
separate hardware lane must verify required GPU cases by name and reject skipped or missing
cases. Its current local measured floor is 51, replacing the old 40-test inventory; CPU-only
build counts must be read from their own reports, not derived from this number.

The source7 checkpoint has 36 passing installed animation CUDA tests and a matched native7
51-test pass. Native7 names the RTX 5090 Laptop, driver 595.79, Toolkit 12.9.41 and capability
12.0. These are local development-toolkit results, not CUDA 12.8 release-artifact qualification.
Reference-mode comparisons use bytes rather than a tolerance. A single measured cohort does
not prove identity for every input, architecture or transcendental implementation. Earlier
sanitizer reports remain earlier evidence; no fresh sanitizer claim is made for M1 additions.
See [the CUDA hardware lane](CUDA_HARDWARE_LANE.md).

For reproducibility, local build reports are retained under `build/m1-team-20260909` and the
lane-specific `build/animation-20260909`, `build/image-20260909` and
`build/deferred-20260909` directories. Those generated artifacts are not distributed as source.
Devbrain's gffx `PLAN.md` and `NUMBERS.md` record complete hashes and superseded failures.
Source8 includes the targeted blur and scalar-colour binding repairs; implementation is not
automatically a green result on that new artifact.

Some execution boundaries and selected costs have been tested. An expected rejection of
compilation/export is not working export, and a tiny timing is not sustained throughput,
jitter or callback safety. No complete real-time, edge-device or cross-platform operation
matrix is claimed. Higher-order derivatives and compiled paths remain unqualified unless
their exact operation/profile has dedicated evidence.

Public PyPI `0.1.x` artifacts are an inherited prototype and are unsupported by the new contract.
The repository source covered by this document is the `0.2.0.dev0` development tree; it does not change
the identity or support status of the public `0.1.x` artifacts.

See the devbrain support matrices for the normative planned gates.
