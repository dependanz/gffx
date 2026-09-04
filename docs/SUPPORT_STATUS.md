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
| Thirteen graphics, geometry, and mesh operations | Implemented against the contract with acceptance fixtures | Internal evidence; not a public API guarantee |
| Gradients for twelve of the thirteen | Implemented with fixtures; `mesh.build_edge_topology` publishes none by design | Internal evidence |
| CUDA provider on Windows x64 | Twelve of thirteen forwards and all twelve backwards, bitwise identical to the CPU host | Internal evidence on **one** host only |
| CPU/CUDA bit-identity across hosts, drivers, or architectures | Measured on a single RTX 5090 (driver 595.79, Toolkit 12.9.41, compute capability 12.0) | Target, not support |
| CUDA in continuous integration | Hardware lane designed and verified locally; no self-hosted runner registered | Target, not support |
| CUDA release artifacts | Not built as release candidates | Target, not support |
| Linux GPU lane | Not provisioned | Target, not support |
| PyTorch Stable-ABI adapter | Built and loaded, with operations and autograd executed | Internal evidence on one combination |
| CPython 3.10-3.14 × PyTorch 2.10-2.13 full matrix | Adapter *load* covered by hosted lanes; adapter *operation* measured on one combination | Target, not support |
| PyTorch 2.9 | Not supported; the adapter refuses to load below 2.10 | Unsupported |
| JAX CPU/CUDA adapters | Not implemented | Target, not support |
| Spatial acceleration and real-time profiles | Brute-force implementations; operation timings not characterised | Target, not support |
| Published PyPI distribution | None for `0.2.0.dev0` | Unsupported |

## What the evidence does and does not cover

Hosted continuous integration runs 36 blocking lanes nightly across four platforms, five CPython
versions, and three PyTorch lines. Every one of them builds **CPU-only**: the foundation verifier
configures `-DGFFX_ENABLE_CUDA=OFF`, so the hosted inventory is 25 tests where a CUDA-enabled build
registers 40. No hosted lane has ever executed a CUDA kernel.

Every CUDA claim in this repository therefore rests on manual runs on one developer machine. The
kernels are compared to the CPU host with `memcmp` rather than a tolerance, and
`compute-sanitizer --tool memcheck` reports zero errors over both parity executables, but a single
host cannot establish that the agreement holds on another architecture, driver, or toolkit.
`log2f` is not required by IEEE 754 to be correctly rounded, so cross-backend agreement on it is a
measurement rather than a guarantee. See [the CUDA hardware lane](CUDA_HARDWARE_LANE.md).

Hosted lanes verify that the PyTorch adapter *loads*. Executing operations through it, including a
gradient through autograd, has been done on one combination — Windows x64, CPython 3.13, PyTorch
2.10.0+cpu — and not across the matrix.

Rendering correctness, gradient correctness, and CPU/CUDA agreement are covered by acceptance
fixtures. Latency, jitter, sustained-run, edge-device, and operation-export behaviour are not
measured at all.

Public PyPI `0.1.x` artifacts are an inherited prototype and are unsupported by the new contract.
The repository source covered by this document is the `0.2.0.dev0` foundation; it does not change
the identity or support status of the public `0.1.x` artifacts.

See the devbrain support matrices for the normative planned gates.
