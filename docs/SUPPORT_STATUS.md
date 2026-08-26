# Support status

GFFX `0.2.0.dev0` is pre-alpha foundation work. “Built,” “loaded,” and “tested internally” describe
evidence; they do not mean a combination is publicly supported. A support claim requires the full
release-candidate matrix, published artifacts, operation conformance, maintenance policy, and
release evidence defined by the project plan.

| Surface | Current state | Classification |
|---|---|---|
| `import gffx` and static capabilities | Verified on bounded Windows/Linux/macOS internal lanes | Internal foundation evidence |
| CPU `cp310-abi3` packaging on Windows x64, Linux x64/ARM64, macOS ARM64 | Four internal wheels built and smoke-tested | Internal foundation evidence |
| PyTorch Stable-ABI adapter | One Windows/Python wheel loaded under PyTorch 2.10-2.13 | Internal foundation evidence; no operation |
| CUDA provider on Windows x64 | Driver/device diagnostic built locally with Toolkit 12.9.41 | Internal scaffold evidence only |
| CUDA 12.8 Windows/Linux artifacts | Not built as release candidates | Target, not support |
| Linux minimum GPU lane (T4, R570+) | Environment selected but not provisioned | Target, not support |
| CPython 3.10-3.14 × PyTorch 2.10-2.13 full matrix | Four of eighty adapter-load combinations measured | Target, not support |
| JAX CPU/CUDA adapters | Not implemented | Target, not support |
| Graphics and mesh operations | Zero implemented against the new contract | Unsupported |
| Real-time streaming profiles | Contracts defined; operation timings not measured | Target, not support |

No functional CUDA kernel exists in the foundation, and device enumeration is not an operation or
performance result. No differentiability, rendering correctness, coverage, latency, jitter,
sustained-run, edge-device, or operation-export claim has been measured yet.

Public PyPI `0.1.x` artifacts are the inherited prototype and are unsupported by the new contract.
The repository source covered by this document is the `0.2.0.dev0` foundation; it does not change
the identity or support status of the public `0.1.x` artifacts.

See the devbrain support matrices for the normative planned gates. This repository document is the
user-facing summary and must remain conservative when evidence and targets differ.
