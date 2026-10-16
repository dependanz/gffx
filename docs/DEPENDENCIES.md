# Dependency and provenance policy

The0.2.0 scoped release retains zero mandatoryPythondependencies, the CPU-built
optional Torch2.10 adapter and lazy Driver-API provider. Test-only dependencies
are pytest9.0.3 (including its built-in subtest support); no NumPy or private fixture is required
by the portable release cohort. Exact newly installed dependency inventories are
recorded with validation; build tools do not become runtime dependencies. See
[the current contract](RELEASE_POLICY_0_2_0.md).


The GPU-capable `0.2.0a2` experimental Windows candidate adds the existing lazy
CUDA provider to the CPU core/optional adapter. See [its exact profile](ALPHA_0_2_0A2.md)
and [GPU build recipe](CUDA_PLUGIN_BUILD.md). Published CPU-only `0.2.0a1` and all
older artifact-specific evidence retain their own scope. Publication availability
is separate from preparation; the reference campaign stays paused indefinitely.


The base `gffx` distribution has zero mandatory third-party Python runtime dependencies. Its
wheel contains the Python namespace, CPython 3.10 Limited-API bridge and framework-neutral
native core. The Windows0.2.0a1 CPU profile also includes an optional PyTorch adapter;
it loads only after explicit framework activation and does not make Torch mandatory.

## Declared source-development groups

PEP 735 dependency groups do not enter wheel `Requires-Dist` metadata.

| Group | Current declaration | Ownership |
|---|---|---|
| `test` | pytest 9.0.3; tomli 2.2.1 only on Python 3.10 | source tests |
| `packaging` | build 1.3.0; cibuildwheel 4.2.0; Windows delvewheel 1.13.2; twine 7.0.0 | local/CI artifact production |
| `development` | includes `test` and `packaging` | contributor convenience |
| `framework` | empty | frameworks are selected explicitly outside GFFX metadata |
| `accelerator` | empty | drivers/toolkits are system prerequisites, not pip dependencies |
| `visualization` | empty | no visualization integration is implemented |
| `testing-dashboard` | `coverage==7.10.7` | optional Python coverage collection for the repository-only testing app; never a wheel runtime dependency; [package provenance](https://pypi.org/project/coverage/7.10.7/) |
| `format` | empty | no mesh/scene file-format integration is implemented |
| `example` | empty | no example-only dependency is admitted yet |
| `benchmark` | empty | benchmark tooling waits for an executable operation |

The build backend is separately pinned to `scikit-build-core==1.0.3`; it supplies CMake 4.4.2 and
Ninja 1.13.0 as build-only tools. These are not runtime or public feature dependencies.

## Optional external prerequisites

- PyTorch is user-selected because its CPU/CUDA variants and package indexes differ by deployment.
  The adapter targets the PyTorch2.10 stable ABI. Current alpha validation profiles and
  exact results are listed in [the GPU alpha profile](ALPHA_0_2_0A2.md) and
  [the preserved CPU alpha profile](ALPHA_0_2_0A1.md); later frameworks
  do not silently gain a tested-support claim. The CPU build uses CPU-only Torch, and
  the wheel must not bundle framework DLLs or import torch_cuda.dll.
- The CUDA provider uses the NVIDIA Driver API. This experimental a2 artifact is built
  with CUDA Toolkit 12.6.20 and declares Windows driver 560.76 or newer; only
  driver 610.88 is tested. The toolkit is not required to install the wheel.
  Broader CUDA build/support targets retain their separate contracts. No CUDA
  Runtime or NVIDIA driver DLL is bundled.
- NumPy, JAX, visualization libraries, mesh I/O libraries, and benchmark packages are not required
  or imported by the current base package.

## Vendoring and licence provenance

GFFX source is distributed under the repository's MIT licence. No third-party source is vendored
in the current foundation, and no generated third-party binary is committed. A future vendored
component must record its upstream project, exact version or revision, source URL, licence,
unmodified or patched state, and reproduction instructions before it enters a release artifact.

Generated wheels, native binaries, caches, and environments are build evidence—not source—and stay
outside Git and devbrain project memory.

## Binary runtime in the Windows alpha

Delvewheel 1.13.2 bundles one Microsoft C++ runtime as
`gffx.libs/msvcp140-7c26614e1d733892c2deac7e245ce115.dll`. This is a Microsoft
binary with its own license terms, rather than gffx source covered by the MIT
license. Other Windows/CPython runtime DLLs remain system prerequisites.
See [Microsoft runtime redistribution documentation](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170).
Torch and CUDA DLLs remain external and are not bundled.
