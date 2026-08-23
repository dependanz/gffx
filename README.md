# GFFX

GFFX is a portable differentiable-graphics and mesh-operations toolkit for Python 3.10+.
It is being designed around stable tensor semantics, a dependency-light CPU implementation,
optional accelerated backends, compilation and export, and real-time or edge deployment.

The project will support multiple autodiff frameworks. PyTorch CPU and CUDA are the first
delivery targets; JAX follows after the first useful PyTorch slice. Framework adapters are kept
thin so they do not define geometry semantics or force framework dependencies into the base
package.

## Development status

The repository is currently establishing its `0.2` package and native-runtime foundation. No
graphics or geometry operation from the new contract is implemented or advertised yet. The first
planned end-to-end operation is `mesh.face_geometry`, where "face" means a triangular mesh face.

The pre-foundation prototype remains available in repository history and on the dedicated
`codex/archive-pre-phase1-20260821` branch.

## Source boundaries

- `include/gffx/` will contain the framework-neutral public C11 ABI.
- `native/core/` owns the dependency-light runtime and CPU implementations.
- `native/cuda/` owns the optional CUDA plugin boundary.
- `adapters/` contains loading and framework-registration glue, not geometry semantics.
- `src/gffx/` contains the stable, dependency-light Python namespace.
- `tests/` is organized by ABI, packaging, Python, and framework contract.

Build, installation, compatibility, and operation documentation will be added as their release
gates are implemented and verified.
