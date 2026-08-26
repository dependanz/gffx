# GFFX

GFFX is a portable, general-purpose differentiable graphics, geometry, and mesh toolkit for
Python 3.10+. Its design centers on stable tensor-level semantics across autodiff frameworks, a
dependency-light C11 CPU core, isolated accelerator providers, compilation and export, and
allocation-controlled integration into streaming and edge applications.

The project is pre-alpha foundation work. No public graphics or geometry operation is implemented
or advertised yet. The first planned operation is `mesh.face_geometry`, where “face” means a
triangular mesh face—not a talking-face specialization.

## What works today

- `import gffx` is lazy and has zero mandatory third-party Python runtime dependencies.
- `gffx.capabilities()` reports static package, host, ABI, and CPU state without probing a GPU.
- Native ABI v1.0 provides checked tensor, execution-context, workspace, diagnostic, and capability
  structures through six public C11 exports.
- `gffx.torch` is a lazy PyTorch 2.10+ Stable-ABI loading scaffold. It registers only a private
  foundation probe and exposes no operation.
- `gffx.cuda.capabilities()` is an explicit setup-time diagnostic that may load the isolated CUDA
  provider and NVIDIA driver. It is not a frame-loop call and exposes no CUDA kernel.
- Internal `cp310-abi3` CPU wheels have been built for Windows x86-64, Linux x86-64, Linux ARM64,
  and macOS ARM64. This is foundation evidence, not a public support or release claim.

## What does not work yet

- No v0.1 graphics, geometry, mesh, proximity, sampling, rasterization, loss, or blending operation.
- No public autograd, `torch.compile`, export, serialization, JAX, or streaming operation surface.
- No functional CUDA kernel or CUDA 12.8 release artifact.
- No supported migration path for the prototype `gffx` 0.1.x Python APIs.

Public PyPI releases through `0.1.4` belong to the inherited prototype. Installing `gffx` from
PyPI today does **not** install this `0.2.0.dev0` foundation. The current foundation is available
from this branch or from explicitly supplied internal artifacts only.

## Install and inspect

Use the detailed [installation guide](docs/INSTALLATION.md) before installing. A default source
build is CPU-only:

```powershell
python -m pip wheel . --no-deps --wheel-dir dist/local
python -m pip install --force-reinstall C:/path/to/generated-wheel.whl
```

The stable entry point is always:

```python
import gffx

print(gffx.__version__)
print(gffx.capabilities())       # static; never probes a GPU

# Explicit diagnostic only; may load the optional provider and GPU driver.
print(gffx.cuda.capabilities())
```

Do not call the full CUDA capability probe from a real-time frame or audio callback.

## Documentation

- [Installation and optional components](docs/INSTALLATION.md)
- [Source and native builds](docs/BUILDING.md)
- [Dependency and vendoring policy](docs/DEPENDENCIES.md)
- [Measured evidence versus support targets](docs/SUPPORT_STATUS.md)
- [Isolated CUDA provider recipes](docs/CUDA_PLUGIN_BUILD.md)

## Source boundaries

- `include/gffx/`: public framework-neutral C11 ABI.
- `native/core/`: dependency-light runtime and future independent CPU operations.
- `native/cuda/`: private optional CUDA provider; no semantics belong here exclusively.
- `adapters/`: CPython and autodiff-framework loading/registration glue.
- `src/gffx/`: stable dependency-light Python namespace.
- `tests/`: ABI, packaging, import, framework-loading, and accelerator-isolation contracts.

The pre-foundation source remains recoverable in Git history and on
`codex/archive-pre-phase1-20260821`. Prototype behavior is unsupported and receives no
compatibility credit toward the new operation contracts.

GFFX is licensed under the [MIT License](LICENSE). No third-party source is vendored in the current
foundation.
