# GFFX

GFFX is a portable, general-purpose graphics, geometry, and mesh toolkit for Python 3.10+,
differentiable wherever differentiation exists and capable regardless of whether it does. Its
design centers on stable tensor-level semantics across autodiff frameworks, a dependency-light C11
CPU core, isolated accelerator providers, compilation and export, and allocation-controlled
integration into streaming and edge applications.

The project is pre-release. Thirteen operations are implemented and carry acceptance fixtures; the
documentation, packaging, and release work is not done. Read [What works today](#what-works-today)
and [What does not work yet](#what-does-not-work-yet) together — the second is as load-bearing as
the first.

![The same ground plane sampled without mipmaps, with mip NEAREST, and
trilinear](docs/showcase/figures/04-minification.png)

## What works today

**Thirteen operations**, each with a written acceptance contract and fixtures:

| Group | Operations |
|---|---|
| mesh | `face_geometry`, `vertex_normals`, `gather_faces`, `build_edge_topology`, `sample_surface` |
| transforms | `transform_points`, `perspective_divide` |
| points | `knn`, `closest_point_on_mesh` |
| render | `rasterize`, `interpolate`, `texture_pyramid`, `texture` |

**Gradients** for twelve of the thirteen. `mesh.build_edge_topology` publishes none, because every
one of its outputs is integer topology and there is nothing to differentiate. Differentiability is
a per-operation property rather than a condition of membership: an operation declares its gradient
support by publishing a `_backward` entry point or by not publishing one, so the absence is visible
when you bind rather than when you first ask for a gradient.

**A CUDA provider** covering twelve of the thirteen forwards and all twelve backwards, loaded as an
isolated optional plugin. `mesh.build_edge_topology` has no CUDA forward.

**Bitwise CPU/CUDA agreement**, not merely closeness, for every operation that can promise it. The
device build passes `-fmad=false` and each kernel mirrors the host operation for operation rather
than reassociating for speed, so results are compared with `memcmp` rather than a tolerance. Where
a gradient scatters onto shared geometry the default path reproduces the host's accumulation order
exactly; a faster atomic path exists but is reached only when the caller sets
`GFFX_EXECUTION_ALLOW_NONDETERMINISTIC`, and it makes no bit-identity claim.

**A framework-neutral C11 ABI**, usable without Python or any autodiff framework. The figures in
this README were produced by calling it from Python with `ctypes` and nothing else — see
[`docs/showcase/`](docs/showcase/).

## What does not work yet

- **The PyTorch adapter requires PyTorch 2.10 or newer.** On an older PyTorch it refuses to load
  and the Python operation surface is inert. The C ABI remains available.
- **`pip install gffx` does not install this.** Public PyPI releases through `0.1.4` belong to an
  inherited prototype with unrelated APIs. This is `0.2.0.dev0`, available only from a repository
  checkout or an explicitly supplied artifact.
- **No published CUDA artifact.** The provider is built from source and needs the CUDA toolkit.
- **CUDA agreement is measured on one machine, not in continuous integration.** No hosted runner
  has a GPU, so every device claim above comes from manual runs on a single RTX 5090. The
  [hardware lane](docs/CUDA_HARDWARE_LANE.md) exists to close this and needs a registered
  self-hosted runner before it does.
- **No JAX adapter**, and no public `torch.compile`, export, serialization, or streaming operation
  surface.
- **No spatial acceleration structure.** `points.knn` and `points.closest_point_on_mesh` are brute
  force, and the rasterizer costs roughly 2.4 seconds per 1000x1000 frame on CUDA. Correct and
  reproducible; not real-time.
- **No materials, lights, BRDF lobes, tangent frames, bump mapping, or colour management.** These
  are staged for a later phase rather than excluded from the long-term scope.
- **No ray tracing or BVH traversal, remeshing, mesh booleans, volumetric rendering, non-triangle
  faces, or image decoding.** `render.texture` samples a tensor the caller already holds; it opens
  no file.
- **Support claims are per combination and measured, not inherited.** See
  [measured evidence versus support targets](docs/SUPPORT_STATUS.md) before depending on a
  platform, Python version, or accelerator.

## What it looks like

The render chain stage by stage, on a procedurally generated sphere. Every pixel comes from a GFFX
operation:

![Coverage, interpolated UVs, sampled albedo, and a Lambert term on the interpolated
normal](docs/showcase/figures/06-full-chain.png)

And the property that makes it a differentiable renderer rather than a renderer — gradient descent
on vertex positions, driven entirely by the library's own backward passes, until a triangle matches
its target:

![Intersection over union rising from 0.53 to 0.95 over forty
steps](docs/showcase/figures/07-gradient-descent.png)

More figures, with what each one demonstrates, in [`docs/showcase/`](docs/showcase/).

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
- [The CUDA hardware test lane](docs/CUDA_HARDWARE_LANE.md)
- [Capability figures and how they are generated](docs/showcase/)

## Source boundaries

- `include/gffx/`: public framework-neutral C11 ABI.
- `native/core/`: dependency-light runtime and the independent CPU operations.
- `native/cuda/`: private optional CUDA provider; no semantics belong here exclusively.
- `adapters/`: CPython and autodiff-framework loading/registration glue.
- `src/gffx/`: stable dependency-light Python namespace.
- `tests/`: ABI, packaging, import, framework-loading, accelerator-isolation, and operation
  acceptance contracts.
- `docs/showcase/`: the figure generators and the figures above.

The pre-foundation source remains recoverable in Git history and on
`codex/archive-pre-phase1-20260821`. Prototype behavior is unsupported and receives no
compatibility credit toward the new operation contracts.

GFFX is licensed under the [MIT License](LICENSE). No third-party source is vendored, and every
figure here is generated from GFFX's own primitives with procedurally constructed inputs, so no
external asset or dataset is redistributed.
