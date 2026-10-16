# GFFX

## Simple scene API — 0.2.1

A user scene should declare objects, lights and a camera, then render and save.
The new [22-line Cornell box example](examples/cornell_box.py) imports only gffx;
mesh generation, sampled area lighting, shadow maps and PNG encoding live in the
library. See [simple scenes](docs/SIMPLE_SCENES.md) for the API and its gradient
and direct-lighting boundaries. These additions are new in 0.2.1. Its Python
distribution installs a compatible PyTorch runtime automatically; the native
core and bare import remain independent of the framework.


GFFX is a portable, general-purpose graphics, geometry, and mesh toolkit for Python 3.10+,
differentiable wherever differentiation exists and capable regardless of whether it does. Its
design centers on stable tensor-level semantics across autodiff frameworks, a dependency-light C11
CPU core, isolated accelerator providers, compilation and export, and allocation-controlled
integration into streaming and edge applications.

The `0.2.0` release line defines a bounded Windows x64 support contract.
Its stable admission covers the declared eager CPU geometry, camera, animation
and image API on CPython 3.10-3.14 with separately selected Torch 2.10.0. The wheel
also contains the existing optional CUDA provider under a finite experimental
profile. See [release scope and gates](docs/RELEASE_POLICY_0_2_0.md),
[installation](docs/INSTALLATION.md) and [current release evidence](docs/RELEASE_0_2_0.md).
M1 development readiness is complete; broader platform, renderer and performance
targets keep their own evidence boundaries. The reference campaign is paused
indefinitely.

The core geometry inventory below is one part of the current camera, scene,
animation, shading and image supply. Experimental filtered-renderer qualification
remains a separate, unqualified workstream; its reference campaign is paused.

![The same ground plane sampled without mipmaps, with mip NEAREST, and
trilinear](https://raw.githubusercontent.com/dependanz/gffx/m1-camera/docs/showcase/figures/04-minification.png)

## What works today

The development checkout also includes the [minimal scene API](https://github.com/dependanz/gffx/blob/m1-camera/docs/SCENE.md):
typed objects, transform hierarchies, frame-major mesh packing, attribute/material
bindings and differentiable transforms. This is an eager PyTorch composition over
the existing operations, not a scene editor, file importer or new native renderer.

**Thirteen native geometry operations**, each with a written acceptance contract and fixtures:

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
[`docs/showcase/`](https://github.com/dependanz/gffx/blob/m1-camera/docs/showcase/).

The working camera implementation is documented in [Eager pinhole cameras](https://github.com/dependanz/gffx/blob/m1-camera/docs/CAMERAS.md):
`gffx.camera.Camera` projects points, reconstructs positions from depth, and batches cameras
using the existing transforms. Its optional `in_front_only=True` filter keeps behind-camera
rejection separate from mathematical projection validity and visibility. It adds no native
primitive and makes no compilation/export or real-time performance claim.

PyTorch also exposes compositional [classic](https://github.com/dependanz/gffx/blob/m1-camera/docs/CLASSIC_MATERIAL.md) and
[principled/custom materials](https://github.com/dependanz/gffx/blob/m1-camera/docs/PRINCIPLED_MATERIAL.md), tagged light integration,
and explicit tensor colour sources. These are not additional native primitives.
Principled shading has separately named, finite-profile float64 budget bundles for
[CPU](https://github.com/dependanz/gffx/blob/m1-camera/docs/PRINCIPLED_BUDGET_V1.md) and [CUDA](https://github.com/dependanz/gffx/blob/m1-camera/docs/PRINCIPLED_CUDA_QUALIFICATION.md);
the table is shipped package data, not a download. Input/layout, energy and gradient
tests do not imply Cycles fidelity, compiled/exported shading, or real-time performance.
See the [shading implementation and evidence map](https://github.com/dependanz/gffx/blob/m1-camera/docs/SHADING_M1_STATUS.md) for
what is locally tested and what remains outside this slice.

## What does not work yet

- **The PyTorch adapter requires PyTorch 2.10 or newer.** On an older PyTorch it refuses to load
  and the Python operation surface is inert. The C ABI remains available.
- **Select the current release deliberately.** The public `0.1.x` releases are the inherited
  prototype with unrelated APIs. The `0.2.x` line packages the native implementation; the release policy and
  exact-artifact evidence define stable and experimental features.
- **GPU evidence is local and artifact-specific.** Historical laptop/RTX 5090
  and PC/RTX 3080 cohorts retain separate identities. This release scope does not claim a
  complete GPU/driver/framework matrix or continuous hosted hardware qualification.
  The [hardware lane](https://github.com/dependanz/gffx/blob/m1-camera/docs/CUDA_HARDWARE_LANE.md)
  retains its independent requirements.
- **No JAX adapter.** Compilation, export and preallocated streaming retain their
  own operation-specific development profiles and limits.
- **No general real-time claim.** KNN and closest-point queries remain brute force;
  the rasterizer has conservative block rejection. Exact historical timings do not
  establish current-artifact throughput or sustained latency.
- **Shading is still a bounded development implementation.** Lobes, built-in lights and classic/principled
    material compositions are available in the development checkout. The exact bundled
    principled loss table has separate CPU/CUDA float64 profiles with a 32-entry standalone query limit.
    Material calls separately support 1024 point/light pairs (8 forward or 4 backward
    points), under a 64 MiB planning ceiling. Fixed image/reference fixtures do not
    qualify arbitrary scenes, sustained performance or compilation/export. See
    [material usage and limits](https://github.com/dependanz/gffx/blob/m1-camera/docs/PRINCIPLED_MATERIAL.md). Tangent frames, bump mapping
    and colour management remain later work.
- **No ray tracing or BVH traversal, remeshing, mesh booleans, volumetric rendering, non-triangle
  faces, or image decoding.** `render.texture` samples a tensor the caller already holds; it opens
  no file.
- **Support claims are per combination and measured, not inherited.** See
  [measured evidence versus support targets](https://github.com/dependanz/gffx/blob/m1-camera/docs/SUPPORT_STATUS.md) before depending on a
  platform, Python version, or accelerator.

## What it looks like

The render chain stage by stage, on a procedurally generated sphere. Every pixel comes from a GFFX
operation:

![Coverage, interpolated UVs, sampled albedo, and a Lambert term on the interpolated
normal](https://raw.githubusercontent.com/dependanz/gffx/m1-camera/docs/showcase/figures/06-full-chain.png)

And the property that makes it a differentiable renderer rather than a renderer — gradient descent
on vertex positions, driven entirely by the library's own backward passes, until a triangle matches
its target:

![Intersection over union rising from 0.53 to 0.95 over forty
steps](https://raw.githubusercontent.com/dependanz/gffx/m1-camera/docs/showcase/figures/07-gradient-descent.png)

More figures, with what each one demonstrates, in [`docs/showcase/`](https://github.com/dependanz/gffx/blob/m1-camera/docs/showcase/).

## Install and inspect

The 0.2.1 distribution is one Windows x64 wheel with the native CPU core and
lazy CUDA provider. It adds the short scene API and automatically installs the
bounded Torch runtime (fresh default 2.14.1; existing compatible CUDA is retained):

```powershell
python -m pip install gffx
python -m gffx.examples.cornell_box
```

The example is shipped in the wheel. Its `auto` device selects usable CUDA,
otherwise CPU. On Windows fresh default PyPI Torch is CPU-only; a GPU environment
can install in one command using the official CUDA index:

```powershell
python -m pip install gffx --extra-index-url https://download.pytorch.org/whl/cu126
```

An existing compatible CUDA Torch is retained. Pip cannot detect GPU hardware
to choose a dependency index. GPU use needs a supported NVIDIA driver/device;
no compiler/toolkit is required. See [installation](docs/INSTALLATION.md) and
[the exact release boundary](docs/RELEASE_POLICY_0_2_1.md). CUDA/scene rendering
retain their experimental scope. Older published versions keep their own policy.

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

- [Installation and optional components](https://github.com/dependanz/gffx/blob/m1-camera/docs/INSTALLATION.md)
- [Source and native builds](https://github.com/dependanz/gffx/blob/m1-camera/docs/BUILDING.md)
- [Dependency and vendoring policy](https://github.com/dependanz/gffx/blob/m1-camera/docs/DEPENDENCIES.md)
- [Measured evidence versus support targets](https://github.com/dependanz/gffx/blob/m1-camera/docs/SUPPORT_STATUS.md)
- [Isolated CUDA provider recipes](https://github.com/dependanz/gffx/blob/m1-camera/docs/CUDA_PLUGIN_BUILD.md)
- [The CUDA hardware test lane](https://github.com/dependanz/gffx/blob/m1-camera/docs/CUDA_HARDWARE_LANE.md)
- [Capability figures and how they are generated](https://github.com/dependanz/gffx/blob/m1-camera/docs/showcase/)
- [Testing-dashboard learning pages and dependency-free mathematics](https://github.com/dependanz/gffx/blob/m1-camera/docs/LEARNING_PAGES.md)

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

GFFX is licensed under the [MIT License](https://github.com/dependanz/gffx/blob/m1-camera/LICENSE). No third-party source is vendored, and every
figure here is generated from GFFX's own primitives with procedurally constructed inputs, so no
external asset or dataset is redistributed.
