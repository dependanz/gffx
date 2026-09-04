# Capability figures

Seven figures showing what GFFX does, and the two scripts that generate them.

Every pixel in every figure comes from a GFFX operation. PIL is used only to arrange finished
images and draw captions; a figure whose content came from the plotting library would showcase the
plotting library. All inputs are constructed procedurally, so no external asset or dataset is
redistributed here and every figure is freely publishable.

## Figures

| File | Operation | What it demonstrates |
|---|---|---|
| `01-filtering.png` | `render.texture` | `NEAREST` against `BILINEAR` under heavy magnification |
| `02-wrap-modes.png` | `render.texture` | all four wrap modes over coordinates well outside `[0,1]` on both axes, so every mode has to resolve rather than only the clamped one. A marked edge row makes mirroring distinguishable from repetition |
| `03-mip-pyramid.png` | `render.texture_pyramid` | the level chain at native size. A fine checkerboard survives level 1, blurs at level 2, and collapses to the mean once the level can no longer represent its frequency |
| `04-minification.png` | `render.texture` | a ground plane receding to the horizon, sampled three ways: level 0 only, mip `NEAREST`, and trilinear. The first aliases severely and the other two do not. This is the case mipmapping exists for, and the three panels share identical geometry and coordinates, so every difference between them is the sampler's |
| `05-lod-selection.png` | `render.texture` | the level GFFX selects, drawn directly by filling each level of a probe pyramid with its own index, so the choice is visible rather than inferred from image quality |
| `06-full-chain.png` | `transforms.transform_points`, `perspective_divide`, `render.rasterize`, `render.interpolate`, `render.texture` | a textured sphere through five operations, shown at each stage. Only the camera matrix and the Lambert dot product are computed outside GFFX |
| `07-gradient-descent.png` | `render.rasterize`, `render.interpolate` and their backwards | gradient descent on vertex positions until a triangle matches a target, intersection over union rising from 0.53 to 0.95 in forty steps |

Figure 7 is the one that answers what GFFX is. Every other figure could have been produced by a
competent renderer; that one shows the renderer has a derivative and that the derivative is usable.

## Regenerating

```text
python make_texture_figures.py
python make_render_figures.py
```

Both need `gffx_core` and look for it beside this file, then in `build/cuda/Debug` and
`build/phase4-red-1/Debug`. They report where they looked if they cannot find one. A copy of the
library placed here is deliberately untracked.

## How they reach the library

Through the shipped C ABI, called with `ctypes`, in `gffx_ctypes.py`.

That was not the first choice — the PyTorch adapter requires PyTorch 2.10 and the machine these
were made on has 2.9.1, so the adapter refuses to load. It turned out to be the better
demonstration: it shows the library used with no framework in the loop at all, which is the
portability claim the project actually makes, and it exercises the same entry points a C caller
would.

`gffx_ctypes.py` is deliberately not a general binding. It covers the operations the figures need
and validates nothing the library already validates, because a wrapper that re-checked arguments
would be asserting its own opinion of the contract rather than exercising the library's.

## Three defects these figures caught

Drawing something exercises a library differently from testing it, and each of these was invisible
until a figure was made or a number measured.

Figure 5 first sampled a three-channel pyramid while declaring one channel. `render.texture`
rejected it through the cross-check that compares `level_offsets[1] - level_offsets[0]` against the
extents and channel count — the only guard available against a caller passing a pyramid built for a
different texture.

Figure 6 rendered a sphere whose triangle winding was reversed. Under `CULL_BACK` that keeps the
far hemisphere rather than the near one, and a sphere's silhouette is identical either way, so the
picture looked entirely correct. It was wrong only once a normal was used: measured rather than
eyeballed, the Lambert term was negative on 89 percent of visible pixels. A figure that looks right
is not evidence that it is right.

Figure 7 was first attempted through an interpolated constant attribute under hard rasterization
and produced gradients of order 1e-14. That is not a defect: with `blur_radius_px` at zero the
interior value does not change as a vertex moves and coverage changes discontinuously, so there is
no derivative to find. `signed_distance` is the continuous quantity and carries a usable derivative
even at zero blur. The figure became a descent rather than an arrow diagram for a separate reason:
arrows show only that a gradient is nonzero, while running the optimisation shows whether it is
useful.

## Not yet covered

- Soft rasterization against hard, sweeping `blur_radius_px` upward, which is the pair the scope
  calls an exact form and its differentiable relaxation.
- `points.knn` and `points.closest_point_on_mesh`, drawn as correspondence lines against a mesh.
- A CPU against CUDA panel showing the difference image is exactly zero rather than merely small.
