# Deferred rendering: M1 implementation in progress

The deferred renderer connects public scenes, cameras, native rasterization and
interpolation to the existing material and light functions. This document records
the intended interface; current acceptance evidence remains in the project records.

```python
import gffx
result = gffx.render(scene, camera, outputs=("colour", "mask", "depth", "normal"))
```

The bound equivalent is `gffx.torch.render`. Both remain namespaces: existing
`render.rasterize`, `render.interpolate`, `render.texture` and `render.texture_pyramid`
remain available. `render.mesh` constructs a scene and uses the same integrator.

Images are batched `[B,H,W,C]` floating tensors on the geometry device. With no
background, colour is linear foreground-premultiplied RGB and mask is coverage.
An explicit RGB background produces composited RGB without changing foreground
coverage. There is no gamma conversion or display clamp. Depth is NDC depth with
infinite background; the camera supplies explicit conversion to metric depth.

Missing normals default to face normals, preserving hard cube edges. Authored
corner normals allow intentional smoothing. Perspective-correct interpolation
keeps sloping surfaces and texture seams coherent. TextureSource binds an existing
pyramid; no asset loading or hidden pyramid rebuilding occurs during rendering.

`supersample=2` renders colour and coverage at twice each dimension and applies
the image module's fixed box reconstruction. Geometric channels are separately
sampled at final pixel centres: averaging face IDs or unrelated depths would be
meaningless. `soft>0` is a front-fragment signed-distance approximation, with
finite local derivatives and explicit cutoff/internal-edge limitations. It is
not transparency, seamless multi-triangle silhouette union or physical filtering.

The original implementation refused more than four gradient-bearing Principled
points. The PC renderer now recomputes ordered four-point groups and batches up
to128 points/1024 light pairs through a separate native operation, inside its
existing material workspace reserve. Exact values, shared first gradients,
independent directional differences and retained/repeated storage are qualified
on the groups35 installed candidate. Public material8/4-point and standalone
32-owner query limits remain unchanged. See `DEFERRED_RECOMPUTE.md` for the
explicit reduction order and preserved failed candidates.
Workspace preflight is a conservative tensor estimate, not a measured process
memory guarantee. Numerical, device, gradient, full-image and consumer acceptance
remain separate gates. No compilation/export or real-time profile is admitted.
The expanded pre-flatten built-in planner and separate 8x8 CPU forward/backward
profiles are green; further estimate-reuse refinements remain pending tests.
Arbitrary trusted callback allocation bounds remain an explicit open admission
question. See `DEFERRED_WORKLOAD.md` before running a larger workload.

The independent small tests are in `tests/python/test_deferred.py`; their
dashboard explanations distinguish analytic checks from visual demonstrations.
`tests/python/test_deferred_recompute.py` separately compares recomputation with
the existing public material, checks all fitted first partials and inspects saved
input storage. That comparison is implementation equivalence, not a new physical
oracle. See `DEFERRED_RECOMPUTE.md` for the explicit graph and ownership boundary.
`tests/python/test_deferred_integration.py` covers manual native-chain equivalence,
independent inverse-square images and light gradients, named camera/light
hierarchies, two-sided lighting, diffuse SH coefficients, and material separation.


## Limited raster acceleration

The current CPU implementation reuses projected triangle setup in32x32 internal
pixel blocks. Conservative rejection preserves the original rounded coverage and
distance predicates;uncertain and extreme inputs retain the original calculation.
CUDA retains its pixel traversal and skips proven-outside distance work. Face IDs,
tie order,precision and output ownership remain unchanged. This internal block
size is not the caller's memory-window setting. No persistent cache or extra heap
workspace is introduced. See RASTER_BLOCK_ACCELERATION.md for the bound and tests.

The measurement-only streamed constant-albedo profile prepares pointwise lighting
once per already admitted material cohort. It retains the original four-point
partials and reverse global gradient order. This does not extend streaming to
arbitrary losses or geometry backward. Complete large jobs must still pass their
own time and memory admission;current results are recorded in devbrain.
