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
points. A bounded first-order recomputation implementation is now under
qualification; its four tiny CPU value/gradient/storage controls passed, but this
does not admit a large-image memory profile. Small point
shading qualification is not permission to retain a full image of graphs.
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
