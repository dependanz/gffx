# Material fields and environment lighting

These are Phase-0 reference recipes in `tools/material_contract_reference.py`.
They are not yet public `gffx.material` functions or a complete renderer. They
reuse the optional PyTorch environment; the base package gains no dependency.

## Material values

Normals shaped `[P,3]` or `[B,P,3]` establish the point and batch layout.
`parameters` validates and expands already-sampled fields without guessing
topology, image formats, channel assignments or colour spaces.

- `classic`: linear RGB `albedo`, `specular` highlight strength, and explicit
  nonnegative `exponent`. Its approximate Blinn highlight is not conservative.
- `principled`: linear RGB `albedo`, perceptual `roughness` in (0,1], and `f0`
  normal-incidence reflectance. It uses the separately specified reciprocal
  energy coupling. `specular` is not an alias for `f0`.

Albedo uses `[3]`, `[P,3]`, `[1,P,3]` or `[B,P,3]`. Specular/F0 also accept an
explicit uniform scalar or a trailing one-channel field `[P,1]`, `[1,P,1]`,
`[B,P,1]`. Use that trailing axis for grey maps; a length-three vector always
means constant RGB, not three grey points. Scalar roughness/exponent fields use
`[]`, `[P]`, `[1,P]` or `[B,P]`. All floating tensors share dtype and device.

Use native `render.interpolate` for face-corner fields or interpolated UVs, and
`render.texture` for each explicit texture map. The current sampler supports at
most four channels per call; colour, roughness and grey F0 can be separate 3/1/1
maps. Omitting LOD/derivatives samples level zero. Preserve UV seams and select
covered fragments before material evaluation. Background zero-fill is not a
valid zero-roughness material. Filter perceptual roughness before squaring it.

The tests exercise actual installed native interpolation and texture sampling,
including gradients to texture pixels and UVs. This verifies the field-input
chain, not gradients through a completed general material renderer.

## Integrating the whole response

`environment_radiance(directions, coefficients)` evaluates raw nine-coefficient
incident-radiance SH. It does **not** multiply Lambert convolution band factors.
Signed reconstruction remains signed; the caller owns any explicit positivity
parameterization.

`integrate_environment(normals, directions, radiance, brdf, weights)` sums
`brdf * radiance * max(N dot L,0) * weights` over samples. Weights are solid angles
in steradians, not PDFs or delta-light intensities. Never normalize their sum to
one or multiply the surface cosine twice. Normals and directions are explicitly
normalized; zero vectors are rejected. No shadows, visibility, gamma or exposure
are inferred.

Point count P and sample count Q are independent. Directions/radiance use `[Q,3]`
or `[B,Q,3]`, weights `[Q]` or `[B,Q]`, and BRDF `[P,Q,3]` or `[B,P,Q,3]`;
singleton batch sharing is also allowed. Output follows the normal query layout.
The BRDF must be evaluated for the matching view and sample directions, retaining
all parameter/budget gradients. The integrator cannot infer a view-side mask.

Test and demonstrate from the repository's installed-wheel environment:

```powershell
python -m pytest -q tests/python/test_material_inputs.py
python -m tools.testing_dashboard.material_environment_demo --address http://127.0.0.1:8791 --run YOUR-UNIQUE-DEMO-ID
```

The tester pages are `shading/material-inputs` and `shading/environment-demo`.
The demo uses exact special-case budgets at roughness one and F0 one under a
brighter-overhead sky. It demonstrates why the full material differs from the
Lambert shortcut; it does not implement general principled shading. The existing
energy chapter explains the fields, units and independently checked mathematics.

Still open: complete material/source-binding API, runtime budget approximation,
two-sided and view-boundary contracts, custom registries, broader full-material
gradients, scene acceptance and performance. No new CUDA/native kernel, dependency,
public API, compiler/export or real-time claim is made in this slice.
