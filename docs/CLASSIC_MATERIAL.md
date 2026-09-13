# Classic material and light integration

The development CPU/PyTorch wheel implements `gffx.torch.material.classic`, also
available as `gffx.material.classic`. Importing the facade is lazy. No dependency
has been added to the base package.

```python
import torch
import gffx

n = torch.tensor([[0., 0., 1.]])
colour = torch.tensor([.2, .4, .6], requires_grad=True)
brdf = gffx.material.classic(
    n, n, n, albedo=colour,
    specular=torch.tensor(.1), exponent=torch.tensor(16.),
)
samples = {
    "measure": "delta",
    "direction": n[:, None, :],
    "irradiance": torch.ones((1, 1, 3)),
    "valid": torch.ones((1, 1), dtype=torch.bool),
}
rgb = gffx.torch.light.integrate(n, brdf, samples)
rgb.sum().backward()
```

Classic returns `albedo/pi + specular*Blinn(N.H, exponent)` on the active
reflection hemisphere. It is empirical and is **not energy-conserving**.
The material normalizes finite nonzero vectors. Normals own `[P,3]` or `[B,P,3]`;
view has `[P,3]`, `[1,P,3]` or `[B,P,3]`; directions have `[Q,3]`, `[P,Q,3]`,
`[1,P,Q,3]` or `[B,P,Q,3]`. A rank-three light tensor always means point/sample,
never batch/sample. Output is `[P,Q,3]` or `[B,P,Q,3]`.

Albedo is explicit RGB. Specular additionally accepts a scalar or point field
with one trailing channel. Exponent accepts the explicit scalar/point/batch
layouts used by the elementary lobes. No Python-number coercion, dtype/device
promotion, hidden texture lookup or colour conversion occurs. All original
inputs are validated even when every pair is inactive or the output is empty.

`two_sided=True` flips the normal when it faces away from the view. Horizon/back
pairs produce zero values and first derivatives; inactive half-vectors are never
normalized in the backward graph. This boundary convention does not promise a
smooth physical function at the horizon. For two-sided lighting, pass the same
oriented normal to the light integrator; it cannot infer the view direction.

`light.integrate` consumes exact tagged packets, not bare point/directional
results. Explicitly arrange their point/sample axes, attach `measure='delta'`
and a boolean validity mask, and omit diagnostic fields such as squared distance.
It multiplies the BRDF by light and one surface cosine. A `solid_angle` packet
instead carries signed `radiance` and nonnegative `solid_angle` quadrature
weights. No packet broadcasting or PDF conversion is performed. Invalid entries
must have zero placeholders. Visibility and background composition stay separate.

Per-corner colour and UV sampling use the existing native interpolation and
texture operations. Their gradients remain connected through shading. Rebuild
parameter-dependent texture pyramids each optimization step; do not retain a
stale derived graph.

Acceptance is currently eager CPU float32/64 plus CPU float64 image/source tests.
Eight saved synthetic-head references match within fixed linear tolerances, on
shared geometry buffers. This does not independently qualify the rasterizer or
constitute principled, physical-resource, custom-material, CUDA/export, Cycles or
real-consumer M1 acceptance. The new demo retains the test head without cosmetic
changes. Principled and custom material entry points remain unimplemented.
