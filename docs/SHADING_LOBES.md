# Elementary shading functions

The development wheel now exposes five eager PyTorch functions through
`gffx.torch.shade` and the lazy `gffx.shade` facade. Importing the latter does not
import PyTorch; calling a function activates the adapter. No new dependency is
added to the base package. These functions are not yet a complete material API.

```python
import torch
import gffx

colour = torch.tensor([[0.2, 0.5, 0.8]], requires_grad=True)
brdf = gffx.shade.lambert(colour)  # [1,3], linear reflectance / pi
brdf.sum().backward()
```

`blinn_phong`, `ggx_distribution`, `smith_visibility` and `fresnel_schlick`
evaluate a highlight, microfacet distribution, masking/visibility factor and RGB
Fresnel factor respectively. See each function's docstring. Complete illuminated
colour also needs incident light and a surface cosine; avoid multiplying either
twice. A microfacet distribution or Fresnel factor alone is not a material.

Inputs are same-device float32/float64 tensors. Cosines have shape `[P]` or
`[B,P]`; Lambert colours have `[P,3]` or `[B,P,3]`. Scalar parameters can have
shape `[]`, `[P]`, `[1,P]`, `[B,P]`; Fresnel colours can have `[3]`, `[P,3]`,
`[1,P,3]`, `[B,P,3]`. The first input owns the output layout. The point dimension
never broadcasts, and a batch cannot be expanded by a parameter. Empty point
sets, strided views and expanded shared fields are supported with gradients.

No Python-number coercion, dtype promotion, device transfer, roughness floor,
cosine clamping or energy clipping is performed. Invalid types/dtypes raise
`TypeError`; invalid shapes/devices/domains or nonrepresentable positive divisors
raise `ValueError`. Shared fields are validated even on empty calls. Horizon
extensions for Blinn and Smith have explicitly zero first derivatives; Schlick
and Lambert keep their polynomial/linear endpoint derivatives.

Current acceptance is an isolated Windows CPU/PyTorch wheel, not a full device,
native-C or compilation/export qualification. Eager value checks may synchronize
a device and are not advertised as an exportable graph. Complete materials and
physical budget-resource admission remain open. Classic Blinn is empirical:
its unit-weight exponent-two furnace gives 25/24, not an energy guarantee.

The testing dashboard's `shading/public-lobe-layouts` and
`shading/public-lobes-lights` pages hold actual test outcomes. The separate
`shading/public-lobes-demo` page reuses the bounded four-sphere illustration with
the public functions; this is a scientific comparison, not new demo artwork.
The energy-conservation textbook chapter already explains the same equations,
endpoint distinctions and independent `1-log(2)` microfacet control.

## Built-in lights

`gffx.torch.light` now provides `point(points, positions, intensity)`,
`directional(directions, irradiance)` and
`spherical_harmonics(normals, coefficients)` as eager PyTorch compositions.
They use the same explicit dtype/device policy. Directions and normals are
normalized without an epsilon floor. Only singleton batch sharing is implicit;
point and light axes remain distinct, with three explicit RGB channels.

Point lights return a mapping with unit `direction`, inverse-square `irradiance`,
`distance_squared` and a boolean `valid` mask. Coincident pairs are invalid and
have zero outputs and first derivatives. Directional lights return unit direction
and unchanged normal-incidence irradiance. Neither includes the surface cosine,
material response, occlusion, shadows or a solid-angle PDF.

SH lighting consumes nine RGB incident-radiance coefficients and returns
cosine-convolved irradiance, appropriate for diffuse lighting. Signed values are
preserved: the band-two approximation can ring negative. Do not use that diffuse
convolution as a substitute for full glossy environment integration. Custom-light
packets and `light.integrate` are not implemented in this slice.
