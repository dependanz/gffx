# Differentiable material-budget reference

This is Phase-0 mathematical tooling, not a public `gffx.material` API. It uses
only the existing optional PyTorch development environment. The base package
still does not acquire a dependency.

The shiny GGX component has a directional reflected fraction E. The approved
principled model uses the remaining fraction for its reciprocal matte component.
An optimizer needs derivatives **through** E and its cosine-weighted average,
not just derivatives through the final coupling formula.

`tools/material_budget_reference.py` integrates fixed light directions using
Gauss-Legendre cosine nodes and, by default, Gauss azimuth nodes. Material-dependent arithmetic
stays in the graph. Only constant nodes are cached. There is no hidden clamp,
external LUT, or detached parameter-dependent table.

```python
import torch
from tools.material_budget_reference import directional, average

c = torch.tensor(0.43, dtype=torch.float64, requires_grad=True)
r = torch.tensor(0.63, dtype=torch.float64, requires_grad=True)
f0 = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
energy = directional(c, r, f0, resolution=256)
view_gradient, roughness_gradient, fresnel_gradient = torch.autograd.grad(
    energy, (c, r, f0))
mean = average(r, f0, resolution=128, view_resolution=32)
```

Run from the repository with the project's installed-wheel test environment:

```powershell
python -m pytest -q tests/python/test_material_budget.py
python -m tools.testing_dashboard.material_budget_demo --address http://127.0.0.1:8791 --run YOUR-UNIQUE-DEMO-ID
```

## Scope and safeguards

Inputs are scalar float64 CPU tensors. The diagnostic view cosine is strictly
between zero and one; roughness is in (0,1], and F0 in [0,1]. These are reference
tool restrictions, not the final public material schema. Invalid types/shapes,
nonfinite values, lost positive divisors and oversized work requests fail explicitly.

Directional resolution is 16..512. The default `azimuth='gauss'` uses the same
number of azimuth nodes on half the circle and doubles the measure by symmetry.
The retained `azimuth='midpoint'` uses twice as many uniformly spaced azimuth
samples on the full circle, reproducing the original reference. Neither rule
moves sample coordinates with the query parameters. Average view resolution is
16..64, with the original conservative work cap of 2097152 light samples. Graph
storage can still be substantial; keep calls serial and release their graphs.
The cap is not a global process quota or measured memory guarantee.

Fixed quadrature can miss narrow highlights. Increasing precision fixed two
initial convergence failures without relaxing test limits. The accepted fixtures
cover three held-out view/roughness/F0 combinations, independent analytic controls,
and an average-budget gradient refinement case; not every positive roughness or
grazing limit. Do not infer an accuracy guarantee from a finite return value.

The value oracle uses a separate visible-normal parameterization. Derivative
references use refined uniform direct-hemisphere central differences. Ordinary
gradcheck separately checks the complete coupling chain, including all three
computed budgets. Passing gradcheck alone does not prove physical accuracy.

The expanded angular-grid checks retain twelve glossy/grazing triples. The old
uniform azimuth grid does not converge at the tested resolutions; the new fixed
Gauss grid clusters samples near phi=pi and is checked both by resolution refinement
and by independent visible-normal (VNDF) quadrature values. This exploits isotropy:
the integrand depends on azimuth through cos(phi), so twice the half-circle integral
equals the full-circle integral. No sample coordinate depends on a query tensor.
The legacy rule remains selectable for reproducibility, not as an automatic fallback.

The tester's `shading/budget-contract` page explains each check.
`shading/budget-demo` shows 33 finite samples of energy and roughness derivatives.
Its white-matte curve is constructed as 1-E, so its sum-to-one is **not** independent
furnace acceptance. The existing 108-case material-contract furnace remains separate.

Public scalar/RGB/batch schemas, vector-normal derivatives, endpoint behavior,
per-vertex/textured material inputs, full environment integration and a practical
runtime budget approximation have separate staged evidence in the current plan;
this reference alone does not accept their public implementation. No new native kernel,
CUDA/compile/export acceptance or real-time inference claim is made here.
