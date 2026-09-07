# Experimental energy-budget table

This is a candidate for the runtime budget calculation, not an admitted public
material implementation. It introduces no external table, library or dependency.

The generator samples our own visible-normal reference at two Fresnel endpoints,
F0=0 and F0=1. Schlick is affine in F0, so intermediate reflectance is interpolated
exactly with respect to that parameter. The other axes are perceptual roughness
and view cosine. The candidate uses local tensor-product quadratic interpolation and retains gradients
through the query coordinates. Constant sampled data do not need gradients.

The average budget is the exact cosine-weighted mean of the piecewise-quadratic
cosine table, not an independently guessed number. This preserves consistency
between directional and average budgets. It does not by itself prove agreement
with the original specular BRDF; the independent furnace still checks that.

```python
import torch
from tools.material_budget_table import generate

torch.set_num_threads(1)  # bound this explicit offline generation job
table = generate()       # expensive: build once, not once per frame
c = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
r = torch.tensor(0.63, dtype=torch.float64, requires_grad=True)
f0 = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
budgets = table.evaluate(c, r, f0)
budgets['directional'].sum().backward()
```

## Deliberate limits

- Candidate coverage is roughness [0.2,1], cosine [0,1], F0 [0,1]. Out-of-range
  queries fail, never clamp. This does not change gffx's physical roughness domain
  or adopt 0.2 as a public minimum. A sharper-material strategy is still required.
- Inputs are CPU float32/64 tensors of matching dtype. Cosine owns the shape;
  roughness/F0 are scalar or exactly that shape. Material RGB/batch adaptation is
  subsequent integration work, not guessed by this generic scalar-field evaluator.
- Derivatives are piecewise slopes. The three-node stencil starts at the enclosing
  cell's left node, capped to the final three nodes at the upper edge. At internal
  knots the right-hand stencil is used. No globally smooth or second-derivative
  accuracy claim is made. Gradcheck fixtures avoid knots.
- Default data have 33 uniform roughness knots, 257 squared-uniform cosine knots
  and two Fresnel endpoints. Each pair shares 256x512 visible-normal geometric
  samples: 1111621632 samples total, two endpoint outputs each. Tensor payload is
  138544 bytes, excluding objects and temporaries. Quadratic weights can be negative;
  a finite 65x513 range screen is not a proof of global shape preservation or accuracy.
- Generation is explicit and capped at 1200000000 quadrature samples per call.
  That is an offline work cap, not a memory quota or time guarantee. Keep jobs
  serial and use a bounded CPU thread count. No implicit disk or parameter cache.
- This development object exposes mutable tensor data; it is not a hardened
  artifact loader or a frozen public material resource. Artifact identity,
  loading, device transfer, compilation/export and package distribution are open.

The initial low-precision generator returned 1.0000000000000013 at the unit-Fresnel
horizon due to floating-point integration. The guard rejected it; the test evidence
is retained. This is roundoff, not a new physical counterexample. No energy clamp
or tolerance change was introduced to admit it.

The acceptance tests separately measure held-out value/gradient error, exact table
mean consistency, independent average gradients and the original 108-case furnace.
Timing demonstrations report local CPU measurements, not a renderer frame rate or
real-time inference guarantee. The independent reference is usable for comparison
only where its own convergence guard passes.

## Expanded accuracy checks

The directional sweep checks 120 triples: ten view cosines, four roughnesses and
three Fresnel values. Each comparison first checks the reference at two integration
resolutions, then the table's values, all three gradients and coarse/fine changes.
The initial expansion found unresolved reference convergence at roughness 0.213
and 0.337 with view cosine 0.025 and 0.11 (twelve triples). Those historical
failures are retained. The subsequent fixed-Gauss azimuth reference repair resolves
all twelve against unchanged refinement limits and independent VNDF values.
It exposes one genuine table screening failure: at cosine 0.025, roughness 0.213
and F0=1, the cosine derivative is -1.9552268993 versus reference -2.0666745679.
The error is about 5.4 percent and exceeds the original absolute-plus-relative
gradient limit. That original table remains reproducible with `cosine_nodes=129`,
`roughness_grid='uniform'`, `interpolation='linear'`; a negative control requires
its measured failure. Doubling the cosine knots fixes that point but leaves three
new grazing-screen failures. A linear sharp-dense roughness grid makes sixteen
checks fail, including furnace refinement, and was rejected as default. It remains
selectable with `roughness_grid='sharp-dense', interpolation='linear'`.

The current uniform/quadratic candidate passes all 81 focused tests, including all
120 triples and the original independent furnace, without changing their limits.
An exactly known polynomial verifies the new interpolator's values, gradients and
mean independently of BRDF quadrature. Query work grows from four to nine endpoint
pair reads. Timings must be measured anew; earlier linear timings do not describe
this candidate. Finite screening still does not establish public-domain admission.

Other checks cover one-sided knot/endpoint derivatives, float32/float64 agreement,
explicit rejection of sharper out-of-range queries, and twelve derivatives through
the complete scalar-colour shiny-plus-matte response. The latter normalizes normal,
view and light vectors and includes both directional and average budget gradients.
A deliberately detached-budget control must give a different roughness gradient.
It shares specular algebra with the comparison path, so it isolates budget/coupling
accuracy rather than independently verifying every lobe or a public material API.

Passing internal gradcheck means autodiff differentiates the implemented table
correctly; it does not mean the table approximates physical derivatives accurately.
The mean budget is independent of the query view cosine: its relevant derivatives
are roughness and F0, not a fabricated viewing-angle dependency.
