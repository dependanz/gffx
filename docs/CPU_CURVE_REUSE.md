# CPU material curve reuse

The principled material now builds one roughness-dependent curve per active
surface point and call. The view, mean and incident directions use that same
current autograd graph. RGB channels mix the two Fresnel endpoint losses after
evaluation. Only the four required angular controls are gathered per light.
Nothing is cached across calls; fitting updates cannot reuse stale curves.

The resource, physical model, public API, CPU float64 restriction, 32 expanded
query-entry limit and conservative 64 MiB workspace accounting are unchanged.
Eight-point backward calls still fail before curve allocation. No new dependency.

The first candidate exposed cancellation in the horizon derivative. Writing the
linear segment as `y0 + u*(y1-y0)` subtracts close endpoints before division,
avoiding cancellation between two large derivative terms. A known-answer slope
test failed before this correction; no accuracy tolerance was relaxed.

## Bounded measurements, not frame-rate claims

On the recorded Windows CPU/PyTorch 2.10 float64 lane, the exact 65,536-direction,
6,554-call environment check took 37.8784 seconds versus 110.2051 seconds before
(about 2.91x faster). Both RGB estimates are identical to the recorded baseline;
the largest independent-reference error remains 2.108e-10. This is **one surface
point at stringent quadrature accuracy**, not a whole frame or ordinary pixel cost.

Three-run median material-only checks after one warmup:

| Workload | Before | After |
|---|---:|---:|
| 1 point, 10 lights, forward | 16.445 ms | 4.242 ms |
| 8 points, 1 light, forward | 26.333 ms | 5.667 ms |
| 4 points, 1 light, forward plus backward | 34.870 ms | 12.546 ms |

The backward check saved 900,588 bytes of unique autograd storage versus
7,537,404 bytes before. This includes shared inputs/constants and is not peak
scratch allocation. The supervised cost worker peaked at 553,504,768 bytes;
the environment worker at 546,414,592 bytes, including Python/PyTorch.
Both stayed below the unchanged 1 GiB sampled whole-worker limit. Twelve repeated
calls per case are finite evidence, not a leak-freedom or sustained-use proof.

Evidence: `build/curve-reuse-20260908/{baseline-costs,optimized-costs,optimized-environment}.json`.
Optimized wheel SHA256: `258f876564cf79e697c51fc40026d0dd0038f152bfb583d865bed4e7d1afed06`.
Each report verifies installed package bytes against its explicit wheel.
Runtime changes are installed only in the isolated `final-installed` directory;
the ordinary environment is untouched. Earlier failed wheels/tests remain saved.

Full environment images still need a separately bounded total-work plan.
This optimization does not close image, device, export, gradient-phase or M1 gates.
