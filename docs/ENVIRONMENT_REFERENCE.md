# When is the environment-lighting reference accurate enough?

Environment lighting arrives from many directions. The reference approximates
that continuous hemisphere with samples, evaluates the **complete material** at
each direction and adds its reflected light. It is an independent development
measuring tool, not an installed public `gffx` API or a fast image renderer.

## The fixed rules: env-sh2-v1

- Use Python binary64 arithmetic and standard-library code only. Incident light
  uses nine RGB spherical-harmonic coefficients in the project's declared order.
  Signed values stay signed; there is no display transform or clipping.
- Integrate over surface cosine `mu` and azimuth `phi`. Solid-angle weights
  sum to `2*pi` steradians. Multiply the complete BRDF by incident radiance,
  one surface cosine and one solid-angle weight—no extra sine or PDF division.
- Use 8/16/32/64/128 Gauss-Legendre rings and 16/32/64/128/256 azimuth samples.
  Evaluate every grid twice, with the second shifted half an azimuth cell.
- Each channel's two offset estimates must agree, and each must agree with its
  previous level, within `1e-11 + 1e-9*abs(current_mean)`. Require two consecutive
  passing transitions, so even a constant answer takes three complete levels.
- Stop at five levels or 87,296 BRDF evaluations per point. A 30-second
  cooperative deadline checks before and after each callback. Work controls may
  be reduced, not enlarged; tolerances are fixed. A blocking callback cannot be
  interrupted by this tool and requires a separate process supervisor.

Agreement is an empirical indicator, not a rigorous error bound. Both grids can
miss the same unsampled feature. Exact-answer tests and additional independent
controls are still necessary, especially for sharper materials. This is not an
accuracy guarantee for arbitrary callbacks or entire images.

## Use it from the repository root

```python
import math
from tools.shading_environment_reference import environment_reference
from tools.shading_independent_evaluator import classic

coefficients = [[math.sqrt(4*math.pi)]*3] + [[0., 0., 0.] for _ in range(8)]
result = environment_reference(
    [0., 0., 1.], [0., 0., 1.], coefficients,
    brdf=lambda n, v, l: classic(n, v, l, albedo=[.2, .4, .6],
                                 specular=0., exponent=2.))
print(result['radiance'])  # Approximately [.2, .4, .6].
```

The callback gets immutable normalized directions in the same coordinate frame.
Require a positive view hemisphere; explicitly flip the normal for a separate
two-sided case. Inputs are validated and SH coefficients are copied. Call only
trusted bounded callbacks. Public gffx shader calls must not supply the oracle.

`ReferenceConvergenceError.report` retains an unresolved diagnostic, reason,
completed histories and actual callback count. Never catch that exception and
use the last estimate as a qualified reference. Malformed or nonrepresentable
input/results raise `ValueError`. No call writes files, starts a server or performs
autodiff. The reference does not admit a principled physical-budget resource.

## Tests and finite demonstration

`tests/python/test_shading_environment_reference.py` checks polynomial moments,
signed/zero/near-zero light, tilted normals and all SH bands, the classic 25/24
overreflection control, and exact roughness-one complete-material moments under
constant and sloped lighting. Special-case exact budgets are not a general table
admission. Other tests cover repeatability, input mutation, sampling aliasing,
work/deadline exhaustion, nonfinite values and overflow.

The finite demonstration keeps three actual refinement histories:

1. Diffuse output settles at `[.2,.4,.6]`.
2. The empirical classic highlight settles at `25/24`; it is deliberately not clamped.
3. An oscillating test pattern fools coarse sampling. Although the third grid's
   averaged estimate happens to equal the analytic answer, its offset disagreement
   is one, so the tool correctly withholds a reference.

Generate a new create-only diagnostic report without contacting a service:

```powershell
python -m tools.testing_dashboard.environment_reference_demo --report build/my-environment-study.json
```

To send that finite demo to an already-running testing dashboard, explicitly
provide `--address <dashboard-http-origin> --run <unique-demo-id>`. This command
does not start the server. Source pages are `shading/environment-reference` and
`shading/environment-reference-demo`; the numerics textbook chapter explains them.

Evidence: `build/environment-reference-20260908/green.xml`, 52 passes (34 new
environment cases, 14 adjacent evaluator cases and four teaching checks). Preserve
the 31-case missing-module baseline and later huge-integer error/missing-demo red
tests. `study.json` holds the actual three-case report with source hashes.
No live deployment, full image, GPU/export or physical-resource acceptance is claimed.
