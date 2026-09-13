# Principled and custom materials

`import gffx` remains dependency-free and lazy. The explicit PyTorch adapter needs
the user's PyTorch installation. The public material APIs are torch compositions;
they do not import development tools or generate a lookup table at runtime.

## What works, and what is not qualified

`gffx.torch.material.principled` combines GGX distribution, height-correlated Smith
visibility and Schlick Fresnel with reciprocal matte. It returns an RGB BRDF, not
illuminated colour. Use `light.integrate` for tagged direct/environment samples.
Texture, UV and corner sources stay explicit native tensor compositions. Gradients
reach source texels, roughness, reflectance and geometry; rebuild derived source
graphs after optimization updates.

The named `PrincipledBudget` profiles use **float64 on CPU or CUDA**, with distinct
resource IDs and finite qualification records; generic CUDA transfers remain
experimental. At most **32 scalar or channel entries per standalone query** are
supported. Material calls separately support up
to **1024 total point/light pairs**, with **8 points forward** or **4 points when
gradients are enabled and any input requires gradients**. Counts include all
batches and inactive pairs. Shared point curves and per-pair live state are charged
against the same default 64 MiB planning limit. Empty work retains its existing
shape/zero-gradient behavior. These are declared
limits, not a measured OS memory reservation or renderer-scale performance claim.
No automatic chunking, generation, downloading or fallback occurs.

For larger scenes, the caller tiles explicitly. The larger material capacity was
checked against small-call values and all-input gradients, including strided batched
fields. Four capacity corners received finite repeated-call memory checks. See
[the measured batching profile](MATERIAL_LIGHT_BATCHES.md); this is not full-image,
GPU, sustained-memory or frame-rate qualification.

The exact `ggx-smith-loss-cpu-f64-v1` bundle has a finite CPU float64 acceptance
profile. Load it without a download or research flags:

```python
import torch
from gffx.torch import material
budget = material.PrincipledBudget.bundled(
    "ggx-smith-loss-cpu-f64-v1", device="cpu", dtype=torch.float64,
)
```

See [its identity, evidence and limits](PRINCIPLED_BUDGET_V1.md). Unknown IDs fail;
a payload hash alone does not prove physics.

For the named CUDA profile, use `ggx-smith-loss-cuda-f64-v1` with a CUDA device
and float64; see [CUDA usage and qualification](PRINCIPLED_CUDA_QUALIFICATION.md).
It needs no experimental material flag, but does not change generic loading or
transfer semantics. All query/material tensors must already share its device.

An unrecognized local research table still uses both opt-ins:

```python
import torch
from gffx.torch import material, light

budget = material.PrincipledBudget.load(
    "my-experimental-budget.json", device="cpu", dtype=torch.float64,
    allow_unverified=True,
)
n = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
brdf = material.principled(
    n, n, n,
    albedo=torch.tensor([.2, .4, .6], dtype=torch.float64),
    roughness=torch.tensor(.4, dtype=torch.float64),
    f0=torch.tensor(.04, dtype=torch.float64),
    budget=budget,
    allow_experimental_budget=True,
)
```

This example requires an existing valid local envelope; it does not create one or
claim it has passed physical tests. Both opt-ins are deliberate. Synthetic test
envelopes establish algebra and interface correctness only. `metadata`, `inspect`
and `to_envelope` return copies; `to` makes a new resource and rejects unqualified
lanes. Queries perform no file reads, parsing, transfers or quadrature. Constants
are retained, but evaluated parameter-dependent losses are never cached.

Normals own `[P,3]` or `[B,P,3]`; output is `[P,Q,3]` or `[B,P,Q,3]`. View and light
directions follow `CLASSIC_MATERIAL.md`. Roughness is scalar or `[P]`, `[1,P]`,
`[B,P]`; F0 is scalar, RGB `[3]`, or grey/RGB `[P,C]`, `[1,P,C]`, `[B,P,C]`, C=1/3.
No per-light material fields or silent channel/scene inference. Inactive pairs
return zero values and gradients. Two-sided lighting needs the same oriented
normal for the material and integration cosine.

## Bring your own BRDF

```python
import math

def matte(normals, view, directions, *, parameters):
    return parameters["colour"] / math.pi

registry = material.Registry()
registry.register("example/matte", matte)
bound = registry.resolve("example/matte")  # resolve during setup
result = material.custom(
    bound, n, n, n,
    parameters={"colour": torch.tensor([.2, .4, .6], dtype=torch.float64)},
)
```

The callback gets only active normalized pairs `[A,3]` and packed parameters
`[A,K]`. Scalar parameters become one channel; one-dimensional fields always mean
constant channels, not per-point values. Return finite nonnegative `[A,3]` BRDFs.
Empty active sets do not invoke the callback. Registries are local, no plugin
discovery occurs, and a resolved callable survives later unregister/re-register.
Callbacks are trusted code and must not mutate inputs. Their compatibility does
not prove physical correctness, reciprocity, purity or compilation support.

`light.custom` calls a light function once, including empty point sets. It requires
exact `direction`, `valid` and `irradiance` keys for `measure="delta"`, or direction,
valid, radiance and solid_angle for `measure="solid_angle"`. No PDFs or mixed tags.
Invalid entries require zero placeholders; valid directions are normalized. The
integrator applies one cosine and one measure weight. Signed environment radiance
stays signed. Coverage, visibility, shadows and background composition stay outside.

## Evidence and outstanding work

Tests exercise closed-form oblique and normal-incidence values, reciprocity and
all-input derivatives, exact curve means versus independent quadrature, knot and
limiter rules, native texel/UV gradients, repeated fitting graphs, safe resource
loading, workspace preflight and custom callback contracts. Public curve code is
a port of the frozen internal recipe, **not** an independent physical oracle.

The finite `shading/complete-material-demo` page has classic, two experimental
principled roughnesses and built-in/custom Lambert on the same analytic sphere.
The synthetic principled budget is clearly labelled; the only equivalence metric
there is built-in/custom Lambert. Display conversion does not alter linear tests.

Beyond the one exact finite-profile bundle, complete principled image
profiles, renderer-sized workloads, CUDA/float32 resources, compilation/export and
real-model M1 fitting remain separate gates. There is no TorchScript promise or
claim that the full shading phase is complete.

### Finite physical screen, 2026-09-08

The saved conditional-allocation tables now pass a prospective installed-public
CPU float64 screen: 80 reference-convergence values, 80 directional loss/gradient
triples, 16 mean-loss/gradient pairs and 192 furnace cases. Reports are generated
by `tools.public_budget_probe` under the existing 120-second cooperative deadline,
180-second watchdog and 1 GiB whole-worker cap. The frozen comparison cohort is
declared in devbrain's `PUBLIC_BUDGET_ACCEPTANCE_V0_1.md` before measurement.

Directional values use independent integration, not the research lookup query.
Mean and furnace checks also use a separately integrated scalar curve. This is
finite empirical evidence: it does not prove error bounds at every possible input.
The cohort is now regression data, not fresh data for future tuning. These passes
do not alone populate a catalog or remove either experimental opt-in. The subsequent
exact catalog and installed-package tests qualify only the named bundle; all other
resources retain both experimental opt-ins.
