# Public materials: first test-first contract slice

Current milestone declaration,15October2026: M1, Group4 and finite Group5 are complete for adopted development readiness. See [exact scope and accepted presentation](M1_DEVELOPMENT_READINESS_CLOSEOUT.md). Historical technical evidence below retains its original artifact/device/workload limits; independent filtered qualification and original corpus profiles remain unpassed/unadmitted. No universal current-wheel, consumer-completion, SIGGRAPH-quality or likeness claim.

Authoritative design: devbrain `projects/gffx/SHADING_CONTRACT_V0_1.md` section
8.12. This document maps execution work, not a second editable interface contract.
The initial baselines below precede implementation. Current public CPU interfaces
are implemented and the exact CPU float64 budget is admitted. Historical missing
API and experimental-only descriptions below are dated evidence, not current status.
See `PRINCIPLED_BUDGET_V1.md` for the finite supported resource and its limits.

The separately named CUDA float64 resource is now locally admitted as well;
`PRINCIPLED_CUDA_QUALIFICATION.md` owns that route and its exact-artifact evidence.
Generic CUDA loading/transfers remain experimental. Older CPU-only rows below
describe their original slices, not a denial of this later named profile.

## Current close-out evidence (2026-09-09)

Both materials now pass small all-input native mip-chain gradchecks and seventeen
explicit fitting controls (eight parameter groups per material plus SH coefficients).
Each control compares two-scale finite differences and takes a fresh loss-reducing
step. Complete direct-light source-texture image adjoints and selected environment
adjoints have separate evidence; neither implies whole-image environment gradients.
Fullgraph and strict/non-strict export reject data-dependent validation. They remain
unsupported; no validation bypass or export/reload success is claimed.
The exact rebuilt CUDA wheel and its finite physical/repeat checks are documented
in `PRINCIPLED_CUDA_QUALIFICATION.md`. Coverage remeasurement waits until after M1.

The complete four-profile environment-image gate now passes as well. See
`SHADING_M1_STATUS.md` for the exact original image artifact, later repaired
CPU/CUDA artifact and scoped local closure; older preflight-only statements below
remain historical. No native shading or Cycles-fidelity completion follows.

`tests/python/test_material_public_contract.py` calls `gffx.torch.material`, not
the private research recipes. Its synthetic two-by-two resource is a plumbing
fixture, never a bundled or physically accepted resource. No grid is generated.
Expected missing APIs fail normally: no xfail, skip or dummy implementation.

## What a red page means

Local baseline, 2026-09-08: 48 cases collected; 46 fail specifically with
`No module named 'gffx.torch.material'`, two sanity checks pass, no errors/skips.
Pytest reports 4.20 seconds (JUnit suite 4.133 seconds). Evidence is
`build/public-material-contract-20260908/red.xml`. The installed adapter loaded
successfully with PyTorch 2.10.0+cu128; this is the installed-wheel missing-API
baseline, not current-source numerical acceptance. The separate source preflight
cannot load `gffx._torch`; no native binary was copied or installation changed.
Four offline teaching/catalog checks pass in 1.44 seconds (`teaching.xml`), and
all 15 test families map to the new page with authored explanations. Assertions
after a missing import have not executed and are not validated by that red result.

The new source catalog page is `shading/public-material-contract`. Each test has
an authored purpose and method; parameter variants keep separate outcomes. Red
means the public implementation is absent or violates its promise, not necessarily
that the independently tested shading mathematics is wrong. A passing import or
synthetic-data check does not make the material usable. Source registration is
not live dashboard publication; this slice starts no server or producer.

## Acceptance map

### M1 execution priority (2026-09-08)

The user prioritizes the functional fitting path over cosmetic demonstration
quality. Keep the procedural head and immutable images as numerical fixtures;
postpone a better-looking showcase. Public elementary lobes have independent
closed-form/gradient reds and no physical-resource dependency. Complete their
shape, sharing, endpoint, empty-input and representability tests, then implement
this bounded Phase 1 slice while Phase 0 remains open elsewhere.

The remaining gates are distinct: public light/material/source/registry behavior;
physical budget admission (synthetic tables are insufficient); complete principled
direct/environment images with bounded reference work; actual device and
compilation/export qualification; and downstream scene/animation/image integration
and real-consumer M1 fitting. Cycles fidelity and prettier demos are later work.
The existing classic image references are complete utility evidence, not public
material acceptance. This ordering does not relabel missing APIs as passes.

The subsequent classic slice implements `material.classic` and tagged
`light.integrate`, needed to execute the complete colour-source and image tests.
Classic signature and all-input gradient tests are now independent of principled
resource setup; their assertions are unchanged. The first selection passes 60
cases with 22 remaining-scope cases explicitly deselected. Expanded classic tests
add grey/RGB sharing, inactive tiny-vector gradients and integration gradcheck;
their only initial failure is the absent comparison producer. That producer now
compares the installed public functions against the explicit saved image bundle.
All eight images pass fixed tolerances, maximum absolute linear error
1.1102230246251565e-16; geometry is shared and the reference is not regenerated.
Evidence is under `build/public-classic-20260908`; usage is in
`docs/CLASSIC_MATERIAL.md`. Principled, budget lifecycle/admission and custom
materials were unimplemented at that checkpoint, so it did not close step 3.

The subsequent step-3 continuation implements PrincipledBudget, principled,
custom material/light callbacks and instance-owned registries. Its 89-failure/
55-pass baseline becomes 159 passing selected tests on an isolated CPU wheel.
Independent mean integration and native texture roughness/F0 gradients extend
that coverage. Additional parser/error-type regressions are retained separately;
NUMBERS.md owns final exact-wheel outcomes. See docs/PRINCIPLED_MATERIAL.md.
That checkpoint had no accepted bundle. Subsequent prospective physical checks and
exact identity/loading tests admit only ggx-smith-loss-cpu-f64-v1 on CPU float64.
Other research tables still require both explicit opt-ins; pictures cannot close M1.

| Group | This slice and remaining work |
|---|---|
| MAT-API | Signatures, lazy import and facade equality. Future installed-wheel acceptance must use the new implementation's exact artifact. |
| MAT-SHAPE | CPU batch/light layouts, RGB F0/resource expansion, striding, shared gradients and empty calls pass. Other resource lanes remain unqualified and are rejected. |
| MAT-GEOMETRY | Classic/principled front/back/horizon, two-sided, opposite directions, normalization and inactive tiny-vector gradients pass. No universal extreme-gradient or higher-derivative promise. |
| MAT-COMPOSE | Independent normal-incidence and oblique principled algebra, reciprocity and all-input gradcheck pass with synthetic budgets. Physical budget/furnace admission is separate. |
| MAT-SOURCE | Native corner/texture/UV-to-lit-output tests pass; additional roughness/F0 map gradcheck and repeated fitting-graph tests pass. Full renderer/consumer integration remains separate. |
| RES-FORMAT | Malformed envelopes, file errors/size/duplicates, copied inspection, roundtrip, same-lane transfer and unsupported-lane rejection pass. New huge-integer and static-option regressions preserve their red baselines. Named allocation interception is not exhaustive allocator accounting. |
| RES-ADMISSION | Exact CPU-float64 catalog binding, installed data hash, unknown IDs/lanes, tampering, no-allocation rejection and load/transfer round trips. Unrecognized data remains opt-in; other lanes are unqualified. |
| RES-QUERY | Public exact mean versus independent quadrature (values and gradients), knot/limiter derivatives, no-query-I/O and declared workspace rejection pass. Estimates are not measured peak allocation. |
| MAT-EXECUTION | Exact isolated CPU/PyTorch wheels are built and verified; ordinary environment is unchanged. No source/native splicing, GPU or compile/export qualification. |
| MAT-TEACHING | Authored test guides, public classic reference comparison, finite experimental complete-material comparison and textbook integration are implemented. Live results and exact artifact identities are recorded in NUMBERS.md. |

## Planned finite demonstrations and textbook update

After composition passes, show the same small object under classic and principled
materials with identical lights/camera and a labelled roughness/F0 or exponent
sweep. A second comparison uses identical constant vertex colours and texture
maps, followed by a seam-aware nonconstant texture and one fitting-gradient
visual. Label any experimental budget prominently. Preserve linear measurements
separately from display transforms; bound frames/artifacts and exit the producer.

Extend the material chapter with the path from source fields to BRDF to illuminated
colour, why a resource digest does not establish physical accuracy, and why learned
textures must rebuild their derived graph while budget constants are reusable.
Link the explained test page; diagrams and MathML use existing in-tree machinery.
Do not invent passing or rendered artifacts before implementation exists.

## Step 4: custom extensions

Local first red slice, 2026-09-08: 36 cases fail on absent public modules (16
material, 20 light), no errors/skips, 3.14 seconds pytest / 3.134 seconds JUnit.
`build/custom-contract-20260908/red.xml` preserves the outcome. Installed adapter
preflight succeeds; no source-tree/native environment failure is counted here.
Four offline guide/catalog checks pass in 1.39 seconds (`teaching.xml`); all 13
families map to the source-only custom page with authored explanations. No
assertions after the missing imports ran; no runtime implementation or deployment.

Contract section 8.13 adopts optional instance-owned registries and direct tensor
callables. Registry changes do not change an already-resolved function. Custom
materials see normalized active reflection pairs and return BRDF values; lighting
stays outside that callback. Parameters and gradients remain explicit.

Custom lights carry `delta` or `solid_angle` tags. `light.integrate` applies one
surface cosine and, for environment samples, one solid-angle weight. PDFs are
rejected at M1, not ignored. Invalid sample masks are distinct from occlusion.
Signed SH environment values remain signed; this is not a positivity guarantee.

`test_shading_custom_contract.py` covers registry isolation/collisions/stable
bindings, direct Lambert equivalence, active/empty callback handling, shared
gradients, output errors, exception propagation, tagged light arithmetic,
point-light equivalence, sample splitting, signed radiance and invalid/visibility
distinctions. Every family has a reader guide on source page
`shading/custom-contract`. A missing import blocks the later assertions; a red
baseline is not evidence that those assertions or an implementation passed.

Remaining step-4 fixtures include custom parameter layout/domain validation,
two-sided callback packing, full empty/strided/batched light cases, missing-key
and invalid-placeholder callbacks, custom normals/light-position gradcheck,
float32, and compilation/export of a fixed callable. Add them to the consolidated
Phase-0 step-5/6/8 acceptance map; do not label this first slice full coverage.
The planned finite demo compares built-in and custom Lambert under identical
lights before showing a visibly different illustrative material. Textbook work
will explain extension ownership and the two light measures, using existing
in-tree rendering machinery; no live artifacts are claimed at this red stage.

## Steps 5/6: boundary and acceptance consolidation

First boundary red slice, 2026-09-08: 24 failures (16 absent material module,
eight absent light module), no errors/skips, 3.36 seconds pytest / 3.355 seconds
JUnit, `build/shading-boundary-contract-20260908/red.xml`. The installed adapter
preflight succeeds. Four offline page/guide checks pass in 1.45 seconds
(`teaching.xml`); all 11 test families have authored explanations and map to the
boundary page. No assertions after absent imports have run, and no source-native
build, resource generation, physics rerun, implementation or deployment occurred.

Authoritative contract section 8.14 maps each operation to its boundary behavior,
gradient obligation, existing private recipe evidence and missing public tests.
`test_shading_boundary_contract.py` adds two-sided active-pair ordering, custom
parameter validation and gradcheck, empty material/light behavior, float32/64
horizon rules, resource query errors/transfer-copy semantics, and independent
two-node resource/normal-incidence RGB algebra. It imports only the synthetic
envelope helper from the earlier public test file, never a material implementation.
The new source-only page is now `shading/public-boundary-contract`; every family
has a guide. Its original ID collided with the older private boundary page;
the 2026-09-08 uniqueness regression and repair below preserve that older ID.

The synthetic two-node resource has a linear angular curve at its final roughness
node, so directional values, cosine-weighted mean and derivatives can be checked
without numerical integration or relying on the private resource implementation.
This is deliberately not a physical GGX reference or admission claim.

The added lobe/light/source slice below supersedes the earlier absence of those
fixtures. Still open: exhaustive edge/roundoff cases, complete principled
gradients/reciprocity/physical attachment and later execution-profile acceptance.
Step 6 and the full step-8 baseline are not complete.

## Step 6: public lobes, lights, sources and the synthetic head

The next partial red baseline has 39 failures: 24 absent shade, eight absent light
and seven absent material modules, zero errors/skips, 3.85 seconds pytest / 3.842
seconds JUnit. Evidence: `build/shading-public-composition-20260908/red.xml`.
Four offline teaching/catalog checks pass in 1.48 seconds (`teaching.xml`).
Sixteen authored test families appear on source-only pages
`shading/public-lobes-lights` and `shading/public-sources`. Imports block all later
formula/gradient assertions; these are not passing shading or sampler results.

The fixtures cover fixed-angle lobes, interior gradients, horizons, signed SH and
its truncated directional kernel, light units, equivalent colour sources, UV
seams, source-texel gradients, repeated optimization graphs and separate
background composition. The synthetic resource is an algebra control, not an
admitted physical material. No runtime implementation or live demo was added.

User-approved contract 8.15 separates two synthetic-head reference profiles:
strict independent correctness for the declared gffx models, and measured Cycles
differences. Matching Cycles remains a product goal, but full Cycles quality does
not block M1. Historical operational-gffit wording was superseded by D282 on 2026-09-09: M1 means readiness to START gffit/Compositron. See M1_CLOSEOUT_RECONCILIATION.md for current scoped evidence and remaining closeout. The direct-light CPU float64 profile uses a
128-by-128 linear image with covered-channel atol 1e-10 / rtol 1e-8 on shared
geometry buffers. That checks shading, not independent rasterization. Physical
resource acceptance remains separate; model differences cannot loosen these
correctness limits. Asset/storage, evaluator and environment convergence still
need admission before any image result. No licensed FLAME geometry is assumed.

Later paired demonstrations should show the strict reference, gffx result and
residual, alongside a separately labelled Cycles comparison. Explain which
differences come from material, lighting or transport in the textbook. Pin
Cycles configuration and convergence, then set fidelity criteria before tuning;
do not invent a passing quality threshold from the observed residuals.

## Complete-material extension and fixture-only scope

`test_shading_public_complete.py` adds nine families / 15 parameter cases:
independent oblique BRDF algebra, reciprocity of values and gradients, full-input
gradcheck for both materials, strided batching, principled inactive/two-sided
rules, query I/O prohibition, declared workspace enforcement and empty/strided
resource queries. All 15 fail at the absent material import (3.76 seconds pytest,
3.758 seconds JUnit); no later assertion has executed. Evidence:
`build/shading-complete-20260908/red.xml`. Four offline teaching/catalog checks
pass in 1.55 seconds (`teaching.xml`, JUnit 1.323 seconds). Every family has an
authored guide on source-only `shading/public-complete`.

The strengthened catalog check first found 69 entries with only 68 unique IDs
(`catalog-red.xml`, one failure). Preserve the older `shading/boundary-contract`;
give the newer public fixture page `shading/public-boundary-contract`. No service
reload or saved database migration was performed, so this is source-only repair.

CORPORA_V0_1.md now specifies the small test-only head: a closed triangulated
ellipsoid with one analytic front protrusion, welded poles, corner UV seam and
diagnostic fields. Generator code and local development storage registration are
implemented; rendered-reference admission remains pending. This is neither a public anatomy generator nor real-model fitting
qualification. Planned fitting demonstrations must be labelled synthetic.

The next slice now supplies the named positive-knot/limiter resource derivative,
allocation-before-rejection, normalized-dot roundoff and batched/strided custom-light
fixtures in test_shading_public_edges.py. Thirteen cases fail at missing public
APIs (nine material, two shade, two light); assertions beyond imports have not run.
Public physical/furnace attachment still needs
an admitted resource and independent reference; do not replace that with these
synthetic tables. The test-only generator and independent scalar evaluator are
now implemented after missing-module tests; see docs/SYNTHETIC_SHADING_FIXTURE.md.
Their utility passes do not close public shading or full-scene admission. GPU,
compile/export and performance acceptance remain their own later executions.

## Public elementary runtime checkpoint (2026-09-08)

Five public shading functions and three built-in lights now have an isolated
CPU/PyTorch wheel implementation. `test_shading_public_lobe_layouts.py` has
58 cases, and the expanded `test_shading_public_lobes_lights.py` has 61. The
initial 57 lobe and 28 additional light missing-API failures remain in
`build/public-{lobes,lights}-20260908/red.xml`; each public demo option also has
its own missing-option failure before implementation. Final adjacent regression
has 287 passes, two skips and six CUDA deselections in 38.74 seconds, recorded in
`build/public-lights-20260908/regression.xml`. This is not the complete repo suite.
The final wheel is CPU-only, SHA256
`b7fc5ab2a71fc3807f766253261b0b660241a027a29bfa6b50b8c49ce8810c46`,
installed under `build/public-lights-20260908/installed`; the normal environment
and earlier artifacts remain unchanged. See `SHADING_LOBES.md` for usage.

This makes the elementary eager API rows executable, not just specified. It does
not close complete materials, custom-light integration, physical budget admission,
complete principled image references, CUDA or compile/export qualification. No
formula, energy ceiling or resource cap changed. Existing scientific demos have
an explicit public-function mode; their artwork was not redesigned.
