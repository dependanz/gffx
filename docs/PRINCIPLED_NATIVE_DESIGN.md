# Native Principled execution, M1 design

The 2026-09-14 approved scope brings the existing Principled forward and first
VJP into native execution. It changes scheduling, not material mathematics or
acceptance. The implementation proceeds from independently tested arithmetic to
loss-curve construction/query, then the material evaluator and deferred dispatch.
No speed or completed device gate is inferred from a standalone scalar probe.

## First boundary: immutable roughness control interpolation

Private `native/core/principled_curve_scalar.h` is shared C11/CUDA arithmetic,
not an installed header or exported gffx ABI. It implements positive-unit log,
bounded negative exponential, and one immutable four-control log-Bezier curve
with its first VJP with respect to the interpolation coordinate. The polynomial,
split constants, normal/subnormal scaling and de Casteljau operation order must
match `_curve_math.py` on the installed vram19 reference. No generic exp/log API,
new table, resource admission, precision or material model is introduced.

Inputs are four finite float64 controls in [-745,0], coordinate u in [0,1], and
finite incoming cotangent. Controls are immutable resource-derived constants.
Outputs are exp(B(u)) and `(cotangent * value) * B_prime(u)`, preserving the
existing analytic `_Exp` then `_CubicU` first-gradient order. B_prime uses the
existing quadratic de Casteljau differences. There is no coefficient gradient.
The subsequent curve family must reduce per-coordinate contributions with the
existing adjacent-pair/carry-odd tree before propagating through log roughness.
No cross-pixel or shared-parameter reduction is part of this scalar boundary.

Raw helpers require validated inputs, disjoint output/input storage and default
round-to-nearest IEEE arithmetic; validation belongs to the eventual native
operation wrapper. The test-only host wrapper checks finite/domain/null inputs,
computes into locals and preserves both output sentinels on failure. Scalar
scratch is fixed-size automatic storage, no heap or cached state. CUDA compilation
uses no FMA, no FTZ, precise division and no fast math. The CUDA test wrapper is
not the optional production provider and is never packaged in a wheel.

The public resource limits remain 32 query entries, 8 forward/4 backward material
points and 1024 point/light pairs, with current workspace accounting. This first
boundary does not change them or admit larger batches. Full native arguments,
caller-owned scratch, alias checks, stream/error behavior and provider-tail
compatibility are to be fixed before adding a public native entry point. The
public Python material API stays unchanged. First VJP only follows the existing
roughness exponential's once-differentiable behavior.

## Prospective qualification

1. Preserve a missing-native red using the complete test harness before creating
   the shared header. Compile the C11 probe under strict floating point.
2. Compare log and roughness values and analytic VJPs against independent Decimal
   Bernstein polynomials and derivatives, including endpoints, constant controls,
   steep losses, representable subnormals and finite signed cotangents. The existing
   log certificate is reused as a bound, not substituted for physical tolerances.
3. Compare every scalar value and first VJP byte with the installed `_curve_math`
   reference at ordinary, endpoint and subnormal fixtures. A discrepancy stops
   this boundary and requires diagnosis; the reference is not rewritten to agree.
4. Compile the same header for CUDA; compare CPU/native/CUDA/reference bytes and
   record input/library/source hashes. This is standalone arithmetic evidence,
   not installed-wheel, physical-material or large-frame qualification.
5. Extend test-first to bounded angular slopes, controls, analytic mean and
   directional query, then full material/adapter/provider integration. Requalify
   changed physical evaluation, build a fresh wheel and measure new small-frame
   calibrations before admitting large profiles. Keep heavy workers serial and
   all historical failures. Do not repeat unrelated expensive completed tests.

## Angular limiter boundary

The next private primitive accepts a strictly increasing float64 log-cosine axis,
paired endpoint values y[node,2], and a directional derivative dy[node,2] for
one roughness coordinate. It produces the same bounded slopes and their analytic
directional derivative. This is internal chain-rule support, not a new public JVP
API or a claim that the complete material first VJP is implemented. Outputs use
caller storage; no allocation and no change to query/material limits.

Primal operations retain `_loss_curve.bounded_slopes` order. At min/max ties,
derivatives split equally as in the current tensor limiter; sign and predicate
choices have zero derivative. Selection of an inactive limiter contributes no
derivative. The division derivative is evaluated as `(da - quotient*db)/b`;
subsequent full first-VJP integration must independently qualify the complete
chain and shared reductions, rather than infer them from this directional check.
A scalar pair uses only adjacent secants and fixed automatic temporaries. The
wrapper validates n in [1,1024], increasing finite axis and finite input tangents
before dispatch; this node bound is not a raised resource-entry limit.

Prospective checks compare rationally evaluated bounded slopes, installed tensor
value bytes, independent two-step directional finite differences away from
limiter switches, explicit linear/constant/tie cases, and shared C11/CUDA bytes.
Any mismatch is retained and diagnosed before integrating this helper.

The first angular probe found one internal negative-zero/positive-zero difference
against the installed tensor slope array. A separate check confirmed all resulting
interpolation-control bytes agree on all six fixtures. Tensor min/max signed-zero
ties depend on scalar/vector evaluation; the boundary gate therefore compares
all nonzero slope bytes and zero masks, and requires all actual control-output
bytes. This does not relax any existing public value/device/gradient criterion.
The original stricter intermediate failure remains in principled-slopes-failed1.
A five-node derivative fixture also crosses the raw-slope zero limiter. Independent
rational branch signatures now certify smooth nodes before two-sided differences;
all nodes still face the declared tensor branch-derivative check. The original
incorrect branch-stability assumption and its failed report remain historical.

## Complete call-local curve construction

A private builder composes the qualified pieces for one roughness coordinate.
Inputs: K log-cosine nodes (1..512), immutable selected roughness controls with
shape [(2*K+1),4] (two endpoint channels per node then horizon), and u in [0,1].
It writes 10*K-6 doubles: values[K,2], horizon[2], controls[K-1,4,2], with an equally
sized derivative array with respect to u. Caller scratch is 4*K doubles for paired
slopes/tangents. K=1 has no angular intervals and retains the two-node resource
case. All arrays are disjoint caller-owned storage; no heap, cache or public ABI.
Resource lookup, log-roughness coordinate and gradient contraction remain the
caller's separate boundary. Tests first require complete builder outputs to match
the installed bundled-resource `_curve` bytes, and analytic directional values
to meet existing derivative controls. This does not qualify a complete first VJP.

## Endpoint loss query

The private query consumes a constructed curve and its derivative with respect
to u, the positive cosine nodes, log coordinates, immutable analytic weights,
and one cosine c in [0,1]. Ten output doubles are directional pair, mean pair,
their four u derivatives, and two directional cosine derivatives. Caller scratch
is 8*K doubles for the mean's fixed adjacent-pair/carry-odd reductions. The horizon
uses the existing linear difference form; positive angular intervals use the
existing de Casteljau controls. The mean preserves the original term order and
reduction tree. No F0 mixing or material/geometry adjoint is included yet.

Prospective tests require exact bundled-resource query values, direct first-VJP
projections against the installed reference at horizon/first-node/interior/end,
and independent analytic two-node directional/mean losses. CUDA uses the same
source and must additionally preserve every native value/partial byte. Existing
physical gates and full material VJPs remain separately required before integration.

### Independent grazing-angle gradient finding

The first query bridge fails against vram19 at cosine1e-10. A150-digit Bernstein
partial independently agrees with the native scalar to roughly1e-15, while the
installed reverse de Casteljau coordinate gradient differs by up to1.12566e-6
on the four captured roughness rows. The subtraction of near-equal primal terms
in the generic reverse graph loses precision before division by a tiny cosine.
The original failed comparison and all28 independent diagnostic rows are retained.
Repair the eager coordinate gradient test-first with the analytic difference
polynomial, preserving its forward value expression and coefficient derivatives.
A fresh installed wheel and affected numerical/physical gates are required; the
native diagnostic does not silently replace the old reference or relax tolerance.


## Production curve boundary (2026-09-14)

The first installed native subset is call-local curve construction plus its first
coordinate VJP. This reduces repeated eager curve work; it is not yet the complete
native material evaluator or a large-frame speed claim. Public C shading entries
accept dense float64 tensors: axis[K], immutable controls[B,2*K+1,4], u[B], with
0<=B<=32 and 1<=K<=512. Axis is finite, strictly increasing and <=0; controls lie
in [-745,0], u in [0,1]. Output and coordinate partial are disjoint [B,10*K-6]
arrays using the existing values/horizon/angular-control layout. Coefficients
and axis have no gradient: only immutable resource construction may feed them.

A separate first-VJP entry contracts partial[B,10*K-6] with finite incoming
cotangent of the same shape, producing grad_u[B]. Products followed by the fixed
adjacent-pair/carry-odd tree define contraction order identically on CPU/CUDA.
The derivative is independently checked against direct eager first VJPs and
finite differences under existing tolerances; no higher-order API is promised.
Roughness-log and shared material reductions remain in the existing adapter graph.

Caller scratch for both devices is 8 + 8*B*max(4*K,10*K-6) bytes, alignment8;
this covers status and per-row construction/contraction scratch without native
heap allocation. Views, outputs and workspace must have matching device/index,
and workspace/output cannot overlap any input or each other. Input-input alias
is allowed because inputs are read-only. All shape/domain checks finish before
output writes. Arithmetic overflow returns OVERFLOW and invalidates outputs.
CUDA uses the caller stream for validation and evaluation, synchronizing only
that stream to return checked status; it neither changes nor owns the stream.
The append-only provider tail must reject missing/short entries before access.
The public material/query signatures and 8/4 material-point,32 query-entry and
1024 point/light-pair caps are unchanged. The native adapter is not activated in
public material execution until fresh installed value/VJP/device gates pass.


The production C11/provider/Stable-ABI operation passed the first fresh nativecurve21
installed CPU and CUDA value/first-VJP, stream and failure tests. Nativecurve22
connects it to `PrincipledBudget._curve`; `_curve_reference` retains the original
eager formula for independent regression, never an implicit runtime fallback.
The native result stores one partial array for the current graph; the adapter
passes its accumulated curve cotangent to the native contraction, then through
the existing log-roughness coordinate graph. This only integrates curve building;
query/material fusion and all-input physical/resource acceptance remain gates.


Nativecurve22 profiling attributes about1 second of an instrumented32-square
frame to101 native curve calls, while substantial eager material/query work
remains. A node-parallel CUDA candidate splits immutable value construction,
paired angular slopes and angular controls into ordered kernels on the same
caller stream. Each node keeps the original scalar operation order. Independent
nodes are parallelized within the existing batch; the32-entry and4/8 material
point caps, allocations and fixed VJP contraction do not change. This scheduling
candidate requires full-cap CPU/CUDA byte checks before any speed claim.

## Prospective native query bridge (2026-09-14)

Fuse directional endpoint queries and one mean per current-graph curve. Retain
the existing curve builder, coefficient recipe and eager reference. Raw C inputs
are immutable cosine nodes[K], log nodes[K], weights[K-1,4], curve and its
coordinate partial[B,10K-6], cosine[Q], int32 owner[Q]. B=0..32, K=1..512,
Q=0..1056 (at most1024 light pairs plus32 view queries; public material limits
remain8/4 points and1024 light pairs; public resource queries remain32).
Owners lie in[0,B), cosines[0,1], positive increasing cosine nodes ending at1,
finite increasing nonpositive logs, finite nonnegative weights and finite curve
values/partials. These are raw numerical operations, not resource admission.

Forward returns query[Q,6] = endpoint values, coordinate partials, cosine
partials; mean[B,4] = endpoint means and coordinate partials. Directional values
use the previously qualified subtraction-before-division horizon and log-Bezier
recipe. Mean uses the existing adjacent-pair/carry-odd tree. First VJP contracts
each two-endpoint cotangent left-to-right, then gathers query contributions by
ascending owner/query index with the same fixed tree and adds the mean
contribution. No atomic gradient scatter; zeros and nonzeros retain the existing
acceptance criteria. Immutable coefficient gradients and higher orders are not
admitted. Duplicate owners, grazing/endpoint queries and shared roughness are
mandatory numerical controls.

Caller scratch is8+8*B*max(4*K,Q) bytes; backward uses B independent Q-element query rows in the same bounded
scratch and requires B/owner shape agreement. Outputs and scratch are disjoint
from each other and all inputs. Dense same-device/index float64 (owners int64);
structural/domain errors preserve outputs, arithmetic overflow invalidates them.
CUDA validates on the caller stream before output writes, returns synchronized
status, and exposes append-only provider tails checked before access. No native
heap allocation, cross-call cache, material/renderer partition change or worker
limit increase. Installed-wheel physical, exact CPU/CUDA, first-VJP and bounded
performance checks are required before promoting this bridge.

Nativequery24 operation-only checkpoint: raw values and coordinate/cosine VJP
match the preserved eager reference. Six both-device/full-cap/scratch/alias/stream
checks pass. Initial CPU ABI test used a float-only borrowed-view helper for
integer owners; its failure is retained and the local test helper is corrected.
The separate material-route red demonstrates the operation is not yet used.
Nativequery25 activates one native query group per material call, retaining one
curve per active point and explicit current-call owner gathering. `_curve` and
`_curve_pairs` remain executable references; first-gradient integration and
physical qualification must pass before any performance or acceptance claim.

## Prospective whole-point material arithmetic (2026-09-14)

The next standalone boundary covers one existing Principled point/light pair
with16 independent scalar inputs in the order normal3, view3, light3, albedo3,
roughness1, F0RGB3. Its immutable inputs are one already constructed current-call
curve/coordinate partial, node axes, endpoint means/partials and the selected
log-roughness interval width. Output is BRDFRGB and its3x16 first Jacobian.
This standalone test ABI does not admit new public batches or enter the wheel.

Preserve scale-before-normalization, the fixed six-step normalized-length square
root, the same dot clamp with unmodified first derivative, discrete two-sided
orientation/hemisphere selection and zero inactive values/gradients. Use the
existing GGX/Smith/Schlick plus reciprocal-matte expression and native endpoint
queries; use fixed-operation directional chain rules for this16-input formula,
not a general compiler or higher-order API. Roughness-coordinate partials return
through the existing log interval and roughness derivative. All constants and
inputs must be finite, axes/curve/resource domain already validated; production
integration will additionally require pre-write validation, owned scratch, exact
provider tails and deterministic shared-input VJP reduction. Native stack/local
arrays have fixed extents, with no heap.

Before production integration, require exact forward bytes against the installed
reference on interior, grazing, two-sided, inactive, endpoints, nonunit and scaled
vector fixtures; first-Jacobian contractions satisfy the existing algebra
atol1e-10/rtol1e-9 and two-step independent differences away from branch changes.
CPU/CUDA values/Jacobians must agree exactly. Record any zero/nonzero or forward
byte discrepancy rather than weakening its check. No point/query/pair caps,
mathematical tolerances, resource tables or large-frame claims change here.

The initial complete-point float64 difference check passed at1e-5 but failed
at5e-6 (max error1.2031536e-9). A bounded three-step diagnostic shows the
installed analytic and native directional gradients differ only around1e-15,
while subtraction error grows as the step shrinks. It is retained as an
inadequate finite-difference oracle, not a native repair or a tolerance waiver.
The replacement uses independent90-digit Decimal geometry/normalization,
Bernstein interpolation, limiter, GGX/Smith/Schlick and matte equations over
frozen table constants, at the original1e-5/5e-6 steps and unchanged thresholds.
No installed arithmetic was changed to accommodate the failing comparison.

## Prospective installed whole-point family

Nativepoint27 uses exactly two fixed C specializations: primal and16-input first
Jacobian. Forward-only calls do not compute or reject unused derivative arithmetic.
Inputs are cosine/log nodes[K], current curves/partials[P,10K-6], means[P,4],
log-roughness widths[P], normal/view/albedo/F0[P,3], light[P,Q,3], roughness[P],
and a two-sided flag. Require P<=8 forward or P<=4 gradient-bearing, P*Q<=1024.
A derivative flag controls an output Jacobian[P,Q,3,16]; forward-only returns an
empty Jacobian. Output BRDF[P,Q,3] has the unchanged geometry and material semantics.

First VJP receives that Jacobian and RGB cotangent. Contract RGB in the existing
left-to-right three-channel order; per-point fields gather all light contributions
by the existing adjacent-pair/carry-odd tree. Light input gradients remain per pair.
The adapter returns separate n/view/light/albedo/roughness/F0 gradients so existing
explicit layout expansion owns caller-shared reductions. No higher-order claim.

C validates dense matching dtype/device/index, exact shapes, axes and finite
coefficients/fields, positive widths and roughness, nonzero directions and unit
material domains before writing. Outputs/workspace cannot alias any input or
one another. The shared family workspace query reserves8+8*P*max(1,Q) bytes; forward uses only its8-byte status prefix. Native fixed
stack/local storage has no heap and must be included in measured worker/device
qualification. Arithmetic overflow invalidates outputs and returnsOVERFLOW.
CUDA validates before dispatch on the caller stream, uses no atomics for gradients
and synchronizes only that stream for status. Provider tails remain append-only.
This is local material-call fusion; the renderer's four-point partition and
recomputation order remain unchanged, and larger admission still needs measurement.

Nativepoint27 operation-only CPU/reference and exact CUDA value/Jacobian/VJP
controls pass. Three final raw ABI/limit checks pass after the empty test fixture
was corrected: expand-to-zero can retain zero strides even after contiguous(),
which the existing C ABI correctly rejects. That original failure is retained.
The separate public-material dispatch red remains evidence of missing integration.
Nativepoint28 routes nonempty admitted material calls through the new native
family; explicit empty-layout behavior uses the preserved `_principled_reference`.
The adapter materializes zero-stride singleton arguments/cotangents using the
existing shared helper. Physical, all-input and device regression remains required.


## Integrated material qualification and retained sqrt diagnostic

Nativepoint28 passes broad public, device and deferred regression plus fresh-graph
memory controls. A new four-point/256-light reference-byte comparison exposed a
one-ULP difference in the preserved eager CPU Smith sqrt. For operand
0.9298300710805834, native returns0.9642769680338649, while torch.sqrt returns
0.964276968033865. A90-digit nearest-neighbour certificate verifies native rounding;
complete-pair Decimal values and both original finite-difference steps pass the
unchanged algebra tolerances. Captured geometry, loss pairs, Fresnel and matte are
exact. Historical strict comparison failures and trace source remain preserved.
The broader shared-input test requires exact CPU/CUDA native values and every VJP,
exact zero support versus reference, and the existing algebra tolerances. It does
not use the older library sqrt as a universal bitwise mathematical oracle. The
original12-case forward-byte controls remain intact and passed; no runtime math
or precision changed to fit this diagnostic.

The ABI inventory rejected the candidate workspace spelling. Nativepoint29 uses
`gffx_shading_principled_pairs_workspace`, matching its forward/backward operation
stem. The older unpublished wheel and failure remain historical evidence. This
renames only that new candidate C symbol; no argument, workspace or public Python
behavior changes. Rebuild, installed-member verification and ABI controls are
required before using the corrected artifact.

## Approved grouped renderer boundary (2026-09-14)

A separate shading_principled_groups C/adapter family accepts0..128 point owners
in consecutive four-point groups with a final1..3-point tail;1024 aggregate
point/light pairs remains the ceiling. It shares the existing pair arithmetic,
validation and per-point light-reduction code. The original pairs family retains
8 forward/4 gradient points. No public material rejection routes around that cap.

The first boundary takes already constructed curves/means with the same explicit
axes, point/light layouts and first16-input Jacobian as the pairs family. Native
workspace is8+8*P*max(1,Q) bytes. Structural/domain rejection preserves outputs;
overflow invalidates them. Provider tail and operation ID are append-only. CUDA
uses the existing per-point kernels and caller stream, with no cross-point sums.

The renderer bridge will prepare immutable selected controls once per aggregate,
then use existing at-most32-owner curve/mean calls for chunks. This preserves raw
standalone32-entry limits while amortizing Python preparation and material
dispatch. It is bounded fusion, not a claim that all work becomes one kernel.
Recomputation must save original inputs, preserve each group's field expansion
and shared gradient reduction, and compare complete grouped VJPs with the previous
ordered four-point graph before activation. New aggregate scratch is explicitly
charged in addition to retained image/scene tensors;64MiB material workspace and
all worker limits remain enforced. A raw grouped operation alone is not renderer
or large-frame acceptance.

### Shared reduction attribution and explicit implementation

Groups33 preserved per-point values/Jacobians but changed engine accumulation of
shared albedo/roughness/F0;the9-point failure reached2.7755575615628914e-17.
Reversing input edges reproduces the old composition on the initial controls.
A129-point diagnostic also exposes old ungrouped CPU/CUDA differences in a
caller-created light expand (1.6653345369377348e-16) and shared roughness
(1.1102230246251565e-16). Those failures are retained, not numerical passes.

Groups34 makes the intended reverse four-point group accumulation explicit in a
single renderer autograd node. Each group's original field layout and native VJP
remain unchanged. Backward recomputes bounded cohorts, visits their groups last
first, and adds each contribution into the original shared field or disjoint
point slice. This removes dependency on autograd scheduling of repeated shared
edges. Compare exact results with independently called four-point contributions
summed in that order, and require exact CPU/CUDA values and every library-input
VJP. A caller's arbitrary broadcast graph outside those inputs is not relabelled
as qualified;full renderer lighting integration must separately test its owned
broadcasts. Keep the historical graph comparisons and unchanged algebra bounds.
