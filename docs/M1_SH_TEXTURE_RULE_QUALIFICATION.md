# Conditional SH texture quadrature qualification

This development-only rule is Gauss5 by Gauss5 on a Duffy-mapped triangle:
25 shading evaluations instead of the current cut-batch Gauss5 by Gauss8's 40.
It is a candidate for the orthographic, zero-specular classic material with
band-two SH illumination and level-zero bilinear border texture. It is not an
order reduction for Principled, curved regions, or arbitrary shaders.

## Independent error construction

A triangle lies entirely inside one texture cell. Its affine fields in triangle
coordinates are the unnormalized smooth normal n and the local texel coordinates
s,t. Bilinear albedo is

    A = T00 + s(T10-T00) + t(T01-T00) + st(T11-T10-T01+T00).

With m=n/sqrt(n.n), each colour's lighting is a linear combination of

    [1, mx, my, mz, mx*my, my*mz, 3*mz*mz-1, mx*mz, mx*mx-my*my].

Effective lighting coefficients include the chosen SH normalization, cosine
convolution and classic diffuse factors. The independently supplied coefficients
are treated as exact inputs to this conditional rule. The saved-fixture exporter
implements the FLAME coefficient permutation explicitly; rounding in constructing
those coefficients is outside this conditional certificate.

Under (r,q) in [0,1]^2, triangle coordinates are (r,(1-r)q), and the
integrand is determinant*(1-r)*A*lighting. The positive determinant is twice the
physical triangle area. No positive area is discarded, however small.

For five-point Gauss on [0,1], the error is bounded by

    (5!)^4 / (11*(10!)^2) * sup |f^(10)/10!|.

The implementation uses exact rational constants, algebraic interval nodes and
weights, and directed Decimal60 arithmetic. The tensor error identity is
(Ir-Qr)Iq + Qr(Iq-Qq). Positive Gauss weights of total mass one give the sum
of the two axis bounds. Interval Taylor coefficients of order ten enclose the
suprema over the entire square, not sampled points. The integral certificate
also contains directed rounding in quadrature evaluation and summation.

The same construction is applied to first parameter differentials, including
normal, texel-coordinate, texel-value, illumination and determinant derivatives.
These are material derivatives after pulling the changing triangle back to a
fixed square. Their validity for a complete scene additionally requires correct
clipping, boundary motion, region coverage and chain derivatives. Unknown signs,
normal intervals containing zero, nonfinite fields, or an unsplit texture cell
refuse. No finite-difference agreement or renderer comparison defines the bound.

## Scope of the October 4 experiment

Artifacts are under
D:/devspace/gffit/qualification-20261003/boundary-sh3.
The independent exporter uses exact rational polygon clipping after freezing
binary64 projected geometry. All nonzero regions are retained. Conversion of
piece fields to binary64, FLAME geometry/normal construction, region ownership,
and production texture-cut admission are NOT certified by this export. Thus
this is an integration-rule qualification on exact supplied piece fields, not
an independent certificate of the complete rendered pixel.

pieces-check1.json certifies all 419 exported triangles for saved shape150,
pixel(x23,y7). Sum of image radii is below2.3e-18 per RGB channel; the sum of
first-differential quadrature remainders is below7.3e-15 when every local
primitive-input differential is independently in[-1,1]. For a concrete upstream
parameter these bounds must be multiplied by the maximum magnitude of its local
field differentials; they are not unscaled bounds on FLAME parameter gradients.
The 419-piece count is an independent partition, not a production cursor count.
It requires10475 rather than16760 samples for that same partition (37.5% fewer).
At that pre-integration checkpoint, native batching, complete-pixel limits and
production speed were unmeasured. The native checkpoints below update that status.

partials1.json checks all55 primitive-input partials on each of the three
regions with the largest independent tangent bounds:165 RGB derivative checks.
The largest distance to the independent integral enclosure is below2.34e-16.
The full419-piece worker took34.69s and35.61MB sampled working set; these are
reference-checker costs, not rendering timings.

The experimental binary64 evaluator is separate NumPy code. Its output is
compared against the integral enclosure with the unchanged1e-10 absolute bound
(a stricter screen than adding the existing relative allowance). Its own
floating-point operations are checked by that comparison, not assumed exact.
The campaign's candidate integral distance uses an outward sum of individual
binary64 piece results; it does not establish native ordered-accumulation error.

## Promotion gates

Keep public orthographic opaque rendering guarded and the default installation
unchanged. The explicit native SH-specific rule selector and scoped installed
CPU samples, weights, first VJPs and thin-strip controls are now qualified as
recorded below; existing rules for other shaders remain intact. Complete the
remaining production cut/field admission and broader pixel obligations, including
upstream geometry differentials and boundary transport, CUDA equality/resources,
and frame behavior before creating a filtered target or running another fit.

Keep all existing resource limits, numerical requirements and failure history.
No test success here authorizes Git publication or a resource increase.

## Native integration checkpoint: October 4

The SH-specific rule is now wired through shared native CPU/CUDA code and the
PyTorch adapter. The additive low-level batch selector is `rule=1`
(`GFFX_OPAQUE_BATCH_SH_BILINEAR25`); `rule=0` retains the existing default.
The new rule refuses curved regions. Selecting it does not itself certify an
arbitrary shader. The private orthographic SH candidate selects it explicitly;
the public scene profile remains guarded pending broader qualification.

Backward replay receives a derivative-family mask: bit 1 requests vertex
partials and bit 2 requests cut partials. Unrequested outputs are zero. Tests
check exact equality of requested gradients against replay requesting both
families. This avoids unnecessary seeded clipping walks without changing
requested arithmetic, accumulation order, or the work limit.

The installed candidate is boundary-sh4/candidate3, wheel SHA-256
`b7ecee1de9f83af3d11864fca7fa97ad82d0842b2c3f01eacaf27496258259dc`.
It passes 45 CPU tests, including the native version of the 165 independent
local derivative comparisons. The saved shape150 pixels (23,7) and (24,9)
complete forward and backward within the unchanged limits: respectively
13 calls / 366 blocks / 9150 sample slots and 6 / 187 / 4675. Complete RGB
values differ from independently clipped and integrated references by less
than 3.3e-16. The reference integration enclosure is rigorous for its supplied
local fields; upstream field construction and rounding are comparison-tested.
These are two-pixel results, not full-frame admission.

The frozen `boundary-sh4/gradient-screen1.json` records the separate finite-
difference screen for shape coordinates 0,49,99 at steps 1e-5 and 5e-6 on both
pixels. It compares native analytic gradients with native differences and
independent-reference differences. Only a terminal passing report admits that
finite screen. It is not an analytic remainder bound for all FLAME gradients.
`boundary-sh4/qualification1.json` assembles terminal evidence and its limits.

Candidate1's backward work-cap failure and candidate2's build deadline failure
remain historical failures. `controls2` imported an older artifact after the
candidate2 installation was refused, so it is retained as invalid evidence.
The qualification runner now requires the exact installed wheel identity.
CUDA source compiled, but CUDA runtime, broader states/parameter groups,
production cut/field error propagation and complete-frame execution remain
open before a new filtered target or fit. No default installation was replaced.

## Expanded saved-state CPU screen

`D:/devspace/gffit/qualification-20261003/boundary-sh5/qualification1.json`
records the terminal passing screen on the same installed candidate3 wheel.
Expression, jaw, texture, weak-perspective camera and lighting each use one
predetermined alternating-sign unit direction through that group's parameters.
The states are the saved synthetic truth and saved local150 shape coefficients
with the other groups fixed at truth. Pixels (23,7) and (24,9), two fixed steps
(1e-5 and 5e-6), and separate RGB channels give 40 direction checks / 120 channel
checks, each compared with both native and independent finite differences.
All pass the existing 2e-8 + 2e-6*abs(reference difference) gradient criterion;
maximum error/allowance is 0.612018. Maximum perturbed RGB distance is below
3.643e-15 against independent clipped/integrated references. This is finite
CPU evidence, not every individual coordinate or an enclosure of upstream
floating-point geometry arithmetic.

Each channel uses a separate fresh render and one first-VJP query. The initial
harness reused one forward for three backwards and exhausted its aggregate
clipping-work budget; its failed report is preserved. No counter, work cap,
renderer source, shader rule or numerical acceptance was changed to fix the
harness. Ten harness fixture controls verify state selection and isolated,
deterministic perturbations. All successful workers remain below 24 seconds
and 772 MiB sampled working set.

The overview's static report is an actual evidence graphic. Its old 32x32 target
image only locates the pixels; it is not a new full-frame render or fit. The PNG
is hash-verified through the separate presentation store. Live logs remain
268432669 bytes before/after publication. See static-publication1.json.

Production region/field bound propagation, CUDA runtime and full-frame admission
still precede a new filtered synthetic target and fitting-recovery trial. The
public orthographic opaque guard remains in place.


## Production-corner enclosures and installed CUDA qualification

`boundary-sh6/forward-campaign1.json` passes all1072native-emitted triangles
across the two saved states and two pixels. The observer checks owners,samples
and receipts byte-for-byte against the unchanged installed candidate3 for each
batch. Independent outward-rounded geometry/field construction and explicit
texture-knot excursion terms bound integration on those observed corner domains.
This does not enclose displacement from ideal FLAME or ideal clipping geometry.
`direction-campaign2.json` passes20dense upstream directional comparisons
(60RGB channels) covering expression,jaw,texture,camera and lighting on those
same states/pixels. The largest distance is below1.105e-10. Shape coordinate
propagation and texture-border scope remain separate obligations; these saved
regions use interior texture cells. Historical harness failures remain retained.

`cuda-prepare1.json` performs separately charged offline preparation using only
the CUDA driver to load the exact embedded PTX,without PyTorch resident. The DLL
matches the wheel and its PTX matches the fresh build. No JIT option changes.
Preparation takes43.754s with1008533504B sampled host peak. It is not evidence
for a cold combined PyTorch/JIT start. `cuda-campaign1.json` then passes20/20
installed candidate3 cases with exact CPU/CUDA colour,mask and full gradient
bytes under the existing limits. A diagnostic directional dot reduction may
round differently; it does not change the renderer gradient-array comparison.

Both saved32x32frames fail the old32-face admission in85pixels;the maximum is
293candidates. `visible-owners1.json` proves at least33strictly positive owners
at one fitted pixel using exact rational visibility on observed projected
binary64 input. Hidden-only culling cannot meet that old cap. User approval of
`FACE_ADMISSION_PROPOSAL.md` authorizes an SH-only512-face prototype,retaining
all sample,clipping,scratch,time and numerical requirements. Capacity tests
first fail4cases on candidate3 with2refusal controls passing (`capacity-red1`).
The prototype uses a bounded512-by512bitset and keeps legacy admission at32.
The separate camera boundary-flux path still requires qualification. No new
filtered target,fit,complete-frame acceptance or public promotion follows from
this partial prototype.


The first512-face candidate builds in100.006s/670625792B and passes6analytical
capacity tests plus45CPUregressions. `crowded512-forward1.json` refuses during
the second batch,not at candidate admission. Isolated replay records the exact
parallel-support failure in `tangent-refusal512.txt`: equal normal coefficients
(0.16842548454359729,0.065463104808006989),offsets
-0.057929516621136774 and-0.057929516621136781. The zero support-intersection
determinant is observed at639work units/31registered supports. No production
arithmetic was altered to hide the refusal;no clipping/time/sample cap was
reached or raised. The saved native inputs make a small reproducer available
without repeating the FLAME scene. A robust treatment must retain nonzero
contributions and validate moving constraints before complete images or fitting.


Candidate5121 also passes all6analytical capacity controls on CUDA:
30.964s worker,843206656Bhost,1149952BGPU after separately charged exact-PTX
preparation. Shared-edge diagnosis maps the parallel lines to one topological
mesh edge (vertices4024/4029),constructed in opposite incident-face orders.
The simplified two-triangle test passes already;the full saved continuation
is the actual failing regression (`shared-edge-red2`:1fail/7pass).
Candidate5122 canonicalises those edges only for SH pixels with>32candidates,
but its diagnostic still refuses on a repeated support. Candidate5123 also
recognises exact moving-support identity while classifying known supported
vertices,without snapping coordinates or conflating distinct line offsets.
Its diagnostic completes the previously failing continuation with32regions.
Installed-wheel and complete-pixel validation remain required. Candidate5122
was never built/promoted and its failing diagnostic is retained.


## Terminal bounded checkpoint

`boundary-sh6/qualification1.json` records partial qualification and remaining
complete-image refusal. Installed candidate5123 wheel SHA-256:
89b6ba6d48ff82cc65fe3c130cc333852189cc2904fb67667aada896271d136c.
It passes45existingCPUregressions plus10newCPUcontrols and11CUDAcontrols.
The latter include33-face exact CPU/CUDA samples/owners/first-VJP bytes and
thin-strip2^-40/2^-48 area/shape gradients. The initial thin-strip harness was
invalid at coincident pixel boundaries;its second version incorrectly dropped
rounded near-knot nodes during selection. Both are retained. Corrected tests
use an interior triangle and entire integration regions,with unchanged strict
accuracy assertions. CUDA qualification takes36.204s/1105805312Bhost/1237504BGPU.

The crowded pixel now fails at the actual total sample cap:quad_batch_offer
attempts block513 after512blocks/12800slots,batch15(zero-indexed). See
sample-cap5123.txt and crowded512-diagnostic2-native15.npz. Existing1x2,2x1,
2x2layouts also refuse. This is not an accepted frame. No cap was raised,
no filtered target was created,and fitting was not restarted. Next investigate
unnecessary visibility/texture fragmentation within existing limits;any genuine
budget change needs an explicit reviewed proposal. Camera-edge admission,shape
bound propagation,texture-border support and complete images remain separate.
The independent reference now explicitly refuses unqualified border cells;
11mathematical controls pass,without changing certified interior arithmetic.
