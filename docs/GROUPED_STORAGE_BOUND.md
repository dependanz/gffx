# Grouped material temporary-storage bound

Prospective refinement,2026-09-14. Existing resource limits and public admission
remain unchanged. K is the number of positive angular nodes (at most512),P the
bounded cohort point count (at most128),Q its light count (P*Q at most1024).
All counts below are float64 elements, except the final8-byte workspace header.

The native path computes its prepared curves under a custom Function forward,
without retaining an eager curve autograd graph. Inputs and full-frame output/
selection/light graphs are charged separately by the renderer planner.

| Stage | Simultaneously live angular storage bound |
|---|---|
| _prepare_curve | At most24*P*(K+1):gathered controls,stack inputs,stack output,pieces and concatenation |
| Chunked native curves/query | At most38*P*(K+1):8 prepared coefficients,20 curve/partial outputs,10 scratch for at most32 owners |
| Grouped material invocation | At most48*P*(K+1):8 coefficients,20 original curve/partial lists,20 concatenated copies |
| First VJP | Saved48*P*Q Jacobian plus point/light/cotangent buffers;no angular curve graph survives its forward |

Use64*P*(K+1) angular elements to exceed all three stage maxima,plus64*P point
metadata and64*P*Q forward or192*P*Q backward elements. The point/light term covers
input packing,value/Jacobian,cotangent and partial copies,including previous
iteration partials. The result is8+8*P*(64*(K+1)+64+Q*(192 if backward else64)).
The former128 angular factor was an intentionally loose reserve,not a measured
requirement. Curve scratch comes from native/core/shading.c;output dimensions
come from adapters/pytorch/principled_curve.inc. No native algorithm changes.

Capacity must still fit inside the old single-four-point material reserve and
at most32 groups/128 points/1024 pairs. Charge the actual worst-case admitted
cohort at the renderer's maximum selected pixel count,instead of charging the
superseded eager query graph for every native call. Small/non-grouped1/2/4-point
calls have a smaller peak than the64-factor bound and use the same conservative
bound. Persistent tables,original tensors and complete renderer intermediates
remain separately charged;do not reduce those charges in this change.

Qualification: preserve the installed-wheel CPU32 backward preflight refusal as
red evidence,then require the same workload to execute under unchanged64MiB
planning and1GiB process supervision. Compare exact outputs/first gradients and
measured GPU allocation with the bound,including largest cohort and many lights.
This refinement alone cannot make resident1080p CPU outputs fit64MiB or establish
tiling/backward accuracy. No larger render is admitted without matching timing.


## Whole-frame scratch selection

Storage36 qualifies the refined bound on the PC. CPU32 backward and9 resource
controls pass;22 device/grouped/VRAM controls pass with exact CPU/CUDA scale32
colour,mask and albedo VJP. The full CPU64 backward estimate still refuses
74036456 B against67108864 B. This is retained as the next red baseline.

Storage37 keeps every non-material component unchanged and chooses the largest
multiple-of-four cohort cap from128 down to4 that makes the complete estimate
fit. The chosen cap is passed explicitly through shading to the grouped node and
is fixed for its complete forward/backward. Per-light pair limits can reduce the
cohort further. Point partition and reverse four-point accumulation do not change;
cohort boundaries only control how much temporary work is dispatched together.
No fit is promised when complete outputs or other components already exceed the
allowance. This selection does not change pixel tiling or global loss normalization.


## Prospective call-local scalar curve reuse

A scalar roughness shared by every point owns one identical curve and its analytic
roughness partials. Prepare those once within each _Blocks forward invocation,
release them when forward returns,and prepare them again in backward from the
version-checked saved scalar. Native pair Jacobians still return separate point
roughness derivatives;the existing four-point/shared reduction remains unchanged.
No cache lives on a budget,module,renderer object or across forward/backward calls.
Each subsequent render,including changed roughness,must prepare fresh values.

Only an original zero-dimensional roughness binding qualifies. Point fields are
not deduplicated by value. The bounded native call still receives contiguous per-
point copies of the same curve and width;there is no new raw ABI or owner limit.
The one-owner curve/partial/mean storage is at most20*(K+1)+8 elements and fits
inside the existing64-factor angular margin even at a one-point final cohort.
Prove exact all-input first VJPs across old ordered calls and updated parameters;
then measure performance. This changes redundant preparation,not the material.

## Original input validation

Validate39 checks finite values, common dtype/device and material domains on the
original bindings once at the start of forward and once in backward. Shape,
resource admission, public per-group limits and workspace checks remain in every
four-point layout. No validation result persists across evaluations. Native
validation and first-VJP math are unchanged. A nonfinite final point must refuse
before the first native group dispatch. The transient finite/domain boolean
arrays scale with original inputs, not curve nodes, and fit within the unchanged
whole-frame input/lighting charges. They are released before material preparation.


## Constant roughness shared-curve candidate (2 October 2026)

The renderer-internal shared-groups operation uses exactly one dense row each
for curve, roughness partial, mean and roughness interval width. It retains the
same P<=128, Q<=1024 and P*Q<=1024 limits, point/light inputs, point ordering,
full first Jacobian and unchanged backward contraction. The dense groups and
original pairs operations retain their old shapes. An appended provider field
makes an older CUDA provider refuse this operation before dereferencing data.

Only an original scalar roughness binding selects this path. Its prepared state
belongs to one forward or backward call. General roughness fields keep the dense
path. No curve is approximated, no sample is omitted, and no cross-call cache is
introduced. Each point reads the same curve bytes it formerly read from a copy.

For K angular segments, P points and Q lights, the conservative byte bound is
8 + 8 * (64*(K+1) + P*(64 + Q*(192 if backward else64))). The former bound
multiplied the64*(K+1) term by P. The one-row term retains the existing allowance
for curve construction, control selection, curve/partial/mean/width and their
simultaneous temporaries; those now occur once for the original scalar binding.
All point fields, full Jacobians, cotangents, VJP results, caller resources,
packed geometry and image buffers keep their previous charges. Group capacity
is still bounded by the same renderer reserve,32groups,128points and1024pairs.

This is a candidate until installed-wheel CPU/CUDA, malformed input, old-provider
refusal, exact complete-pixel and resource controls pass. A reduced storage bound
alone does not establish full-frame performance.

The planner additionally takes the maximum with the unchanged dense bound for
up to four points: a window can contain very few covered samples and select the
direct material path. Shared curve savings never remove that fallback reserve.


The opaque179 candidate retains the already charged packed geometry, projection,
normal and horizon-cut tensors only inside a single overflowing pixel's forward
continuation. Subsequent batches reuse those exact tensors instead of allocating
another geometry preparation. The existing geometry reserve remains unchanged.
The local dictionary is not stored on autograd context or in a global cache;
input versions are checked before each batch. Backward keeps the original fresh
preparation per differentiated batch. This candidate requires fresh installed-wheel
byte, gradient, mutation and storage qualification before frame admission.


Opaque180 candidate: forward horizon arithmetic compacts the already selected
faces, then scatters exact coefficients to the same dense output shape. No
selected face or contribution is discarded. Differentiated execution stays dense.
The compact path is used only when all vertices, normals, camera poses, light
parameters and light matrices have magnitude at most1e50, and positive clip-w
is at least1e-50. View differences are at most2e50; transformed directions at
most3e100; dot/error sums and inverse-w-weighted coefficients remain below1e203,
far below binary64 overflow. Thus omitted unselected arithmetic cannot fail its
original finite checks. Extreme or nonfinite inputs fall back to dense execution.
Existing geometry reserves remain unchanged and cover gathered arrays plus dense
scattered outputs. This candidate still requires installed-wheel qualification.


Opaque181 candidate reuses the same packed scene, projection, base material cuts
and normal tensors across internal tiles within one public forward window. The
existing geometry reserve is retained: its four cut-column copies cover retained
base cuts, region horizon cuts and concatenated outputs. It never adds a second
packed scene. Input versions are checked before every tile and after completion.
The local preparation dictionary is not stored on either autograd context;
differentiated replay keeps fresh preparation per pixel/batch. Qualify exact bytes,
release, mutation, gradients and actual allocation before frame use.


### Ordered continuation carry (opaque182)

The internal RGBA carry uses one native call per existing batch. Each of four
channels starts from its incoming float64 carry and adds samples in ascending
order with separately rounded additions. It keeps zero and subnormal values;
there is no batch subtotal or reassociation. CPU uses a volatile double carry;
CUDA uses `__dadd_rn`. Backward replay and its accumulation are unchanged.
The operation admits at most 800 samples, uses 8 bytes of caller scratch and
32 output bytes, covered by the existing 32 KiB continuation reserve. Outputs
and scratch cannot alias inputs. Finite-domain errors precede writes; arithmetic
overflow invalidates output. The append-only CUDA table field is size-checked
so preserved older providers refuse safely. This is not a new general reduction
API or permission to increase per-pixel work, tile storage or worker deadlines.


### Constant forward packing (opaque183)

Within the existing admitted native cohort, constant material forward evaluation
uses the original contiguous point range directly rather than constructing
four-point Python lists and immediately concatenating them. Every distinct
four-point shape/budget obligation still passes `_constant_layout`; native limits
and validation remain unchanged. General material fields and all backward
group construction and reverse accumulation retain the prior path. No new
storage allowance, persistent cache or numerical operation is introduced.


### Conservative forward face selection (opaque184)

For each closed forward window, only kept faces whose screen-space axis-aligned
bounding boxes intersect that window are sent to native quadrature. The existing
selection uses inclusive comparisons; a touching or uncertain boundary is kept.
Finite binary64 extrema are the same extrema used by the native per-pixel test.
A box outside the whole window is outside every contained pixel. Face order and
vertex identities are preserved, so depth ties, sample order, continuation
receipts and the mapping back to original faces are unchanged. There is no area
threshold and no small contribution is removed. The 32-candidate cap still
applies independently inside each native pixel.

Before compact dispatch, all projected vertices retain the native finite and
absolute-coordinate <=1048576 checks, and all kept cut/curve rows retain finite
checks, including excluded faces. Every face index was already validated by
the full _normals face-geometry call. Primitive shape/device/workspace checks
still run. Dense tensors remain covered by the existing geometry reserve;
selected gathers replace larger existing gathers. The per-pixel forward cache
stores this selection only for that same pixel; the shared base keeps full faces.
Differentiated scene quadrature stays dense with its previous accumulation.


Generated scene horizons are pixel-local (opaque185). Multi-pixel windows with
any generated linear or curved horizon use the existing canonical single-pixel
fallback. This avoids changing a neighbour's integration rule through a padded
cut column. Explicit material and native cut contracts, all contributions, and
per-pixel budgets remain unchanged. No extra persistent buffer or new quota.


Host execution profile v4 (opaque186) separates request_pixels (1..256) from
window_pixels (1..16). A request never crosses its row or the existing256-pixel
storage block. The runtime receives an explicit native tile hint and its normal
64MiB planner charges the complete request output before downselecting tiles.
Only call-local preparation is amortized. File buffers, per-pixel quadrature,
canonical backward order, whole-worker deadlines and host/GPU caps are unchanged.
The profile and both limits are part of the frozen frame/anchor identity.


Opaque187 CUDA preparation uses separate lightweight kernels for independent
domain predicates and exact output initialization. They share the existing status
word and write already charged outputs; scratch remains1MiB. Domain failure is
checked before any output write. The geometry and floating-point accumulation
worker remains serial, including first-VJP seed order and continuation receipts.


Opaque188 charges one additional call-local forward horizon coefficient set and
exact boolean crossing masks: faces times ((3*cut_bound+6)*element_bytes +
cut_bound+1). Every local window selects columns by crossing masks, never by
nonzero values. The existing64MiB planner may downselect tile size. Union cut
overflow falls back to local preparation; backward stays fresh/dense.

Opaque tile growth preserves the one-pixel plan's admitted Principled point-block
and group-point capacities. A larger tile is not selected if it would reduce either
capacity. The planner still charges all geometry, outputs and horizon-cache storage;
one/two-point fallback remains available when the one-pixel budget requires it.
This changes forward scheduling only, not shader arithmetic or backward ordering.

One opaque forward request may retain one scalar-roughness curve per eligible
material. Charge an additional 64*(cosine_nodes)*8 bytes per such material, covering
the retained curve, partial, mean and width with the existing conservative inventory.
This reserve is additional to current material scratch and horizon storage. Source
roughness and resource tensor versions are guarded; no cache survives forward or
is attached to an autograd context. Differentiated recomputation stays unchanged.

Forward snapshot reuse retains scene/camera wrappers and borrowed tensor views
within the existing request metadata/input reserve. It creates no new owned tensor
storage. Per-window/per-batch version guards remain; every differentiated replay
and every new request restores and validates fresh bindings. When an admitted
workload is supplied, the otherwise unused sample-budget camera is not constructed.

Native-confirmed uncovered forward samples skip geometry interpolation and shader
setup only. The existing positive-zero shader output is allocated with identical
shape; multiplication by every native weight and both adjacent-pair reductions
are unchanged, including signed-zero behavior. Full scene/native validation still
precedes this branch, and differentiated replay keeps the original path. The new
array replaces the existing shaded array, never increasing the storage inventory.


For sole constant RGB-albedo backward, the existing bounded material cohort may
use direct layout and the same full16-column native Jacobian/full native VJP.
All original four-point layout and backward storage obligations remain checked.
The returned point gradients reduce independently within each original group
using the same adjacent-pair tree, then enter the original reverse-order carry.
No cross-group subtotal is formed. At most128points and32RGB partial rows replace
the former temporary graphs within the existing material/carry storage reserve.
All other derivative targets retain their existing graph composition.


Explicit fixed-albedo replay may share its one scalar material curve across
forward and backward invocations inside that call. A scoped context, reset in
`finally`, checks matching target/roughness storage and versions plus immutable
budget tensor versions. Detached aliases preserve the same borrowed storage;
copied fields and other differentiated parameters do not share this state.
The one prepared curve replaces repeated invocation-owned curves within the
existing material reserve. It is never retained by an autograd context or any
resource, and all native domain/full-Jacobian/VJP checks still run per invocation.

The explicit replay temporarily disables autograd engine multithreading so CUDA
backward callbacks execute within the caller-owned curve scope. The prior setting
is restored on normal return and exceptions. This keeps the replay serial and
changes neither native CUDA streams nor kernel arithmetic.


Scoped fixed-albedo lighting (opaque218): the existing native unit-vector
operation replaces eager normalization only for nondifferentiated inputs within
the guarded replay scope. The one-valid-delta lighting wrapper splits at128points
and retains only its four input fields (normal,BRDF,direction,incoming):12float64
values per point,at most76800bytes over800points. Input slices borrow existing
storage; even charging full copies, the outputs, BRDF VJP and cotangents add at
most72bytes per point plus the native operation's queried scratch for128points.
These replace the old eager lighting intermediates inside the existing
192float64 pair-elements-per-point backward reserve (1536bytes per point).
No curve/cache survives the call; the full material Jacobian and original
four-point reduction/carry remain unchanged. General derivatives, multiple light
samples and invalid-sample masks retain eager integration.
