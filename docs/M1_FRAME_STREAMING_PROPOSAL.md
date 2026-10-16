# Group 3: bounded per-pixel streaming proposal

Status: user approved implementation and qualification of this bounded prototype
on 1 October 2026. Production integration now has finite installed-wheel
qualification; see the production checkpoint below. Original-frame admission remains open.

## Problem and verified scope

Group 2 remains qualified for its declared selected-pixel motions. The next gate
is complete original frames and scale. The pre-integration installed wheel was
69d7913553cf3431f529a093c430288cea8f81b4619fbc6504f949e1b5f83242;
74 installed package members match it.

The original 32/64-square startup frames exceed the 32-candidate limit even
when exact rational clipping removes bounding-box false positives: examined
maxima have 68/42 positive-area triangles. At 128-square the AABB candidate
maximum is 30, but an isolated copy of the shared native geometry implementation
finds 119 capacity refusals among 16,384 pixels: 76 flat-block and 43 curved-block
refusals. This is a geometry inventory, not complete rendering qualification.
The trace agrees with the installed wheel on the saved 760-slot passing pixel
and first refused pixel. Initial build-invocation and instrumentation failures
are retained and excluded from this finding.

Uniform subdivision is not an adopted repair. Of the 119 refused pixels,
66 pass geometry at 2x2 subdivision, 39 at 4x4, and another 10 at 8x8.
Four still refuse at 8x8. One successful 8x8 decomposition consumes 18,813
sample slots in total. These are geometry feasibility results, not colour,
gradient or frame-runtime acceptance. Exact rational visible-fragment counts
are decomposition-specific and are not claimed to be a minimal polygon count.

## Approved bounded prototype

Preserve the current path for pixels that fit. For overflow pixels, stream the
original visible pieces in deterministic order through at most 16 serial batches.
Each batch retains at most 32 integration blocks and 800 sample slots, yielding
at most 512 blocks / 12,800 sample slots per pixel. This explicitly changes the
TOTAL per-pixel work ceiling; it does not increase active storage or hide work.
All attempted/counting/replay work is charged, with no timer resets.

Retain 32 candidate faces, 16 cut columns, 64 MiB active storage, 1 GiB CPU host /
2 GiB CUDA host, one thread, 120-second whole-worker and 180-second watchdog
limits, numerical tolerances, exact reference-mode CPU/CUDA bytes and all history.
A pixel that cannot finish within the finite batch ceiling still refuses.
No contribution-size cutoff, omitted visible piece, approximate occlusion or
relaxed accumulation is allowed. Do not turn a first overflowing partial result
into an accepted pixel. Deterministic forward/reverse ordering and input guards
must survive the continuation boundary. The existing 32-candidate cap remains
an independent obstacle to the 32/64 startup anchors.

## Qualification before full-frame execution

1. Define a caller-owned continuation and accounting contract, including count,
   completion, cancellation, mutation, tamper and overflow behavior. Add failing
   controls before implementation. No hidden persistent renderer state.
2. Check analytical complete pixels and derivatives across batch boundaries;
   include signed and tiny contributions, depth ties, overlapping surfaces,
   linear cuts and curved horizons. Independent references decide accuracy.
3. Build and test a fresh installed wheel on CPU/CUDA, preserving existing
   pixel behavior where the old path fits. Qualify new overflowing pixels and
   their complete accumulated error, output bytes and reverse contributions.
4. Inventory original-frame capacity again only after a justified implementation
   change. Qualify actual full-frame storage, replay, elapsed time and memory.
   Geometry feasibility alone supplies no admissible timing anchor.
5. Retain D297's admission gate. Any change to its 32/64 startup-anchor policy
   requires an explicit recorded decision; this proposal does not authorize it.

This is a bounded prototype, with no guarantee that the batch ceiling, accuracy,
or 120-second frame deadline will pass. It authorizes no Git transaction or
consumer implementation. Keeping the old total per-pixel caps is the alternative;
that would require a more compact integration representation and its independent
qualification before complete frames can proceed.

## Internal evidence

All receipts are under the mapped PC resource directory:
D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque156-20261001

- capacity-diagnostic1.json
- native-frame128-preflight1.json
- native-refusal-trace2.json and native-trace2-manifest.json
- dark-horizon-diagnostic1.json
- frame128-geometry-inventory1.json
- subpixel-capacity-feasibility1.json
- subpixel-eight-feasibility1.json
- exact-fragment-diagnostic1.json

## Prototype checkpoint, 1 October 2026

The native continuation accounts for every visible piece in batches. Fourteen
cursor controls pass. All119 previously overflowing original128-square pixels
complete geometry in at most6batches/3752slots/74blocks; all16265old passing
pixels preserve exact owner/sample bytes. The changed prototype was rechecked
in streamed-geometry-inventory2.json:6.1771258s geometry,9.7085767s worker,
599179264sampled host bytes. This is not a shaded-frame timing measurement.

Nine analytical geometry controls pass, including the exact unit-square area,
moving material at ordinary and signed1e-20 scales, and curved-horizon area and
motion. The initial dense translation test failed: gradient-0.9963750469 rather
than-1. A source-vertex clipping experiment worsened it and was not adopted.
An exact-sign separating-edge fallback repairs the failure without an epsilon
or contribution cutoff; unsafe exponents return unknown. Its signs are checked
against309exact rational determinants and an unsafe-exponent refusal. The
fallback is currently research-only; legacy execution is unchanged. All failed
reports and the rejected implementation are preserved.

Two CUDA controls compare all owner/sample bytes and every selected vertex/cut
first derivative on crowded flat and curved fixtures. They pass exactly, in a
17.6975201s supervised worker with861347840sampled host bytes. This research PTX
is not an installed-wheel qualification. The host protocol separately binds
input bytes/context, validates receipts, shares a deadline, counts prefix work,
rejects cancellation/mutation/replay and exposes only a completed pixel.

Next: integrate the qualified prototype through the native and Torch interfaces,
with production storage planning and ordered forward/backward accumulation;
qualify actual nonlinear pixels against independent bounds, then install and
check a fresh wheel on CPU/CUDA. D297's32/64startup policy remains unchanged
pending the separate128/256proposal; no full original frame has been launched.


The39prototype controls comprise14cursor,9geometry,2CUDA and14host-protocol
cases. All repository tests have authored explanatory pages, and the44dashboard
JavaScript checks pass. These counts are evidence scope, not a completion percentage.
Q26 in devbrain holds the pending startup-size decision. Production integration
is still required; the research probes are intentionally not public library APIs.

## Production integration checkpoint, 1 October 2026 (opaque157)

The native C and Torch continuation API and scene dispatcher are implemented.
Only refused pixels use bounded serial batches; fitting pixels retain the old
path. Forward carries individual weighted samples in order. Backward rebuilds
and releases one batch graph at a time, checks input versions and receipts, and
charges replay work. Background and principal-point transport are applied once.
Queries expose caller-owned scratch (632,960 bytes on CPU; at most 1 MiB).
The storage planner additionally reserves 32,768 bytes per scene frame for
weighted contributions and receipts within the existing active-storage ceiling.

A real motion failure was found during integration: translated varying material
failed finite differences, and constant coverage acquired a spurious derivative.
Retaining the original clipping constraints and using fixed forward topology
repairs ambiguous coincident pixel-corner derivatives in the batch path, without
changing forward coordinates or dropping contributions. Failed candidates and
a corrected raw-cotangent diagnostic are retained in opaque157.

Installed wheel5 SHA-256:
464050e528b5699d1788b7dcc520fcf948cea742a7a5f40391f80fc3ddc35a7b.
All 75 installed package members match the wheel. Final checks: 21 CPU rendering
and compatibility passes; two workspace-query passes after correcting a test
that incorrectly demanded exactly 1 MiB on CPU; 15 CUDA-profile native/interface
passes including closed curved motions and exact CPU/CUDA bytes; and an
independent nonlinear Principled pixel/albedo-tangent interval check on both
devices. The independent intervals have radius at most 1e-12; unchanged
1e-10 absolute plus 1e-9 relative renderer acceptance applies. The actual
colour, mask and albedo-gradient bytes agree across devices. The shader is
smooth on this fixture; this is not arbitrary original-mesh accuracy acceptance.

The repaired research geometry inventory preserves exact legacy owner/sample
bytes on all 16,265 fitting original128-square pixels. All 119 former refusals
complete geometry, still at most 6 batches, 3,752 slots and 74 blocks. The
5.2776152-second geometry loop is not a rendering-time admission anchor.

Evidence is under the adjacent opaque157-20261001 resource directory: source5.json,
wheel-build5.json, final-cpu1.json/xml, workspace-cpu2.json/xml, final-cuda1.json/xml,
final-reference1.json/xml, final-teaching1.json, streamed-geometry-inventory3.json.
The initial final-cpu1 report is deliberately retained as failed (21 passes and
two fixture failures); the narrowly rerun workspace controls pass separately.

Next: independently qualify the original mesh's newly streamed nonlinear pixels
and gradients, then qualify complete-frame storage/cost under D297 admission.
Q26 still requires a decision before replacing the original32/64 startup anchors
with128/256. Four1080p backward scale cases, synthetic gffit readiness and the
actual SIGGRAPH-quality showcase remain. No full original frame is admitted here.


## Original-pixel continuation checkpoint (opaque158, 1 October 2026)

The unchanged installed wheel5 executes three actual original128 streamed pixels:
(59,10), (48,15), and (67,117), covering the first saved refusal and the minimum /
maximum saved streaming work. Colour, mask and per-channel albedo-sum gradients
match CPU/CUDA exactly. The combined CUDA worker took 10.0994501 seconds,
1,295,945,728 sampled host bytes and 32,872,448 peak allocated CUDA bytes.
These are finite execution/device checks, not independent nonlinear accuracy.

The exact visibility reference now has an opt-in disjoint-overlap proof which
reduces the first pixel from 81 to 22 triangles without changing any per-face
covered region. Positive overlaps, including tiny slivers, remain. Historical
reference decomposition and production rendering are unchanged.

An optional whole-pixel reference shares one 256-cell / 90-second work pool and
one basis error allowance across independent pieces, retaining depth10 from each
physical root. Independent roots do not consume imaginary binary grouping nodes;
ordinary partition-cover depth and work accounting remain unchanged by default.
Optional exact discriminant bounds and planar aggregate residual bounds reuse
previously qualified mathematics. A shared frozen angular table avoids repeated
preparation; no result certificate is reused. Controlled integral, derivative,
resource-refusal and table-sharing checks pass. Failed fixtures and trials remain.

The latest independent pixel trial (pixel0-reference6.json) remains UNADMITTED.
It reached its 90-second reference deadline after 238 charged cells. Its last
complete cover, at 236 cells / 129 leaves, had specular width about2.005e-12 and
matte width about1.141e-5, above the unchanged1e-12 basis-width target. These are
conservative checker uncertainties, not measured renderer errors. Three saved
regions show smooth-base remainder dominates. A bounded comparison of existing
25/36/48-sample rules preserves overlapping complete region enclosures but does
not establish a sufficient whole-pixel accuracy/runtime repair.

Next: qualify a better coordinate map / smooth remainder bound on those saved
regions, reusing the Group2 horizon and angular-map work where applicable; then
recheck the complete pixel and broaden to the other selected streamed pixels.
No full original frame was launched. Q26 still needs explicit resolution before
128/256 startup frames replace32/64. M1 and Groups3-5 remain unfinished.


## Starting-resolution approval (opaque159, 1 October 2026)

The user explicitly approved128x128 and256x256 startup frames for the original
opaque mesh. Devbrain decisionD504 resolvesQ26 and restates the progressive
forecast policy with this narrow exception; other profiles retain32/64startup.
The32-candidate cap, numerical accuracy, worker deadlines, memory limits,
two-anchor slower-rate forecast,2xmargin and30-second reserve remain unchanged.
This approval does not claim independent pixel accuracy or admit a full frame.

A bounded continuation compared opposite transverse coordinate orientations on
three identical saved physical regions. Exact triangle permutation preserves
physical coverage;108 point/derivative-through-order2 checks and six complete
region enclosures pass. The first diagnostic incorrectly assumed swapping any
last two vertices reversed the radial parameter; its early identity failure is
retained. The corrected probe preserves the certified radial anchor explicitly.
The resulting matte-bound reductions are small (less than0.3percent), so no
production/checker orientation change or new full trial was adopted. Continue
with a qualified smooth-bound/angular-coordinate repair, not another unchanged
accuracy trial. The installed production wheel and all historical evidence remain.


## Opaque160 checkpoint — 2026-10-01

Independent-checker work only; installed wheel5 and all production limits remain unchanged. Original startup128/256 approval D504 remains settled.

Test-first optional localized smooth remainders, conditional near-horizon linear base with the full residual, and exact polynomial view/light first-slope bounds are qualified by closed controls. The final combined suite passes18tests; the earlier26-test localized-proof suite also passed (overlap exists; do not sum counts). Failed red controls and diagnostic harness/environment failures remain preserved.

Three fresh whole-pixel trials all stop at the unchanged90s reference limit. Last complete matte widths are4.116103220658965e-6,5.713948718330714e-8 and3.085602342028738e-8. Trial3 improves the opaque158 width by about370times but remains above the1e-12 basis-width gate. Its last complete cover has232charged cells/127leaves; the terminal count is234cells, worker92.8180677s and sampled host639021056bytes. These are proof uncertainties, not measured render errors.

Saved-region follow-ups distinguish light-only slope bounds, exact two-role slopes, finite2x4derivative covers and fourth-order boundary corrections. The finite derivative cover removes one monotonicity refusal but does not yield a sufficient bound; it is not adopted for a new whole-pixel trial. Fourth-order corrections improve latest limiting regions by roughly2–6times at substantially higher cost; no full trial is justified by that diagnostic alone. No fullframe was launched and no prior failure was deleted.

The user approved [offline per-piece certification](M1_STREAMED_PIXEL_REFERENCE_PROPOSAL.md), recorded as D505: at most22fresh serial piece workers and one aggregation worker for this one frozen pixel, preserving every per-worker and accuracy requirement while increasing total offline work. Standard shared whole-pixel256-cell/90s behavior remains unchanged elsewhere. The preapproval source snapshot, every terminal report and original proposal hashes remain in opaque160-20261001/checkpoint-source1.json. The approved candidate is frozen separately in opaque161-20261001/campaign-preview1.json; its one campaign stops at the first failure. No partial pixel or frame is admitted.

## Opaque161 approved offline campaign — 2026-10-01

D505 was implemented and qualified with57certificate/profile/controller controls and15shared-path, supervisor and teaching controls. The exact230-file candidate is frozen in opaque161-20261001/campaign-preview1.json (SHA25692e36d3c3f4e5197f36a96d5f710140f4d96c253e004937c88c003292cc14e66). All original input bytes,22 area shares and installed wheel5 are bound to the campaign. The allocation rounds downward with spare summation digits; no radius was increased.

The single approved campaign is terminal: piece index0 passed77cells in49.3677861reference seconds,54.3075477worker seconds and616509440sampled host bytes. Piece index1(face682) hit the90s reference deadline after158charged cells; its last complete157cell/79leaf cover had specular width8.594655251259464e-19 and matte width1.774679607949280e-11, versus allocated basis width2.918184203864486e-15. Worker94.1186445s, sampled host582324224bytes. The remaining20pieces and aggregation did not run. Controller elapsed153.7164749s. One passing piece is historical partial evidence, not pixel acceptance or a reusable certificate for a changed candidate.

The saved enclosure identifies a specific next repair: smooth matte width3.285540964576929e-16 is already below this piece's allocation, while shading-boundary corrections contribute99.9981percent of the unresolved matte width. Diagnose the three saved largest correction regions before another trial; do not simply tighten smooth bounds again or repeat the same campaign. Keep correction signs and all nonzero contributions. D505 authorizes no automatic second campaign or changed-candidate reuse. Group3, full-frame admission and M1 remain unfinished.

## Opaque162 extended offline outcome — 2026-10-01

The user approved the time/work extension in D506.65controls passed before freezing manifest96cc7b79219512c8ad4240d72ae08dcee6badd78df0b3fd00e81a334b981b803. The new profile is explicit and leaves all standard/historical modes and rendering limits unchanged. The campaign started with the previously failing piece index1; all certificates would have been fresh.

The campaign is terminal: index1hit900.0066972reference seconds after1438charged cells; worker903.5510821s,supervisor904.2549791s,peak host596353024bytes. Its last complete1437-cell/719-leaf cover has matte width9.839841759955241e-15 against2.918184203864486e-15:3.371905times too wide,1803.565times narrower than the prior90-second result. Smooth matte uncertainty is4.312223890215293e-20; shading-boundary correction uncertainty remains dominant. No new piece passed;21later pieces and final aggregation did not run. This does not establish a rendering defect or a complete-pixel acceptance.

The next concrete decision is a time-only extension to3600reference/3630worker/3660watchdog seconds while keeping4096cells/depth14/1GiB/one thread,exact input/allocation and all accuracy/rendering requirements. It is pending explicit approval in M1_STREAMED_PIXEL_REFERENCE_HOUR_PROPOSAL.md; it has not been implemented or launched. All earlier failures,partial certificates and dashboard history remain preserved. Group3 and M1 remain unfinished.
