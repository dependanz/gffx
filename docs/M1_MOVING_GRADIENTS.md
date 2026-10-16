# Moving-geometry gradient qualification

Current checkpoint: Group 2 qualified, 1 October 2026.
The selected original pixel passes ten installed CPU/CUDA public comparisons:
normal tilt, principal-point X, world-X, aligned camera-X, and XZ shear with
recomputed area-weighted normals. All five device pairs match exact output and
gradient bytes. The worst endpoint error is 0.488135150207774 of the unchanged
allowance. All13shear pieces are independently audited; the complete shear
pixel has maximum derivative radius8.881363920041670e-13, below1e-12.
The last piece passed in88.3400441reference seconds,252work units,depth10,
within every original cap. Historical failed checker and renderer trials remain.
The corrected reference audit, packed-input/face checks, installed horizon and
linear-cut checks, and grazing/topology/capacity refusals are reconciled.
This closes Group 2 in its declared motion scope; it does not claim arbitrary
camera poses or deformations. M1 remains unfinished. Next is Group 3: reconcile
current-wheel complete-frame evidence and qualify missing frame/scale cases,
including the four remaining1080p backward profiles. Synthetic gffit readiness
and the final SIGGRAPH-quality 3D showcase follow separately.
Evidence: internal opaque155/group2-closure-audit1.json and its hashed reports.

Historical tensor-audit checkpoint (superseded above):
Current checkpoint: opaque155 tensor audit, 1 October 2026.
Group 2 is still open. New independent two-dimensional tests exposed a checker
error: a continuous boundary integral was used in a preliminary bound for a
discrete quadrature sum. The renderer has not been changed by this repair. The
checker now uses the sufficient discrete-line plus continuous-strip identity.
53 new analytic/refusal controls and 107 existing controls pass. The audit covered
51 saved motion certificates: 35 need renewed qualification and 16 avoid that
path. Historical reports remain preserved. Fresh serial requalification is in
progress under the same accuracy, memory, time and work limits; previous affected
passes must not yet be used to close Group 2. All13 physical world-X pieces now
have individual certificates, including fresh complete pieces3/8 in88.03/89.27
reference seconds. Their summed blue derivative radius1.421508e-12 still exceeds
the whole-pixel1e-12 gate (B5). Camera motion, recomputed normals and remaining horizon and
topology checks remain open. Final original-frame/showcase acceptance follows
these numerical gates. No consumer implementation is included.

Historical checkpoint before the tensor-proof audit:
Current checkpoint: opaque155,1October2026.
All thirteen selected original-pixel pieces have independent normal, view and attached-screen references (39 of 39 directions). All six complete faces and all three complete selected-pixel directions pass installed CPU/CUDA comparisons against both independent endpoints with exact device bytes. Every complete-pixel comparison retains 760 samples and six active faces. Opaque155 catalog3 normal gradient radius9.949319051167309e-13 meets the unchanged1e-12 allowance, resolving B4. Actual public original-scene normal tilt with explicit ordered caller reduction and camera principal-point x also pass. Production normal-input Jacobians match exact bytes; the first scalar harness one-bit difference and all timed-out trials remain historical. Actual world vertices, camera pose, recomputed normals and changing-horizon qualification remain separate. Group2 and M1 are unfinished.

Historical checkpoint: opaque126,24September2026. The principal-point horizon
regression is repaired on the fresh CPU/CUDA wheel
`01cb9c2757c1dbece142e5bf2b93124257287aac4ae30dd6585205050b614ce1`.
Group two remains open for actual original-mesh geometry/normal/view directions.
The historical opaque123/124 sections below retain their original failures.

## Historical opaque123 diagnosis

Opaque123, 24 September 2026. Group two remains open. The installed opaque122
wheel and all production source files are unchanged. Its qualified fixed-pixel
accuracy and the thirteen-piece independent reference remain valid.

For a bounded piecewise-continuous shading field F, continuous at the integration
endpoints, let I(theta)=integral from a+a_prime*theta to b+b_prime*theta of
F(x+s*theta) dx. Changing variables gives exactly:

    I_prime(0) = (s+b_prime)*F(b) - (s+a_prime)*F(a).

This identity includes internal resource-knot jumps and opaque horizon flux.
Pointwise automatic differentiation alone need not include those terms. Endpoint
radiance uses independent outward interval Principled algebra with the actual
frozen loss resource, roughness1, F0.04, albedo.2/.4/.6 and light1,1,2. Separate
90-digit Decimal point algebra checks those enclosures. Both endpoints must meet
1e-10 absolute + 1e-9 relative; reference radius remains at most1e-12. This is a
derivative certificate for explicitly translating affine fields, not an image
integral certificate or proof for arbitrary world-space motion.

Four small-knot directions and two moving-visibility directions meet independent
CPU/CUDA bounds. Four horizon directions fail ordinary sampling (worst556.62
times tolerance); existing curved sampling also fails (24.22times). Diagnostic
fourth/eighth-power maps do not repair this. Eight spatial bands pass normals but
not views. Fifteen bands use800slots and pass all four CPU/CUDA directions, with
worst error about0.052times tolerance. This caller-supplied diagnostic subdivision
is not adopted as a general production policy. Some external harness reductions
differ by tiny CPU/CUDA bytes; passing tolerance does not waive exact parity.

A first15-band CUDA probe hit its180-second watchdog. It repeated full geometry
backward for every four shading samples. A first-order chain-rule repair in the
probe collects disjoint sample adjoints and direct parameter terms, then calls
native geometry backward once per colour. Five closed controls pass. CPU values
are unchanged, derivative differences are below2e-16; the fresh CUDA run passes
in23.32seconds, with4.69MB peak allocated GPU storage. The old timeout remains
failed. An overlapping missing-module/import control is recorded; subsequent
drivers refuse any active numerical worker and freeze sources before execution.

Two public-renderer camera-principal-point controls use a plane with constant or
vertex normals, the full original material and an independent enclosure of all
pixel-edge radiance. Both fail accuracy (about99.38and13.96times allowance) on
CPU and CUDA despite exact device colour/gradient bytes. This establishes a real
production integration gap; it is not just a diagnostic-harness discrepancy.

Next: qualify a general bounded sampling/transport repair against these public
scene regressions, preserve original fixed-pixel accuracy, then qualify actual
original-mesh geometry/normal/view directions and topology/capacity transitions.
Full frames/scales, gffit integration and the final SIGGRAPH showcase remain open.
No quotas, numerical criteria, worker limits or history were changed.

Evidence: the internal qualification resource, m1-continuation-20260913/
opaque123-20260924, including qualification1.json, device-comparison1.json,
factored-harness-comparison1.json and the preserved per-trial reports.

## Opaque124: rejected production trials, 24 September 2026

Two fresh native candidates were built within the existing1GiB/120second bounds.
The first used spare capacity for composite ordinary and curved quadrature. The
second also sent constant-normal view horizons through their exact homogeneous
polynomial and shifted more nodes radially. Neither passed the two saved public
scene gradient checks. Candidate1 reduced errors to28.94and1.62times allowance;
its original-pixel accuracy check passed, but this does not admit its gradients.
Candidate2 failed both domains. No failed candidate is the installed production
wheel: the exact opaque122 runtime and wheel have been restored and verified.

Twenty additional CPU diagnostic rules (eight symmetric, six asymmetric and six
graded radial rules) also failed. The resampling helper passes separate closed
area/flux checks. The initial capacity fixture lay on pixel boundaries and used
an inapplicable interior gradient expectation; preserve its six failures. The
corrected dyadic-inset fixture passes all seven cases. Eight native curved
controls pass on candidate1. No new CUDA candidate qualification is claimed.

An exact rational separation of the stored RGB gradients uses
G(albedo)=specular+albedo*matte. On the actual scene trials the specular endpoint
error is below8e-16, while the reciprocal-matte slope remains inaccurate. The
worst specular error across all diagnostic rules is1.85e-11, still within the
unchanged absolute allowance. This localizes these failures to matte angular
integration; it does not prove the precise source of its residual or certify
arbitrary motion. Do not keep tuning radial nodes against this single fixture.
Next isolate the matte angular-resource branches in a small independently
bounded transport diagnostic, then qualify a branch-aware production repair.

Evidence: opaque124-20260924/term-separation1.json, preserved source1/source2,
per-trial reports, production-restoration1.json and qualification1.json. All
sample, worker, accuracy, dashboard quotas and historical failures are retained.

## Opaque125: finite-volume principal-point repair

The scalar loss-branch diagnostic integrates each native polynomial derivative
in log cosine and agrees with independent endpoint transport. This distinguishes
valid scalar loss derivatives from insufficient pixel-interior integration.

Changing camera cx translates the whole image field. Its pixel derivative is
integrated radiance on the left edge minus that on the right; cy similarly uses
top minus bottom. The production first VJP now uses this identity, including
native depth-ordered edge visibility and linear/quadratic shading boundaries.
The ordinary forward image and all other parameter-gradient paths are retained.
See OPAQUE_BOUNDARY_TRANSPORT.md for the public C/Python contract and refusals.

Four independently bounded full-original-material controls (both axes and both
normal domains) pass on CPU and CUDA. Their maximum error/allowance is
2.63287701036676e-6; all image/gradient bytes agree across devices. Enabling the
gradient leaves forward bytes unchanged. Named cameras, shared-batch normalized
loss, mask/background terms, other normal gradients and second-derivative
refusal pass. Original selected-pixel endpoints remain qualified without rerunning
the expensive thirteen-piece reference. Existing geometry/horizon/streaming
controls pass; API validation, provider-tail and public-surface checks pass.

Historical failures include the initial8missing-entry-point tests, one overly
strict reconstruction assertion, one diagnostic-header validation defect and
one compile-signature error. All are retained; the corrected tests pass. The
first ABI test driver also needed an explicit discovered compiler path.

The retained candidate uses Gauss12 per native visible edge segment, at most32
segments and800slots. Boundaries are shaded sequentially before volume buffers;
64scalar slots per pixel explicitly reserve flux/result/VJP storage. Sample,
cut, candidate, worker, GPU and accuracy limits stay unchanged. This is qualified
principal-point transport, not a universal shader integration certificate or a
repair/admission of every normal, view and arbitrary vertex-motion direction.
M1, complete frames/scales, gffit integration and the SIGGRAPH showcase stay open.

Evidence: opaque125-20260924/qualification1.json, device-comparison1.json,
source3.json, wheels3, ABI controls, bounded per-worker reports and dashboard audit.


## Opaque126: original-mesh validation finds a subdivision gate

The qualified opaque125 wheel and all runtime sources remain unchanged. At the
preserved original pixel (row78,column101), joint mesh/camera translation and
positive normal scaling pass exact zero-derivative invariance checks on CPU/CUDA.
A normal tilt agrees with two-step Richardson screening to1.88e-14 on CPU;
finite differences are not an independent gradient certificate. The screen uses
fixed original normal attributes for position motion; it does not qualify
normals recomputed from moved vertices.

Small separate mesh/camera translations and radial scene scaling refuse. A
copied C11 diagnostic with unchanged arithmetic matches installed baseline sample
bytes and the refusal status. The exact failed guard is the32logical-block cap:
radial scaling and camera translation request34blocks, mesh translation36.
All four represented input geometries have13visible pieces in an independent
rational visibility calculation. The native count pass emits15fans at baseline,
18for radial scaling,16for mesh translation and18for camera translation, before
failing; some extra fragments have area below1e-20. These observations localize
subdivision sensitivity; they do not authorize epsilon deletion, a larger cap,
or a claim that every extra fan is spurious. Disabling curved cuts succeeds as a
diagnostic only and is not a shading repair.

Retain the initial radial/translation failures, CUDA's generic image-transfer
error wording, the first diagnostic's spurious baseline line attribution, and
four independent-oracle calls rejected for nonlocal coordinates. Corrected
instrumentation and exact rational origin translation resolve those harness
issues. The next production work is robust visibility/subdivision under the
existing bounds, followed by independently bounded nonzero motion derivatives.
Group two and M1 remain open. Evidence: opaque126-20260924, especially
native-refusal4-values.json, device-comparison1.json and qualification1.json.

All45018per-normal input Jacobian entries match CPU/CUDA bytes exactly. Tiny
normal-scale/tilt scalar differences occur after this boundary in caller Torch
parameterization/reduction; host fsum of identical input arrays agrees. This
localizes the observed byte difference and does not prove nonzero gradient
accuracy or silently waive an end-to-end deterministic parameterization gate.


## Opaque127: bounded clipping and subdivision repair

The selected original-mesh pixel and all six motion probes now complete on CPU
and CUDA without raising the800sample/32logical-block/16cut/32candidate caps.
An outward-rounded orientation predicate proves when two projected triangles
have no interior overlap; shared vertex identities give exact endpoint zeros.
Only proved separation skips subtraction. Uncertain signs retain clipping.
Four frozen represented geometries retain exact-rational per-face visible areas
within1e-12, using655samples each. The predicate also passes259 rational overlap
controls. This is not an assertion that all geometric configurations are admitted.

A new positive-overlap stress case exposed an additional defect: interpolation
from a distant endpoint rounded a2^-60-height triangle away. The shared clipper
now interpolates from whichever endpoint has smaller absolute signed distance,
including the corresponding scalar-dual derivative. No epsilon removes small
area. Heights2^-20,2^-40and2^-60 retain positive area and analytic height
sensitivity1/2on both devices. The original failed test remains preserved.

Fresh installed wheel
`755181d0b600045168328317ce2757e73135bd2037f6472138b92d80193fb279`
passes both endpoints of the preserved opaque120 certificate for original colour
and albedo first-VJP, with exact CPU/CUDA colour/mask/Jacobian bytes. Worst
error/allowance is0.6605933780984398; acceptance remains1e-10absolute+1e-9relative.
The completed thirteen-piece reference was not rerun. Three exact invariances
per device pass. Normal tilt and separate mesh/camera translation have only
finite-difference screening, not independent nonzero derivative certificates.
Recomputed-normal geometry motion is still a separate unqualified direction.

Seven fresh builds stopped at the unchanged1GiB tree limit. A mistaken reading
of Ninja's completed-job log initially attributed the peak to the adapter. The
temporary adapter refactor was restored byte-for-byte. The actual bottleneck
was embedding expanded CUDA code. CUDA noinline separation/clipping helpers
reduce duplication without changing arithmetic flags. Final build9 passes in
100.8012102seconds at1020334080peak bytes; all74wheel members were verified.
The corrected registration test includes134CPU/CUDA entries plus one composite
entry. Both the tiny-area failure and the registration harness failure remain.

Use opaque127-20260924/qualification1.json, source9.json, install9.json,
separation-candidate3-values.json, device-comparison1.json and individual frozen
reports for exact scope. Group two remains open for independent nonzero moving
geometry/view/normal bounds and the explicit end-to-end scalar-byte boundary.
Original frames/scales, gffit supply and final SIGGRAPH-quality showcase follow.


## Opaque128: localize caller reductions and repair inverse-pose VJP order

The original selected pixel now has exact CPU/CUDA bytes for all90,084 public
input-gradient entries:45,018position,45,018normal and48camera-pose entries.
All six explicit pairwise directional contractions also agree. Ordinary caller
Torch reductions retain their own accumulation order; this does not qualify
arbitrary parameterizations or change the numerical acceptance conditions.

The initial matrix-chain probe found exact clip-matrix and inverse-matrix
adjoints but four differing pose entries, at most3.552713678800501e-15 apart.
The repair keeps torch.linalg.inv forward and evaluates its first VJP as
`-(inverse.T @ gradient @ inverse.T)` using four separately rounded products
and a fixed pairwise addition tree per matrix product. Exact rational triple
products independently bound the regression outputs for both precisions and
both contiguous/strided gradients. Eager first/second derivative regression
checks and all42camera tests pass. Nonzero image-gradient accuracy is still a
separate unresolved gate; these inverse controls do not certify the renderer.

Installed wheelc91a4a5d5d5e4fbdb369a67d19724cdec6f11659eb9fc82de888e98f05ed9671
is a verified Python-only update of opaque127 wheel9. All native binaries are
reused byte-for-byte from that qualified PC build. Only camera.py and the wheel
RECORD changed; all74installed members match. A fresh full rebuild first stopped
at1,098,248,192process-tree bytes, so no resource cap was increased and no
unchanged native code needed recompilation. The failed build, four red inverse
controls, two interception failures and one diagnostic output-path collision
remain preserved. The final nine-run affected batch has90passes, nofailures.

Both original-pixel independent certificate endpoints still pass, with exact
colour/mask/albedo-VJP bytes and worst error/allowance0.6605933780984398.
The expensive thirteen-piece independent reference was not repeated. Next is
independent NONZERO normal_tilt/vertex_x/camera_x transport on the frozen original
pixel across angular knots, visibility changes and horizons. Fixed and
recomputed normals remain distinct gates. Group two, full frames/scales,
gffit supply and the final SIGGRAPH-quality showcase remain unfinished.
See opaque128-20260924/qualification1.json and device-comparison1.json.


## Opaque129: first independently qualified nonzero original-piece normal gradient

The first strictly front-facing visible piece (piece0, face5739) now has an
independent normal-tilt certificate, including ordinary angular derivatives and
finite-table jump transport. The outward reference finishes in 111cells and
47.372096seconds, under the unchanged256cells/depth10/90second limits.
Both installed CPU and CUDA quadrature/Principled results meet BOTH certificate
endpoints for RGB and the nonzero first normal derivative. Worst error/allowance
is 0.574691367370; all six value/derivative bytes match across devices. This uses
25native samples and 3876352GPUbytes. The installed opaque128 wheel is unchanged.

Two failed references remain preserved: the first exhausts256cells with matte
derivative width1.4677789994180636789e-10; the second reaches90seconds with width
1.3623183644859243858e-11 after208cells. Mixed value/tangent residual bounds and
signed spatial crossing witnesses have closed-form logarithmic controls. An
exact cyclic triangle map reduces the derivative width by about30.42times at
the same63-cell diagnostic budget; the physical triangle and parameter direction
are unchanged. Coordinate choice changes integration difficulty, not appearance.
The final affected checker suite has111passes and teaching checks34passes.
All red/harness/reference failures remain historical, without relaxed accuracy.

This qualifies ONE fixed-domain piece and ONE normal direction through installed
quadrature and shading, not a complete original-pixel moving-scene gradient.
Other pieces, moving geometry/view, changing horizons and visibility, recomputed
normals and final whole-pixel error accumulation remain open. Next extend the
independent transport reference through those cases before full frames/scales,
gffit supply and final SIGGRAPH-quality presentation. No13-piece value campaign
was repeated. See opaque129-20260924/qualification1.json and the frozen
normal-piece3-reference.json (SHA 67b8f4b0714edd81731ffb90afe9a4a5be8af92ad3eb1190d93bcc9a0ed489df).


## Opaque130: view and attached projected-geometry gradient extension

Exact inventory classifies the original pixel's thirteen visible pieces as nine
strictly front-facing and four curved horizons. Piece0/face5739 now has fresh
independent view_x and attached screen_x RGB/derivative certificates, followed
by installed CPU and CUDA comparisons to BOTH endpoints. Worst error/allowance
is 0.19491566468515636 (view) and 0.06698215127337365 (screen); corresponding
six-output bytes match across devices. Screen translation keeps attributes
attached and includes visibility clipping, but is not world-vertex/camera motion.
Production and the installed opaque128 wheel are unchanged.

The first curved piece8 normal-tilt reference remains unresolved. Three preserved
trials refuse unproved spatial monotonicity, then unproved knot transversality,
then angular segment count at depth10. A controlled interval-range fallback
encloses ordinary derivatives without assuming crossing counts. An exact
unflattened chart encloses parameter transport without the fourth-power map's
coordinate zero. The saved failed region passes this diagnostic, but that is not
whole-piece acceptance. An optional eighth-power map preserves angles, tangents
and Jacobians; its cheap domain preflight still stops at depth10 with nine
segments,102 visited cells,47 accepted domain leaves, box u=[19/64,77/256],
s=[0,1/4]. No new full trial follows this failed preflight.

Next investigate domain subdivision on saved boxes, with no increased segment,
cell, depth, time, memory or accuracy allowances and no dropped contributions.
Then qualify all remaining pieces and outward whole-pixel derivative accumulation,
including world geometry/camera, changing horizons/visibility and recomputed
normals. Group two and M1 remain open. The final checker suite has154passes and
teaching suite34passes. Every red/control/collection/reference failure is
retained. See opaque130-20260924/qualification1.json and frozen per-run sources.


## Opaque131: bounded subdivision investigation (25 September 2026)

A new opt-in partial-monotonicity bound restricts a proved axis to its extremal
edge while keeping the full interval on any unproved axis. Exact polynomial
controls cover both axes/orientations, interior extrema and tiny tangents. An
initial shared-contract regression caught extra evaluations for legacy callers;
explicit opt-in restores their original behavior without changing the old test.
Final affected controls:172passes; teaching:34passes.

The original piece8 normal-tilt horizon remains unresolved. Partial monotonicity
alone repeats power8's102cell/47leaf/depth10/9segment refusal. A three-parent
diagnostic records angular bounds, directional hints and both possible child
pairs. Quarter-near-horizon splits reach87cells/40leaves before depth10 refusal;
harmonic horizon-distance splits reach136cells/66leaves, but still9segments at
u=[8/83,1/10],s=[3/8,1/2]. Exact partitions keep every contribution. Graded
splits are explicitly refused for full integration pending integrator
qualification. No full accuracy trial ran after these failed cheap preflights.

Next inspect angular-knot-aligned partition placement on the saved regions.
Certify physical coverage/Jacobians/tangents and charge every actual evaluation;
keep8segments/256cells/depth10/90seconds and renderer/worker limits unchanged.
Require domain feasibility before qualifying and running the integrated method.
These failures concern reference feasibility, not demonstrated renderer error.
The previously qualified original value and piece0 normal/view/screen evidence
are retained; all runtime and installed opaque128 wheel bytes are unchanged.
See opaque131-20260925/qualification1.json and frozen per-run sources.


## Opaque132: knot-targeted cuts and curved-boundary evidence

Exact nonmidpoint domain cuts now have independent integration controls: full
area, direct parameter derivatives, moving measure and jump transport, in both
axes including a positive2^-60 sliver. Invalid cuts and exhausted cell/depth
budgets refuse. Guidance is bounded to14field evaluations per suggestion and
cannot admit a cell. Existing default integration remains unchanged. Final
affected suite190passes, additional curved-root controls3passes, teaching34passes.

Three saved parent regions show that a centre-line knot target is not a
constant-knot edge across the whole region. The fresh power4 piece8 normal-tilt
domain preflight stops at159cells/77leaves/depth10 with9segments. It records
81guidance calls/1134field evaluations under the original time limits. The
original knot-targeted full accuracy profile is explicitly refused until its
domain cover is qualified. No full accuracy trial ran.

A new smooth-field root primitive certifies strict radial monotonicity and
endpoint bracketing, then encloses the root and implicit parameter/transverse
derivatives. Closed quadratic roots in both orientations, uncertain-topology
refusal and tiny-root/tangent retention pass. On three fixed transverse slices
of an expanded radial span around the saved failure, the same knot has disjoint
root intervals near0.0831376,0.0843267,0.0856046. All three parameter and
transverse derivatives are nonzero;51field evaluations total. These are proved
slices, not a complete curved strip or high-order quadrature map.

Next qualify a knot-following strip over the whole transverse interval: root
existence/uniqueness, boundary entry/exit, complete moving Jacobian, mixed/high
spatial derivatives and parameter flux. Verify closed curved-domain integrals
before original domain feasibility and any accuracy trial. Keep8segments,
256cells/depth10/90reference seconds and all worker/rendering limits unchanged.
Production and the installed opaque128 wheel are unchanged. Earlier value and
piece0 normal/view/screen certificates and every failed report remain preserved.
Group two and M1 remain open. See opaque132-20260925/qualification1.json,
knot-domain1-reference.json and contour-slices1-reference.json.


## Opaque133: a certified knot-following region

The independent checker now certifies a unique interior root uniformly over a
transverse interval using strict radial monotonicity and uniform endpoint signs.
It refuses boundary touches/exits and unproved topology. Bounded interval Newton
only tightens a proved enclosure; an approximate point never replaces the root.
Implicit Taylor coefficient isolation encloses spatial orders0..10 and their
first parameter derivatives. The caller must establish field smoothness.

Closed square-root coefficients in both orientations, complete curved-domain
integrals, varying physical Jacobians, unequal boundary traces and positive
2^-60 strips pass. The strip map retains moving area and parameter transport;
continuous shared-edge terms cancel, while unequal traces retain their flux.
Both Decimal and the pinned Arb backend pass these controls. No transport term
may be added twice when a moving strip map already supplies that term.

For original piece8 normal_tilt, the expanded radial span around the saved
opaque132 failed box is certified across its entire transverse interval in one
region,32 field evaluations and1.1225429000041913 reference seconds. This proves
uniform existence, uniqueness, interior topology and order10/mixed root jets for
that local region. It does not certify every knot, the full piece, shading
integration, CPU/CUDA gradients or final images. Boundary entry/exit splitting
outside this interior-root scope remains unqualified.

Final selected240 passes:199 affected checker,6 compiled-backend controls,
34 teaching and1 original-region diagnostic. All250 outcomes retained, including
5 expected test-first failures. Production and installed opaque128 unchanged;
no full accuracy trial ran. Next connect these charts to bounded angular branch
integration with complete coverage, physical Jacobian and parameter flux, then
establish full-piece domain feasibility before a new accuracy trial. Existing
8-segment,256-cell/depth10/90-second and worker/rendering limits remain unchanged.
Group two and M1 remain open. Evidence: opaque133-20260926/qualification1.json
and original-region1-reference.json, with their frozen producer snapshots.


## Opaque134: actual shading in a moving angular band

A certified two-knot chart now preserves positive pointwise width even when
separate root ranges overlap. Strict radial monotonicity and the mean value
theorem bound that width; only value intervals are tightened, never derivatives.
Analytical polynomial, logarithmic and reciprocal integral controls pass with
moving nonlinear coordinates, full physical Jacobians and changing outer area.
The pinned Arb backend also passes. Boundary flux carried by the map is not
added a second time.

The saved original piece8 region has adjacent view branches at knot70/71 and
one uniform light branch487. Its actual Principled specular/matte values and
normal-tilt derivatives integrate in5cells/depth2,8.616416699951515reference
seconds and1566root-field evaluations. The maximum basis derivative radius is
3.71065840771937974094392062703049637612378864063279547642e-13.
The worker takes10.079546799999662seconds with97587200peak sampled host bytes.
These are local-band bounds, not a whole-piece or installed-gradient admission.
Root-field calls do not count the additional final mapped-input evaluations;
all calculations are included in the measured time and worker limits.

A guarded full-piece domain-only experiment accepts19curved parent cuts but
still refuses at155cells/75accepted leaves/depth10 with9view/1light segments.
It records79curved attempts,60straight guidance calls and1419guidance field
calls. Parent partitions preserve complete coverage; both children are charged
and must meet the eight-segment limit. Boundary touches/exits and uncertain
brackets refuse. No full accuracy trial ran and the full-trial guard remains.

Final selected246passes:205checker+5compiled controls+34teaching+2original local
checks. All253outcomes retained, including3test-first failures and1domain refusal.
Production and opaque128 wheel unchanged. Next diagnose boundary entry/exit and
placement at the saved unresolved parent, qualify the resulting complete cover
and multi-branch integration, then attempt whole-piece accuracy under unchanged
limits. Group two and M1 remain open. See opaque134-20260927/qualification1.json,
band-integral1-reference.json and curved-cover1-reference.json.


## Opaque135: endpoint repair, balanced subdivision and genuine entry events

Opt-in tight_endpoints refines a root bracket only after proving monotonicity
across the whole transverse edge. It preserves interval root and mixed derivative
semantics, refuses uncertain interior crossings, and leaves the old default path
unchanged. Closed controls pass for both radial orientations. The first repaired
preflight reduces the refusal point to123cells/59leaves, but retains the same
9segment/depth10 failure. A bounded116-evaluation diagnostic checks all ten knot
targets on its exact depth-nine parent; none certifies a uniform interior root.

Curved preflights now use exact transverse midpoint cuts, retaining radial knot
guidance. Both children remain independently checked and charged. This removes
the former depth failure; a fresh preflight proceeds until255cells, where the
next two children would exceed256. It retains50accepted curved cuts and stops.
This is an unresolved complete cover, not a domain or accuracy certificate.
Neither cell nor depth limits were raised. Both preflight refusals are preserved.

The new knot_edge_events primitive proves strict whole-box monotonicity in both
coordinates, classifies radial-edge endpoint signs, and encloses any unique
transverse crossing and its parameter motion. Closed quadratic roots/derivatives
pass; uncertain contacts, folds and tangencies refuse. On the saved original
parent, knot70 enters the lower radial edge at approximately0.5758290948915847,
with normal-tilt derivative approximately-0.48572208499714394. The upper edge has
no crossing. This certificate takes25field evaluations/0.7812639999901876seconds.
It does not yet provide a clipped moving-domain chart or its integral.

Next construct and qualify the entry/exit-clipped chart, retaining the moving
transverse split, radial width, physical Jacobian and parameter flux. Verify
closed integrals and exact coverage before using it to reduce full-piece cover
cost. A midpoint or interval endpoint is not a substitute for the certified
moving crossing. Preserve all8segment/256cell/depth10/90second limits and the
original full-trial guards. Original local-band and earlier whole-value/piece0
certificates remain frozen; production and opaque128 wheel unchanged.

Final selected249passes:209checker+4compiled controls+34teaching+2diagnostics.
All259outcomes retained:253pass/4expected test-first failures/2coverage refusals.
No full-piece accuracy trial ran. Group two and M1 remain open. See
opaque135-20260928/qualification1.json, curved-cover1-reference.json,
curved-cover2-reference.json, parent-targets1-reference.json and
original-edge1-reference.json.


## Opaque136: complete entry-following mapping, renewed coverage validation

KnotEntryChart now represents the entire single-lower-entry parent as a
pre-entry slab and two post-entry strips. Both the transverse entry position
and radial knot boundary retain their parameter derivatives. Each returned
Jacobian includes moving transverse extent and radial width; the caller
multiplies by the full physical Jacobian. The collapsing entry edge is allowed
without dropping its positive interior wedge. Root value intersections use
proved topology; no derivative coefficient is discarded.

Closed full-area, first-moment and unequal-color integrals pass with changing
outer measure, including a positive2^-60 scale. A genuinely nonlinear inverse
root also matches closed square-root antiderivatives. One early compiled control
refused because the auxiliary bracket reached a zero radial derivative; the
repaired bracket stays above zero. That failed report is preserved. Unsupported
crossing topologies and out-of-unit chart coordinates still refuse.

On the saved original knot70 parent, all three charts pass in both quadrature
directions with order10 spatial and first-parameter mixed physical-Jacobian
bounds. Their conservative view/light segment counts are7/1,6/1,7/1.
This local map uses110root-field calls and1.5192938999971375reference seconds;
additional physical-map evaluations are included in total measured time.

The fresh complete-piece domain preflight accepts2entry-following partitions
and49interior curved cuts but refuses after256cells. It records20entry attempts,
129curved attempts,77straight guidance calls and2113guidance field evaluations.
Each entry split charges four binary-tree nodes and three leaves at depths
one/two below its parent; no depth, cell or angular allowance was increased.
A partial cover is not a certificate. entry_knot remains guarded as explicitly
curved-domain-preflight-only. No full-piece accuracy trial ran.

Next reduce complete-cover cost using qualified coverage-preserving scheduling
or additional edge topologies, guided by the saved frontier and refusals. Every
new map needs closed integral and moving-Jacobian/flux controls; a full original
accuracy trial still requires complete domain and integration qualification.
Production/opaque128 wheel and all earlier certificates remain unchanged.
Final selected253passes:214checker+4compiled controls+34teaching+1original map.
All264outcomes retained:259pass/3expected initial failures/1auxiliary-bracket
refusal/1coverage refusal. Group two and M1 remain open. Evidence lives in
opaque136-20260928/qualification1.json, original-map1-reference.json and
entry-cover1-reference.json with frozen producer snapshots.


## Opaque137: complete angular coverage and isolated integration-bound failure

Whole-piece radial monotonicity permits direct angular bands in the unflattened
horizon coordinate. MonotoneAngularCover certifies a positive radial derivative,
a fixed zero horizon and a strictly decreasing outer edge. Ordered levels give
unique radial roots; outer-edge events split clipped bands into complete smooth
charts. Both event motion and all implicit-root mixed coefficients remain in
the Jacobian. A balanced band tree and its exit splits charge every internal
node and leaf. This is a coverage construction, not an accuracy exemption.

The fresh original piece8 normal_tilt cover passes: 68bands,
72charts, 143charged cells, depth8,
four outer-edge exits, 1034root-field evaluations and
3.4847316999803297reference seconds. Every chart stays within8angular segments.
The exact-integral controls include moving measure, unequal color traces,
nonlinear inverse roots and both-boundary exits. A2^-60 angular band has a
strictly positive range-integral certificate; its earlier one-cell Gauss claim
failed due to loose implicit-root derivative bounds and remains a failed report.

The first local actual-Principled integration on chart32 refused an Arb series
square-root tangent denominator. The saved discriminant is strictly positive,
but its broad series output constant included zero. Intersecting that constant
with an independently directed scalar square-root enclosure repairs the
calculation, retaining all spatial coefficients. Closed order2/order10 and
first-parameter coefficients, existing adapter controls and all saved original
calculation stages pass. Production and the installed opaque128 wheel are
unchanged; frozen historical checker producers remain authoritative for their
old certificates.

The second local integration reaches a complete eight-leaf enclosure but fails
its error tolerance at the remaining depth3. It uses15quadrature cells,
157total charged cells (143cover+15-1), 7497root-field calls and
42.500027899979614reference seconds. Specular value/derivative widths are
5.03851810350179498216641743670838645595769384126896236E-15 / 3.78521256270266139638701050215171518699751614658737597E-14, within the local criterion. Matte widths are
0.00000810379546013405107288779208707793548445480141154065584353698 / 0.000165014985615362585874764215675504619794320442446102624941543, unqualified. The dominant error is the smooth
matte base bound, not coverage or the separately retained jump correction.
No complete-piece accuracy trial or installed CPU/CUDA gradient comparison ran.

Next inspect the largest saved chart-integral2 leaves, separating the angular
loss factors and moving-Jacobian high/mixed bounds. Qualify a tighter correlated
angular-coordinate or equivalent analytical bound with independent integral
controls before another local integration. Preserve the143-cell cover charge,
chart depths and remaining global budget; do not gain capacity by relabeling
partition work or discarding positive bands. Complete integration remains a
prerequisite for a fresh piece8 accuracy certificate and later group-two cases.

Final selected 284passes; all324outcomes retained
(312pass/12fail). Evidence and exact producer
snapshots: opaque137-20260928/qualification1.json, original-cover3-reference.json,
high-controls1-reference.json and chart-integral2-reference.json. Limits and
acceptance criteria remain unchanged. Group two and M1 remain open.


## Opaque138: matte bounds tightened; dashboard orientation repaired

The independent checker now offers a centered power evaluation of the SAME
cubic in log cosine. Its constant/linear control trends cancel algebraically
before outward interval evaluation. An exact local midpoint is held constant
within each evaluation; translation is a polynomial identity for every anchor.
All spatial and mixed coefficients remain, and the coefficient cache is bounded
to128 entries. The experimental original-chart path uses this form for both
the smooth base and ordinary residual. The horizon segment, knots, jump model,
production renderer and installed opaque128 wheel are unchanged.

Independent constant/cubic closed coefficients through order10 and first motion
pass on both interval backends, including extension outside the middle segment.
The final selected checker set has225 passes;34 teaching checks and one saved
factor diagnostic also pass. All300 pytest outcomes (289pass/11fail) survive:
four test-harness failures from an unsupported Jet.exp, four expected missing
method failures, and three refused local integrations. No full-piece trial ran.

The final chart32 diagnostic still refuses at depth3:15 quadrature cells,
157total charged cells,8leaves,40.50045609998051 reference seconds. Matte
value/derivative widths are1.2127063071907184e-13/1.0800940049996594e-12,
versus8.103795460134051e-6/1.6501498561536259e-4 before this batch.
The derivative width is still8.01% above this local1e-12 width criterion.
This is NOT an8% estimate of work remaining or a complete-piece tolerance.
The largest leaf is [0,1]x[7/8,1]; its transverse high/mixed view-loss and
physical-Jacobian bounds remain broad. The saved factor diagnostic evaluates
all nine stages and both integration directions. Next qualify a correlated
angular coordinate (or equivalent tighter enclosure) on this saved leaf before
another local trial, keeping all complete-cover cell/depth charges.

Dashboard: /testing/ now starts with the finite five-group M1 checklist,
current blocker, scope, and curated existing visual examples. Raw engineering
checkpoints show their text; technical records and retention inventory use
disclosures. All old history remains. See M1_DASHBOARD_UX_AUDIT_20260928.md.
No new final-showcase or complete-gradient acceptance is claimed.


## Opaque139: local moving-gradient integration passes

The previously failed original piece8 normal_tilt chart32 now passes using
an implicit angle-coordinate inverse on its certified interior band. All
spatial and first-motion coefficients of the inverse and physical Jacobian
remain. The view cosine is exactly the affine target angle in this coordinate;
its zero transverse and motion derivatives follow from the inverse identity,
not a threshold or discarded contribution. An analytic radial partial is
checked against independently seeded original-field differentiation.

Exact area, weighted angle, nonlinear inverse coefficients through order10,
first motion and a 2^-60 angular scale pass on both interval backends. The local
Principled integration uses 13 quadrature cells / 155 total charged cells / seven
leaves, with cover depth7 and at most three extra depths. It takes
49.577026599959936 reference seconds (50.65853529999731 worker seconds,
104624128 sampled peak host bytes). Matte derivative width is approximately
1.869125024593e-13, below the unchanged local 1e-12 width criterion and about
5.8 times tighter than opaque138. This is a local pass, not a complete-piece
error allocation, whole-pixel gradient or installed CPU/CUDA acceptance.

Extending angle inversion to every edge chart fails the nonlinear closed-area
control at its unchanged127-cell allowance; that candidate and failure remain.
The final combined cover uses angle coordinates only on positive, fully interior
bands and preserves the existing radial map at horizons and edge exits. Exact
complete-area/first-moment/jump-motion controls pass. Fresh original coverage
contains62 angle charts and10 radial charts,143 charged tree cells/depth8,
1032 root-field evaluations and3.502619999984745 reference seconds. Coverage
is complete; shading accuracy over all72 charts has not been established.

277 selected Python checks pass. All295 outcomes (280pass/15fail) remain,
including missing-method baselines, an interval/Jet dispatch implementation
error and the rejected all-angle integral. The passing local producer is frozen;
its map_angle implementation is retained unchanged in the final source.
Next qualify shared complete-piece error/work accounting, reduce inverse/root
calculation cost if needed, then run one fresh complete-piece reference only
when justified. Never reset90seconds/256cells/depth10 per chart, add up local
passes as a whole-piece certificate, or omit small positive contributions.
Production sources and the installed opaque128 wheel remain unchanged.


## Opaque140: shared integration accounting and inverse cost

The adaptive square checker now accepts a caller-certified disjoint chart cover.
It uses one retained-leaf sum for every value and first derivative, one adaptive
queue, one clock/checkpoint and one work counter. Binary-tree internal nodes are
charged once and chart depths are inherited. Malformed depth accounting,
missing channels, shared deadline exhaustion and sums of individually acceptable
but collectively excessive uncertainty are refused. Geometry and Jacobians
remain separate caller proof obligations; depth weights alone do not prove them.
Exact combined angle/radial area, unequal-band integral and moving-measure
controls pass, including2^-60 measure with proportionally tightened tolerance.
The initial exploratory31-cell tiny control refused; the same accuracy passes
under the existing standard256-cell/depth10 profile. No operational limit changed.

Inverse solving now omits unrelated light and half-vector cosine calculations.
Every saved coefficient endpoint through order10 and first motion matches the
original helper on Decimal and Arb. Fresh chart32 integration has identical
value/derivative endpoints, retained leaves and error accounting to opaque139:
13 quadrature cells/155 charged cells/10018 root-field evaluations. It took
42.46408969996264 reference seconds versus49.577026599959936 previously, a
14.3% observed reduction, not a guaranteed whole-piece speedup. The installed
opaque128 wheel and production renderer remain unchanged.

Initial full-chart diagnostics for horizon0, interior32 and exiting-edge64
complete all27 preparation, high/mixed derivative, residual and transport stages.
These are diagnostics, not integral acceptance. The shared engine is qualified
on independent closed controls but has not yet been connected to the original
Principled chart callbacks. Next connect those callbacks with separate per-chart
state, the full moving measure and residual/jump terms, preserve the original
single90-second construction-plus-integration deadline and143-node cover charge,
then qualify the complete piece. No full-piece accuracy trial ran in this batch.


## Opaque141: complete-piece connection; deadline failure retained

All72 original piece8 normal_tilt material charts now feed the shared adaptive
integrator. Each chart owns its extension, prepared angular bounds and residual
cache. Interleaving horizon0, interior32 and edge64 preparation leaves each
chart unchanged. Center, radial/transverse order10, residual and transport
outputs exactly match the hash-pinned opaque140 implementation. Construction
is inside the same90-second deadline; the143-node cover tree is charged once.
No production renderer or installed wheel changed.

One fresh complete-piece trial stopped at90.05316199996741 reference seconds
(91.27772670000559 worker seconds;96976896 sampled host bytes). It retained28
initial chart roots, stopped during chart28 (the29th chart), charged100cells
including71internal cover nodes, and made21403 inverse-field evaluations.
It produced no complete enclosure and no accepted piece certificate. The trial,
frozen producer and all49 outcomes (46pass/3fail) are retained; two failures
are expected missing-API baselines and one is this real deadline failure.

The initial-bound investigation on horizon0, last-completed chart27 and local
chart32 completes27stages. Across those initial regions, transverse high/mixed
smooth-shading and physical-Jacobian enclosures dominate the separate residual
and jump corrections. On chart27 the initial matte derivative width is about
3.66449e16; this is a conservative enclosure, not an observed rendering error.
The angle-coordinate view factor already has zero transverse/motion terms,
while light/half-vector and moving-area bounds remain very wide. Thus a speed-only
change is insufficient evidence for another full trial. Next tighten the
transverse horizon/inverse and physical-field bounds on these frozen regions,
qualify the repair against independent exact controls, then reassess complete
initial-cover cost/error under the same256cells/depth10/90seconds. Keep all
residuals, jump terms and tiny contributions. No aggregate campaign ran.


## Opaque142: exact horizon cancellation; three local gradients qualified

An explicit checker-only quadratic-square-root candidate forms its Taylor
numerators in exact rational algebra before outward interval evaluation.
It retains coefficient motion, coordinate motion and order10 mixed terms,
proves positivity over the complete interval including its quadratic vertex,
and keeps the existing8192bit polynomial budget. No contribution is dropped
by size. Default checker arithmetic and the installed renderer are unchanged.

Independent moving-square, nonlinear-composition, tiny-positive-curvature,
parabolic area/complement and moving-boundary controls pass. Original-mesh
point derivatives through order10 agree with the separate original algebra.
Default curved controls and frozen original-stage equivalence also pass.
On saved charts0/27/32, whole-width transverse high/mixed smooth-shading bound
magnitudes fall by approximately23orders; this reduces proof overestimation,
not measured rendering error. The stage diagnostic is slower (3.15/5.19/5.06s
versus1.87/3.90/3.81s), so speed and complete-piece cost remain separate gates.

Fresh local Principled integrations all pass with residuals and conservative
jump uncertainty retained. Chart0 uses3quadrature/145charged cells; charts27
and32 each use5quadrature/147charged cells. Limits remain256cells/depth10,
90reference/120worker/180watchdog seconds,1GiB and one serial heavy worker.
Exact times, radii, test counts and frozen source hashes are in opaque142
qualification1.json and local-chart*-1-reference.json. All test-first failures
are preserved. No new complete-piece or aggregate trial ran. These separate
local passes cannot be summed into a new complete-piece certificate.

Next connect the explicit tighter-discriminant candidate to the shared
original-material driver, then assess complete-cover cost and summed error
under one unchanged budget before a fresh whole-piece qualification. Group2
remains open, followed by full frames/scales, gffit supply and the actual
qualified SIGGRAPH-quality showcase. No runtime, wheel or Git publication.


## Opaque143: shared exact-bound repair connected; runtime still refuses

The explicit tight_discriminant selector now reaches all72 original piece8
normal_tilt material charts and the shared complete-piece driver. Frozen
opaque142 center, high/mixed derivative, residual and transport endpoints match
exactly on charts0/27/32 after interleaved preparation. One construction-inclusive
clock,256cells/depth10 and the global retained-leaf sum remain authoritative.

A first fresh combined trial hit90.0008982999716reference seconds after21roots,
93charged cells and16317inverse evaluations. It produced no complete enclosure.
A qualified invocation-local64entry exact horizon-result cache then preserves
all input coefficients, motion/order, arithmetic implementation and precision in
its key. Stored endpoints are immutable; callers receive separate Jets. Exact
controls cover mutation, eviction, backend/precision changes and failed calls.
The connection check fell from11.76 to7.71seconds, an observed local improvement.

A second fresh trial still hit90.00058820005506reference seconds:34retained
roots,106charged cells,25958inverse evaluations,93452cache hits/97misses and
64retained entries. Worker time91.19724549999228seconds; sampled peak101593088B.
No complete enclosure exists, so combined accuracy is not measured and runtime
qualification fails. All common completed initial trace rows match exactly.
Both trials and all failures remain preserved. No numerical criterion changed.

A bounded chart27 initial-cell profile identifies map_angle/implicit inverse
and normalized-field work as the dominant remaining cost:4.80 of6.12profiled
integration seconds in map_angle,3.91 in mapped_angle; these nested times must
not be added. The expected one-cell accuracy refusal retains the local error
bounds. Profiling includes instrumentation overhead and is not runtime acceptance.
Next reduce repeated normalized-field/implicit-inverse work, test against frozen
high/mixed endpoints and independent controls, then reassess a fresh complete
cover under the same shared budget. No per-chart budget reset or sum of old
local certificates can qualify the complete piece. Group2 and M1 remain open.


## Opaque144: exact checker reuse advances the complete trial to64of72regions

Prepared transverse inputs preserve all original field/radial-partial coefficients
through order10 and normal/view/screen motion. Newton brackets use only their
required value and first spatial slope; full mixed recurrence and final moving
Jacobians remain unchanged. A64entry cache shared across all material charts keys
complete input coefficients, chart/bounds/evaluator and arithmetic context, stores
immutable endpoints and returns separate Jets. No threshold drops any term.
Frozen center/high/mixed/residual/transport endpoints on charts0/27/32 remain exact.

The first fresh trial reached56initial regions/128charged cells/42160inverse calls
before90.00185270002112reference seconds. The material-cache trial reached64of72,
136charged cells/47261calls before90.0004683000152seconds; worker91.46227519999957s,
105119744sampled host bytes. It had4524material-cache hits/2952misses/64entries.
Neither has a complete enclosure; combined error is not measured and both runtime
trials failed. More completed initial regions is not a percentage of M1 completion.
The prior opaque143 trial reached34regions within the same limits.

One proposed high-order spatial-only shortcut was rejected as an exact-operation
optimization: removing motion coefficients changes the sparsity-dependent Arb
multiplication path, so two endpoint comparisons fail. Its complete frozen producer
and results remain in spatial-controls1-source and spatial-controls1.json. No full
trial used it. A closed inverse control passing cannot override this failed claim.
The accepted low-order bracket shortcut is separately exact and remains enabled.

All prior certificates, installed wheel and renderer sources remain unchanged.
The next checker work must reduce the remaining implicit inverse/high-order field
cost, qualify any changed interval arithmetic independently, then require a fresh
complete72-region error sum within256cells/depth10/90seconds. Getting the initial
cover finished does not guarantee enough refinement budget. Remaining piece/motion,
whole-pixel accumulation and world geometry/camera/recomputed-normal gates still
prevent group2closeout. Full frames/scales, gffit readiness and the final actual
SIGGRAPH-quality showcase follow; no consumer implementation or Git publication.


## Opaque145: cheaper inverse; exiting-region error remains unresolved

The explicitly selected quartic inverse replaces repeated normalized square-root
recurrences by R=A^2-k^2 B, after the original positive F=A/sqrt(B) root and branch
proof. At that certified root only, R_u=2*k*B*F_u supplies a positive derivative.
It is not a Newton slope away from that root. Independent direct-vector and closed
inverse controls cover order10, first motion, both axes and tiny complete measure.
This changes interval arithmetic; it does not claim old endpoint equality.
The original local chart27 qualifies with9cells/14.211337899963837reference seconds.

Two fresh complete-piece trials remain failed and preserved. The first refuses
an expanded residual slope at3.9084175999742seconds/84cells. The root identity fixes
that refusal. The second reaches chart66 and refuses boundary transversality at
77.23040769994259seconds/142charged cells/48220inverse evaluations. Neither yields
a complete enclosure. The saved refusal also occurs with the previous inverse.

A geometric-coordinate coarea bound removes the collapsing-edge transversality
refusal. It retains full chart motion, physical measure and tiny nonzero terms;
material roles are weighted once. A separate positive exit-angle coordinate keeps
all moving inverse and Jacobian terms; horizon and other charts retain their prior
mapping. Exact cross-product identities tighten first spatial and parameter
light-cosine derivatives without dropping mixed/higher terms. The former broad
light derivative forced a range fallback. Its removal tightens the complete
chart66 matte derivative width from4.42073408802118e-10 to1.24327410444724e-12.
The latest chart66 trial still fails the existing1e-12combined-basis width rule
at15local cells/inherited local depth3(global10),25.117875500000082reference seconds.
Specular derivative width is3.31396938778964e-15; matte value width1.07434436871258e-13.
The fixed RGB reference-radius requirement remains1e-12; the checker basis budget
has not been relaxed. These are enclosure widths, not measured renderer errors.

The exact positive-exit map controls pass, but the original integral remains
unqualified. Forwarding the residual split hint does not improve this saved trial;
its cached hint is now paired with the correct box. The new paths remain explicit
experimental selectors, not default production rendering. No new full-piece trial
was run after the complete-region failure. Next tighten the remaining ordinary
mixed-derivative/Peano correction bounds on the saved largest chart66 leaves, then
require a fresh shared72-chart complete error sum within the unchanged256cells,
depth10 and construction-inclusive90seconds. Local passes cannot be summed into a
certificate, and no per-chart work/deadline reset is allowed. Further pieces and
motions, installed CPU/CUDA endpoint comparisons, whole-pixel accumulation and
world geometry/camera/recomputed normals remain group2 gates. The original wheel,
renderer, earlier value/gradient certificates and historical failures are retained.


## 2026-09-29: correction and physical inverse checkpoint (opaque146)

The prior local chart66/68 correction failures are repaired. Exact physical
outer-edge algebra avoids horizon cancellation; the direct angle/light
polynomial differential retains coordinate and intrinsic motion. An explicitly
selected unsquared inverse computes each mixed coefficient once. A separate
physical-radius map retains the original complete ordered-cover proof and all
moving Jacobians on every positive interior, entry and exit chart. Zero-angle
horizon charts retain their original map. Closed area/moment/motion tests include
2^-60 measure and motion; complete material comparisons retain order10 and first
parameter coefficients. Context-aware polynomial coefficient preparation keeps
exact outward endpoints. Redundant cross-product refinement is omitted only on
the physical path with its independently checked direct light proof.

Six fresh complete-piece trials are terminal failures, preserved with all
frozen producers and previews. The latest completes its initial72-chart cover
at143charged cells/53.94122750003589integration seconds, then refines to a last
complete enclosure at187cells/94leaves/88.38817140000174integration seconds.
Its maximum basis derivative width8.374639179333583e-10 still exceeds1e-12.
Construction-inclusive reference timeout is90.00085239997134seconds. No piece
certificate, new installed comparison or whole-pixel acceptance is claimed.
This is independent-checker uncertainty, not a measured rendering error.

The final numerical/control regression passes91checks. Local charts57,63,65,66
and68 pass on their declared frozen candidates; these results are not summed
into a whole-piece certificate. Removing duplicate calculations reduces the
chart63 reference from16.414007599989418seconds to7.624561799981166seconds while
retaining its complete local error gate. Two alternative high-order algebra
rewrites did not improve the failing local bound and were reverted; their
failed evidence remains. Numerical defaults, runtime sources, installed wheel,
all acceptance/resource limits and historical certificates remain unchanged.

Saved complete-piece6 evidence now points to smooth order10 remainder bounds
on initial interior regions46,45,44,43,42. A single-cell profile is preserved
in physical-cost1-reference.json, with profiler overhead clearly included.
Before another full trial, investigate tighter integrated derivative bounds
or cheaper shared field evaluation on those saved regions. Any changed proof
must pass independent exact controls and local accuracy/cost qualification;
then require one fresh complete72-region sum within256cells/depth10/90seconds.
Remaining normal/view/screen directions, installed CPU/CUDA endpoints, whole-face
and whole-pixel sums, world geometry/camera and recomputed normals still follow.
The five-group M1 roadmap remains unchanged; this is group2 engineering evidence,
not the final SIGGRAPH-quality showcase.


## 2026-09-29: localized smooth remainder and repeated-work reduction (opaque147)

The explicitly selected checker now bounds the same tensor Gauss5 rule with
localized order-ten Peano masses on full positive interior charts. Its 25
quadrature points are unchanged. I-minus-Q has a nonnegative order-ten kernel
by the Gaussian Hermite interpolation remainder; its exact truncated-power
antiderivative is enclosed outward on each dyadic derivative interval. Both
tensor directions and all first-parameter coefficients remain. Smoothness is
required of the extension; ordinary residual and moving-boundary corrections
remain separate and fully included. The fixed derivative partition is bounded
at eight intervals per axis, and every evaluation is charged to the existing
clock. These intervals are derivative enclosures, not additional quadrature
cells or separately accepted patches. Default checker stages are unchanged.

On saved interior chart46, four transverse derivative intervals reduce the
maximum derivative width from1.2819929421453283e-10 to3.1870856787127165e-12;
eight reduce it to1.0239056854832865e-12. Corresponding local evaluation times
are0.643927099998109,0.8360817999928258,1.122868100006599seconds after reuse.
The method costs more per initial cell but can avoid full quadrature refinement.
Entry chart63 remains faster with its existing remainder, so entry/exit/horizon
charts retain it. All three local chart63 alternatives pass their accuracy gate.

A bounded64-entry transverse polynomial-row cache preserves every scalar Horner
operation and exact outward endpoints, including context changes and tiny terms.
Alternating cache-disabled/enabled/enabled/disabled same-cell timings are
0.7438830999890342/0.6299680999945849/0.6450631000334397/0.7243027000222355seconds:
about13.16percent less time on average, with identical value/gradient endpoints.
The disabled control retains cache lookup overhead; this is not a full-piece
speed guarantee. Exact controls pass in Decimal and pinned Arb arithmetic.

One fresh complete-piece trial is preserved as failed. It reaches a complete
90-leaf sum at179charged cells/88.19492159999209integration seconds, then stops
at181cells/90.00834660005057construction-inclusive reference seconds. Largest
combined gradient width1.09907644855573027091715172480011415448897740074472084e-10
versus1e-12: about109.91times the limit, improved about7.62times from opaque146.
This is checker uncertainty, not a measured renderer error. There is no new
piece certificate, installed CPU/CUDA comparison, or group-two completion.

Every retained leaf now has compact error-component evidence. Of90retained
regions,15have smooth width above the whole-piece1e-12allowance: charts40..53
and67. Ten leaves have correction value or tangent-transport width above that
allowance, overlapping smooth chart67:24distinct regions. This threshold is a
diagnostic, not a per-region acceptance rule; smaller contributions must also
fit in the combined sum. Matte-gradient smooth widths sum about7.61984e-11,
and correction/transport widths about3.37092e-11. Next target the common
transverse smooth bound on charts40..53 and the mixed entry bound on67, while
retaining all correction contributions. Qualify any improvement locally and
measure total cost before a further complete-piece trial. No resource limits,
accuracy thresholds, renderer sources, installed wheel or historical
certificates changed. All test-first and five driver compatibility failures
are preserved; the latter were repaired and their six affected checks pass.


## Opaque148: directional smooth proof and combined cost, 29 September 2026

The explicit Gauss3x8 candidate uses24smooth samples instead of25. Its first
direction has a sixth-derivative Gaussian remainder. Its second direction
uses an outward L1 bound on the order-ten Peano kernel of Gauss8: polynomial
power-to-Bernstein algebra, not sampled convergence or an assumed kernel sign.
Positive tensor weights and IxIy-QxQy=(Ix-Qx)Iy+Qx(Iy-Qy) bound both axes.
Exact rational node moments through degree15, closed polynomial/rational
integrals, signed motion and2^-60terms qualify the rule in Decimal and pinned
Arb. Every physical Jacobian, ordinary residual and moving-boundary term stays.
One attempted region is still one sample-equivalent cell; all derivative and
correction work stays in the shared clock. Positive fixed-upper-angle interiors
and entries select this rule explicitly. Exits, horizons and default checker
stages keep their prior rules. No renderer or installed-wheel change occurred.

Across the15saved dominant smooth contributors (40..53and67), fresh paired
one-cell comparisons tighten smooth widths about13.5to16.6times. Alternating
old/new measurement order gives13.127754999964964versus12.012416000012308seconds
of total local evaluation,8.50percent less. This excludes repeated geometry
construction from the local timing; the entire comparison worker is timed too.
Thirteen full local enclosures pass without subdivision. Separate fresh complete
local checks pass for53and67using3charged cells each; entry controls63and65
pass using5charged cells each. No local certificate is reused to accept the whole piece.

All122affected existing regressions and15new numerical/driver controls pass.
The compiled wrapper repeats the independent exact controls. The13expected
test-first failures and one missing-driver-selector failure remain recorded.
One fresh complete-piece trial still fails at190cells and90.00214150000829
construction-inclusive reference seconds. The first complete72chart/143cell
sum takes53.340640800015535integration seconds versus59.130829800036736before,
9.79percent less. This is one whole-cover comparison, not a repeated benchmark.
The last complete sum uses189cells/95leaves at89.06231619999744integration
seconds. Maximum gradient width2.4477023465689002622439055681154114762192201085238509e-11
is4.49times tighter than opaque147 but24.48times the1e-12allowance. The checker
uncertainty is not a measured rendering error. No complete-piece certificate,
installed CPU/CUDA acceptance or group-two completion is claimed.

This current disjoint cover has three smooth contributions above the entire
1e-12allowance: unrefined charts53,54,55. Six separate correction contributions
exceed it: chart68right half,66right half,63upper half,62upper half,64middle-right
quarter and71lower quarter. Smooth matte-gradient widths sum9.523154935117527e-12;
moving-correction widths sum1.4953868530571475e-11. Smaller terms also count.
These diagnostic counts describe the latest adaptive subdivision, not new
rendering defects or independent local acceptance gates. The next repair targets
the shared boundary-correction bound/cost and remaining smooth remainder, then
requires another fresh combined enclosure within the existing limits.

Evidence: opaque148-20260929/fifteen-regions1-reference.json, local53-1 through
local67-1 reports, regression1, piece-preview1.json, complete-piece1-reference.json,
combined-analysis1.json and qualification1.json. All earlier certificates and
failures remain. Memory, dashboard history and quotas remain authoritative.


## Opaque149: complete curved-piece reference passes, 29 September 2026

The original piece8/face5742 normal_tilt reference now passes as a complete sum,
with every smooth, ordinary correction, knot contribution, moving measure and
conservative boundary-transport enclosure retained. The successful fresh trial
uses111leaves and221charged cells (including110internal ancestors), depth10,
88.3851845000172construction-inclusive reference seconds, and89.52227760001551
whole-worker seconds. Sampled peak host usage is115298304bytes, one thread.
All limits remain256cells/depth10/90reference seconds/120worker seconds/
180watchdog seconds/1GiBhost. Heavy workers were serial.

Maximum basis gradient width is9.08017359629866136603000354592796924070015164696573e-13,
below the unchanged1e-12width allowance. Maximum basis value width is
6.8062747571522389305216825753986927378779942668040e-13. Complete RGB value and
normal_tilt gradient radii are each within1e-12. Smooth matte-gradient width sums
4.038604104043087757043562475e-13 and all correction/transport widths sum
5.041569492255573608986441073e-13. These are reference uncertainty bounds, not
measured rendering errors. The earlier opaque148 bound was2.4477023465689003e-11.
This pass does not certify the production face integral, other motions, a full
pixel, a full frame, world-space vertices/camera or recomputed normals.

The selected explicit checker profile combines these qualified changes:

- Directional Gauss3x8 retains24smooth samples and derivative orders6/10.
  Four transverse derivative domains use independently bounded quarter-kernel
  L1 masses, computed through outward Bernstein polynomial integrals. Quarter
  weights are not assumed equal. Horizons/exits retain their prior smooth rule.
- Fourth-order correction proofs retain25samples and every jump order0..3.
  The positive tensor identity Qy(Ex)+Ix(Ey) allows actual quadrature lines and
  exact-measure transverse proof domains. An ordinary enclosure is normally
  tried first; its1e-14width work hint never determines acceptance. Narrow radial
  boxes (width<=0.5) with complete natural residual ranges wider than1e-12 may
  directly select five quadrature lines plus four transverse domains. That
  independently sufficient identity uses nine proof domains without the two
  preliminary ordinary domains. All computation remains on the shared clock.
- Complete natural range integrals are retained when width<=1e-15; small
  nonzero contributions are not replaced by zero. If monotonicity is unresolved,
  four exact-measure range domains with the parent extension fixed may tighten
  the fallback. This is proof work, not additional quadrature or a fresh budget.
- Fixed-F0 mixing commutes with polynomial evaluation. Outward combined
  coefficients evaluate the identical material with fewer operations, retaining
  horizon and segment differences and every parameter coefficient. Only private
  correction/transport queries omit their unused view-highlight field; full
  smooth radiance still computes it. Consumed input coefficients match previous
  endpoints at orders0/4/10 on five actual chart classes.
- Pinned python-flint0.9.0 supplies compiled inverse-series arithmetic. Certified
  Newton brackets can stop redundant contraction while retaining complete root
  and slope enclosures. Fast Decimal-text/Arb and directed dyadic/Decimal export
  retain rigorous enclosures; identical former Decimal endpoints are not claimed.
  Scalar-conversion and native-series reuse experiments remain disabled.
- Geometry-only layout3 specifies111disjoint leaves. Its full cover, inherited
  depth and110ancestor charges are verified. No numerical integral, root,
  derivative or certificate is reused to accept the trial. Every contribution
  is recomputed; completed expensive tests need not be repeated for status.

Independent controls include closed polynomial and nonlinear logarithmic/rational
integrals, exact kernel moments, both signs of tiny motion, all jump orders,
complete moving Jacobians, arithmetic context and mutation safeguards, rejected
selectors, sample/depth/cell accounting and construction-inclusive cancellation.
301distinct numerical/driver control nodes pass, including the earlier affected
regressions;34teaching and77dashboard JavaScript tests pass. The compiled wrappers
also run independent cases internally. All647Python outcomes across99reports are
preserved:535passes and112failures. Failures include test-first checks, harness
mistakes, rejected experiments and the nine earlier full-piece attempts. They
are history, not silently reclassified as current failures or successes.

Earlier candidates progressively reduced the complete gradient width but did
not satisfy accuracy and runtime together:

| Trial | Last complete gradient width | Outcome |
| --- | --- | --- |
| 1 | 2.1089783407065875622641479094830752739678527301066711E-11 | Stopped at the unchanged deadline |
| 2 | 9.567456607466893569025636130073507626105399476188597E-12 | Stopped at the unchanged deadline |
| 3 | 8.723221913834721385945743110274266416614808520041460E-12 | Stopped at the unchanged deadline |
| 4 | No complete sum | Stopped at the unchanged deadline |
| 5 | 3.033543730633897769873618489303773706309561816974547E-12 | Stopped at the unchanged deadline |
| 6 | No complete sum | Stopped at the unchanged deadline |
| 7 | 1.563107908179430988617870198994455474812765397935365E-12 | Stopped at the unchanged deadline |
| 8 | 1.461938196551908390967371090894323914097297398819981E-12 | Stopped at the unchanged deadline |
| 9 | 1.099312516729144843430283602270222805341520787332288E-12 | Stopped at the unchanged deadline |
| 10 | 9.08017359629866136603000354592796924070015164696573E-13 | Passed |

The first111leaf sum is also the accepted sum:87.62880980002228integration
seconds, with construction and final checks included in88.3851845000172reference
seconds. Runtime headroom is small; one pass does not prove stable throughput.
Local old/new comparisons preserve unchanged and slower cases. First-case cold
setup can affect their timings; they are not repeated global benchmarks.

Certificate: opaque149-20260929/complete-piece10-reference.json, SHA256
51461768fb6d5d1567b473fcd93cd9a6f8129b4980b52e3d9576fe01654089e9.
Use its frozen producer snapshot/manifest and piece-preview10.json. The frozen
input, pinned arithmetic, opaque128installed wheel and all prior certificates
remain unchanged. qualification1.json includes every report, resource measurement
and control node. Final-verification and dashboard audit reports establish
preservation separately; a passing audit is not additional numerical acceptance.

Next extend independently qualified references to remaining pieces and declared
motions. The current angular driver explicitly targets piece8normal_tilt; a
certificate cannot be relabelled for a different direction. Sum every applicable
piece outward before comparing a complete installed face on CPU/CUDA. Whole-pixel,
world-geometry/camera, changing horizons and recomputed normals remain separate
group-two gates. Frames/scales, gffit readiness and the final actual SIGGRAPH-quality
showcase follow; M1 remains unfinished.


## Opaque150: extend motion identity and repair front reference connections

The difficult curved piece now passes independent checks for changes to normals, viewing direction and screen position. The two new checks took 86.81 and 55.52 seconds, within the same 90-second limit. The next front-facing piece still needs work: its latest check reached 122 regions, but uncertainty remained 1.97 times the allowance when time expired. Complete face/pixel comparisons and the remaining motion and M1 gates are still open.

The piece8 view certificate is `068d9aafbfa5a3ce40f6529c92b348b9a8023b00cb800a3f4f10c1e26b32c26a`; screen certificate is `8302398d13629ac2f1b2a4028c8c2df9b3ec1657bc191e517440681a3a568a97`. Each has its own frozen producer. Attached screen translation is not a world-vertex/camera or recomputed-normal certificate. The opaque149 normal certificate is preserved.

The complete-face composer requires every applicable piece exactly once and enforces outward combined RGB value/gradient radius1e-12. Eleven failure-first composition controls pass. No native fan/visible-piece partition equivalence is assumed.

The first face5740 campaign stopped before its second job. Every subsequent front-piece trial is separately named, frozen, and terminal; no result was reused as acceptance. Explicit fourth-order correction, exact crossing slopes, both positive tensor identities and gradient-directed subdivision retain every jump and distributional transport term. Material-only correction queries omit an unconsumed half-vector;12controls retain exact consumed endpoints and the three saved complete rows are identical. Paired local time2.684850300080143to2.3514725000131875seconds is a local12.4percent improvement, not whole-piece throughput.64affected final controls pass. Exact parameter slopes remain an unselected experiment: the complete saved bounds improved less than0.02percent while cost increased.

All trial limits remain256cells,depth10,90construction-inclusive reference seconds,120worker/180watchdog seconds,one thread and1GiBhost. Geometry-only layouts charge every ancestor and recompute all fields/bounds. Installed wheel/runtime sources remain unchanged. Evidence: `opaque150-20260929/qualification1.json`, stopped and passing reports, candidate manifests and source snapshots under the internal qualification root. All historical failures, quota and coverage safeguards remain.


## 30 September 2026: neighboring normal-change piece qualified

The neighboring front-facing piece now passes its independent normal-change check: 125 regions, 249 of 256 charged cells, and 72.02 of 90 reference seconds. Its summed gradient uncertainty is below the unchanged allowance. The difficult curved piece retains its three declared motion references. Remaining motions, complete face/pixel comparisons and world geometry/camera checks are still open; M1 group 2 is unfinished.

The three saved boundary regions have39-61percent narrower complete tangent bounds. Certified absolute Peano-kernel mass retains coefficient variation, root uncertainty and all four jump orders. Continuous transverse proofs are omitted only on quadrature lines where the final enclosure never consumes them; every line derivative/jump remains.102affected controls pass, and the second repair preserves all local endpoints exactly.

The fresh complete-piece certificate has basis gradient width9.799452585870477048225265466651844148779483356041850E-13 versus1e-12, maximum RGB gradient radius2.939835775761161349600778113440836986500994687119951E-13 versus1e-12, depth8 and101806080peak host bytes. Certificate SHA256 249f3edcc433515553ad738f7174062b82a9c3da2336f00f3270273ba7684ddc. No installed face/pixel comparison is inferred.

The first trial remains failed at90.02456219994929seconds before a complete cover. CPU was observed at100percent after that run, dominated by Blender/Creative Cloud, then fell to31-43percent before the fresh trial. Local wall-time comparisons are contention-sensitive. This does not excuse a failed trial or prove capacity on other workloads. The final trial passes the original limits.

Evidence: `opaque151-20260930/certificate-audit1.json`, both frozen trial previews/reports, `mass-comparison1-reference.json`, `line-comparison1-reference.json`, controls and CPU-load evidence beneath the internal qualification root. Next: remaining piece1 view/screen directions, the other pieces of face5740 and subsequent motion/face/pixel gates. Existing piece0 and piece8 certificates remain. Runtime, installed wheel, quotas and historical failures are unchanged.


## 30 September 2026: recovered PC and complete moving-face qualification

Five of thirteen selected original-pixel pieces (0,1,2,3,8) now have independent
normal/view/attached-screen motion references. The three pieces of face5740 sum
outward with maximum RGB gradient radii6.971141172199583e-13,
2.3452961097279805e-13 and4.7398480714683795e-14, below1e-12.
The first installed face comparison failed nonlinear gradient accuracy; the
capacity-counted shared flat rule now uses5x8 nodes for this face, preserving
curved priority and every fan within800samples/32blocks/32faces/16cuts.
A freshly rebuilt wheel passes six serial complete-face comparisons against
BOTH endpoints at1e-10absolute+1e-9relative and exact CPU/CUDA value/gradient bytes.
This is one complete visible face with three packed-input motions, not a
whole-pixel or world-geometry/camera/recomputed-normal certificate.

Wheel SHA256cc69f81ae55d200573df9b8d52df72b9fc3812108ce5f166765fc618b8164893;
all74installed members verified. Full fresh build4 took59.986seconds with
599994368sampled peak process-tree bytes. Exact Windows DLL-resource PTX
embedding removes the giant compiler initializer; all native components and the
Torch adapter were rebuilt. No laptop binary or old adapter was reused.
The three memory-refused builds remain, including their corrected hypotheses.
22CPU and30CUDA native-flat controls pass. The first mixed-device control run
exceeded the CPU-only host profile and remains failed; subsequent GPU jobs use
the pre-existing2GiB CUDA-host profile, not a raised limit.

A first complete-face audit failed one gradient bit even though its independent
accuracy checks passed. Trace comparison localized the mismatch to the diagnostic
assembly’s default Torch broadcasting, normalization and reduction. The repaired
probe uses existing installed fixed-order production helpers;12closed-chain and
independent scaled-vector/channel controls pass, followed by a freshly frozen
six-job campaign with exact device bytes. The failed campaign remains historical.
New-wheel original selected-pixel colour/albedo regression also passes both
devices with exact image/mask/Jacobian bytes and30,705,664GPU bytes.

Windows recorded bugcheck0xA and an unclean restart; this does not establish a
cause or blame this session. Interrupted piece2 trial retains its incomplete
artifacts and has no fabricated certificate. The dashboard database survived
integrity checking, and its service uses internal storage.
Evidence: internal opaque152 and opaque153 reports, frozen manifests,
face5740-installed-campaign2.json, original-regression-audit1.json,
resource-binary-audit1.json and wheel-build4.json.

Next: eight remaining piece references, outward complete-pixel comparisons,
world geometry/camera/recomputed normals and changing horizons. Then original
frames/scale (including four1080pbackward cases), synthetic gffit readiness and
actual SIGGRAPH-quality3Dshowcase/closeout. M1 remains unfinished. All numerical,
worker, memory, history and quota requirements remain unchanged.


## Opaque154 — complete front pieces and remaining outside-horizon motion

Thirty-three independent motion directions now qualify across eleven original
visible pieces:0through10. All newly produced bounds include construction and
keep256cells/depth10/90reference seconds. Faces5740and5741 pass installed CPU
and CUDA against both outward endpoints, with exact device bytes.

Piece10 required a geometry-only radial schedule:100fresh leaves,199charged
nodes,depth8; normal/view/screen reference times64.10/57.92/43.96seconds. The
first90-second trial and local direction investigation remain historical.

Piece9 normal trials1/2/3 remained terminal failures. The curved fourth-order
transverse profile is explicit;67closed-integral/refusal controls retain every
smooth term, jump, tiny signed motion and separate flux. Avoiding optional
quadrature-line root contraction and no-jump field work preserved the saved
bound improvement while reducing cost. Trial4 qualifies89fresh leaves,177cells,
depth10 in74.9678802reference seconds. Fresh view/screen qualify88leaves,175cells
in70.8793798/46.2207236seconds. Frozen producer, input, profile and schedule bytes
are independently audited. No partial certificate was reused as numerical proof.

B4: even just the outward sum of faces5740and5741 has normal blue-gradient
radius1.227434991472215e-12, exceeding the unchanged1e-12whole-pixel reference
allowance. This is a reference uncertainty gate, not a renderer discrepancy.
Inside-horizon pieces11/12, tighter complete-pixel bounds, installed remaining
faces/pixel, world geometry/camera/recomputed normals and changing horizons
remain. A packed attached-screen direction does not close those gates. Groups3–5
and the actual SIGGRAPH-quality original-mesh showcase remain unfinished.

Evidence: internal opaque154 front-progress1.json, piece10-motion-audit2.json,
face5741-installed-campaign1.json, partial-pixel-radius1.json, the piece9 motion
certificates/audits and preserved terminal trial/control reports. Native wheel
and every numerical/resource/history safeguard remain unchanged.


## Opaque154 inside-horizon and installed-face continuation — 30 September 2026

A separate inside profile reverses the original radial coordinate so the moving horizon is at zero and the physical origin is at one. It proves a strictly increasing radial angle and a constant transverse origin edge, retains every band and its full moving Jacobian, and leaves the old outside-only guards intact. The unsquared residual uses the signed inside width; the faster inverse substitutes the reversed physical radius exactly into rational coefficients before outward evaluation. Forty-five closed/mixed-field/known-root controls pass, including signed2^-60motion, and five outside bindings pass.

The baseline inside trial and first angular trial both reached the90-second reference deadline and remain unresolved. The physical-coordinate repair qualifies piece11normal/view/screen in44.48/44.60/31.93seconds (127cells/depth6), and piece12in36.50/37.10/25.70seconds (131cells/depth7). All six certificates are independently audited for input identity, source/artifact bytes, complete disjoint coverage and the unchanged bounds. No installed binary was rebuilt for this checker change.

Fresh installed face5743/5943 and preserved-reference face5739 comparisons all pass on CPU and CUDA, including both endpoints and exact device bytes. Together with earlier faces,34installed jobs form17exact device pairs. Original piece0 normal metadata was normalized only into a new linked wrapper after verifying its input, normal-only producer and cyclic anchor1; old numerical fields and artifact bytes remain. Its frozen Python source is verified; the broader historical C/CUDA inventory is not claimed to be frozen by that producer.

The39-reference catalog exposes the remaining whole-pixel uncertainty explicitly. Normal/view blue gradient radii2.829511498626245e-12/1.925791857752707e-12 and screen blue colour radius1.117336920467054e-12 exceed1e-12. No whole-pixel trial was launched using them. Every earlier failure and harness error remains; no limit or numerical requirement changed.


## Opaque154 complete selected-pixel view and screen checkpoint

All six complete faces now pass all three packed-input directions on the rebuilt
installed wheel `cc69f81ae55d200573df9b8d52df72b9fc3812108ce5f166765fc618b8164893`:
36 device jobs and 18 exact CPU/CUDA pairs. Stronger selected-piece references
use only fresh calculations on geometry-only seeds; all ancestors are charged.
The stronger piece10 normal attempt timed out and is excluded. Earlier outcomes
and all immutable catalogs remain historical.

Complete-pixel composition is test-first: 21 expected missing-function failures
plus 21 legacy passes, followed by 42 passing controls. A separate motion-sum
control had eight expected failures plus five legacy passes, then 13 passes.
Every artifact and input is checked, every face sum is recomputed outward, and
passing individual faces cannot bypass the final combined uncertainty test.
Signed tiny terms and all active owners remain mandatory. A compile-before-write
preparation error and two auditor metadata/format errors were preserved and
repaired before admission.

The independent whole-pixel view derivative radius is
8.417111116298923e-13; attached-screen colour radius is
9.237195261291895e-13. All endpoints meet the unchanged allowance. Fresh complete
CPU/CUDA comparisons pass all RGB values and first derivatives against both
endpoints at 1e-10 absolute plus 1e-9 relative, with exact device bytes.
All 760 active samples and six visible faces participate. CPU jobs took
7.2762–7.4726 seconds; CUDA jobs 16.2880–17.2508 seconds. Peak CPU host is
690343936 bytes, CUDA host 1286328320 bytes, and allocated GPU 6474240 bytes.
These remain selected-pixel diagnostic measurements, not full-frame performance.

Proofs and actual outputs: `opaque154-20260930/pixel-reference-audit1.json`,
`pixel-installed-audit1.json` and `pixel-installed-campaign1.json` beneath the
internal qualification root. Dashboard checkpoint
`pc-opaque154-motion3-20260930` explains the result and remaining gate.
Normal-motion uncertainty, world-geometry/camera/recomputed-normal/changing-horizon
checks and all later closeout groups remain open.


## Opaque154: qualified checker algebra and stronger normal references

Exact rational light/norm polynomials retain every spatial and first-motion
coefficient. Eighteen independent closed/Taylor controls and nineteen physical
map/profile controls pass. Twelve horizon controls additionally prove that
forwarding the material-only flag preserves every consumed angle, light and
Jacobian endpoint exactly; full smooth radiance retains its highlight.

The fresh inside-piece11 normal proof passes at private1e-13 target:137cells,
69leaves,depth7,50.4174134reference seconds,51.8373967worker seconds and
107331584host bytes. Its independent artifact/coverage audit is
opaque154/piece11-normal-light12-audit.json (certificate SHA
 e6a27b0dac341ffd50eaeaa294b56898807373cf57249430904b9f0f2bec7c92).

The fresh outside-piece9 proof passes at private3e-13 target:193cells,97leaves,
depth10,74.0391522reference seconds,75.1296074worker seconds and112082944host
bytes. Its independent audit is opaque154/piece9-normal-horizon14-audit.json
(certificate SHA c5fea6e4e35b01f0bcb28bf65e2dca189d8f94772b1e77c44aa66e22b0a1daf0).
The earlier private2e-13 trial remains failed and excluded. New targets remain
stronger than the original5e-13 per-piece requirement; no failure is relabeled.

Piece8 transverse10,light11 andhorizon15 all remain terminal timeouts before
complete coverage. Preserve every report. Geometry-only seeds never reuse their
numerical bounds. These repairs change independent checking, not the native
image, accumulation, installed wheel, full-pixel1e-12 gate or any resource cap.
Evidence is under the internal qualification root specified by MANIFEST.json.


## Opaque155: complete normal pixel and public controls

The shared compiled inverse now also returns its complete physical slope and
first-motion coefficients. Independent closed controls include inside/outside
maps, orders4/10, both coordinate axes and signed1e-30 motion. The polynomial
slope alternative was slower on saved regions and remains unselected. The
material-input cache applies fresh local clamps after reusing global inputs.

A fresh piece8 proof and stronger front0/4/5/6/7 proofs close the outward normal
sum. Piece9-shared-slope1 and piece6-tight1 timed out; neither is admitted.
Piece6-tight2 recomputes all bounds on a verified complete geometry-only seed,
charging195nodes and60.7075287reference seconds. The final13-piece normal
reference radius is9.949319051167309e-13. Installed CPU/CUDA full-pixel checks
pass with exact bytes and worst endpoint error/allowance0.48355526625414086.

The public camera principal-point x comparison passes with exact device bytes.
An ordinary scalar normal parameterization initially differed by1.0842e-19 in
one derivative channel. A fresh[3,5002,3] production normal-input Jacobian audit
matches exact device bytes, localizing that difference to caller reduction.
The test now uses the already qualified fixed-order broadcast reduction, with
unchanged forward bits and every signed contribution; both devices pass.
This does not promise byte identity for arbitrary caller-defined Torch graphs.

Evidence: opaque155-20261001/original-motion-reference-catalog3.json,
pixel-installed-audit1.json, public-input-audit1.json, public-normal-audit2.json
and public-installed-campaign1.json under the mapped internal qualification root.
World-vertex/camera-pose/recomputed-normal/changing-horizon gates remain open.
