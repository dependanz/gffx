# Localized angular-boundary reference bounds

20 September 2026. Development checker only; the installed renderer is unchanged.
The opaque67 original-pixel campaign remains a terminal failure. This method is
qualified on finite analytical controls. Its approved original campaign now also
ends in a second-piece cell-budget refusal,as recorded below.

## What changed and why the bound remains conservative

On a physical cell X=[a,b], let m=(a+b)/2 and let the transverse coordinate vary
through its entire cell. For every root q(x,y)=k, the mean-value theorem gives
x=m+(k-q(m,y))/q_x(xi,y), with xi inside X. Outward bounds on q(m,y), q_x over
the whole box, and the stored knot therefore enclose every crossing. Intersect
that enclosure with X. If q_x includes zero, retain X instead of dividing.
An empty intersection certifies no crossing. Existing multiplicity bounds remain:
at most four view-cosine roots or two light-cosine roots on noncurved lines;
monotone lines have at most one. Curved nonmonotone crossings still refuse.

For normalized Gauss5 error E=Q-I, define
K_j(t)=E[(x-t)_+^(j-1)/(j-1)!]. Piecewise Taylor expansion gives an ordinary
third-derivative remainder plus jump terms [f^(j)] K_(j+1), j=0,1,2.
A jump is now charged the maximum kernel magnitude over its certified crossing
interval, rather than the maximum over the entire cell. Kernel pieces are
polynomials of degree at most three. Outward Bernstein coefficient hulls bound
every overlapping Gauss-node segment, including both sides of uncertain nodes.
Taking the smaller of this valid bound and the valid global bound stays safe.
This uses no sampled maximum, continuity assumption or omitted jump term.

Physical powers remain h^4 for the ordinary third-derivative integral and
h^(j+1) for each order-j jump. Positive tensor weights let the two directional
error bounds add after multiplying by the transverse width. The producer must
cover every crossing, either globally or through localized witnesses. At most
16 witnesses are accepted per row, matching two roles with at most eight knots.
The ordinary bound, global jumps and localized jumps are retained separately.
No moving-domain parameter derivative is inferred from spatial derivatives.

A work refusal now preserves the last complete-domain enclosure before a parent
cell was removed for refinement. Incomplete active leaves cannot be labelled as
a whole-domain enclosure. This is diagnostic evidence, not relaxed acceptance.

## Finite evidence

Opaque68/localized2 has20 passing controls;regression1 has16 existing Peano
passes and square1 has7 square-integration passes. Tests start with exact
rational step/ramp/curvature integrals, oblique crossings carrying all three
jump orders, uncertain Gauss nodes, signed root-location bounds, invalid
witness rejection and nonmonotone fallback. An actual piecewise-log material
fixture is checked against a separate closed antiderivative, including nonzero
stored value and slope jumps. Its bounds improve about36-fold;the oblique
polynomial case improves about136-fold and near-endpoint curvature about1170-fold.
These are uncertainty reductions on named fixtures,not renderer speedups or
original-mesh accuracy claims. Original pixel radius1e-12 and native acceptance
1e-10 absolute+1e-9 relative remain unchanged.

The implementation trusts the independently tested interval/jet machinery and
Python Decimal60 directed arithmetic. It is not an external Lean certificate.
All red tests and the two fixture operand-order mistakes remain in the reports.

## Fresh original experiment approved as D319

User-approved opaque68-20260920/campaign1 uses the frozen original pixel(101,78) and
30 source/test hashes in candidate-source1.json (SHA256
 e33086b0d6cf469e2af4b296f40ffc1695e204035f070e35171efc6984632426).
At most13 serial jobs, each256cells/depth10/90reference seconds/1GiB/one thread,
120s cooperative and180s watchdog. Aggregate3328cells/1170reference seconds/
1560supervised seconds. Stop at first failure and charge failed work. No earlier
first-piece result can be reused because the numerical source changed. All13
outward certificates and their summed1e-12radius must pass before the saved
matching native CPU/CUDA result can be compared. D319 resolves Q17 with explicit approval.
The prior campaign and all historical failures remain unchanged.


## Actual campaign outcome: terminal failure

The campaign ran once and stopped at its first failed piece. Piece0 passed
133cells in35.09285200003069reference seconds. Piece1(face5740) exhausted256
cells after72.45100709999679seconds. Its worker stayed below90reference seconds
and1GiB;the refusal is the cell limit. Eleven pieces were never started.
Including failure,389cells/107.54385910002748reference seconds and
110.65124999999534supervised seconds were charged;peak host114163712B.

The last complete enclosure,at255cells with128leaves,has matte width
1.60979189512636927131761523951372970872230120741943e-12. The allocated radius
is6.16155435667164489328157280191951655650246391818375471613732e-14,so that
width is13.0631964106858 times the allowed width. This is a full-piece bound,
not just a last-cell estimate. The last cell separately records dominant
localized jump uncertainty;its component split is not a full-piece split.

The first piece uses8fewer cells than the prior attempt,but this method has not
qualified the full pixel. No complete certificate or native accuracy comparison
was produced. Both campaigns remain terminal. The next method must improve the
remaining boundary-error enclosure,with closed independent controls before any
fresh reviewed original run. No accuracy,material or resource relaxation follows.
See opaque69-20260920/campaign-audit1.json for the complete charged audit and
opaque68-20260920/campaign1 for immutable source,input identity and receipts.

## Opaque70: signed boundary-error enclosures

The independent development checker now retains the sign of a certified
single-crossing contribution. For E=Q-I, a line error is the ordinary remainder
plus sum of physical h^(j+1) times jump_j times K_(j+1)(root), j=0,1,2.
Signed Bernstein hulls enclose each K over every overlapping node segment.
The global absolute bound is only an outward intersection. Exact Decimal sign
reversal is required: a test at the central node caught ambient-precision
rounding of unary Decimal negation. That failed candidate was never used on
the original mesh.

For two dimensions, QxQy-IxIy = Qy(Ex)+Ix(Ey). Both transverse operators are
positive with mass equal to their physical width. Therefore uniform signed
line enclosures add after multiplication by that width, and the final integral
is the sampled quadrature MINUS this signed enclosure. Ordinary third-derivative
and unsigned-jump terms remain symmetric. Refinement uses enclosure radius,
not the magnitude of a known correction; the callback retains its previous
nonnegative upper-bound API.

The producer uses signed jumps only for strictly nonzero derivative intervals.
A decreasing crossing reverses high-angle-minus-low-angle to spatial
right-minus-left; spatial jet chain factors remain intact. Endpoint intervals
certify a crossing on every transverse line. If that proof is unavailable,
include zero in the entire crossing contribution. A nonmonotone or weakly
monotone crossing retains its previous absolute bound and multiplicity; curved
nonmonotone crossings still refuse. At most16 witnesses and all existing
precision,tolerance,cell,depth and worker limits remain unchanged.

Finite qualification:17 signed controls,25 localized/material controls,16 Peano
controls and7 square controls pass. Exact step/ramp/quadratic integrals cover
both crossing directions and nonunit physical boxes. Oblique and curved rational
polynomial integrals have about616-fold narrower enclosures. Two exact log
material fixtures at normalized knots.13 and.87 need3cells versus7 for the
unsigned version,with the same1e-12 radius and256cell/depth10 caps. They are
finite checker results,not original-mesh qualification or renderer speedups.

The near-endpoint log fixture retains an ordinary remainder that dominates
its total bound:8.585387238119091e-9 versus8.851148021868832e-9 (about3percent
narrower). The initial whole-bound2x assertion failed. It was corrected to test
exact containment and the signed boundary contribution separately; no numerical
acceptance or resource limit was weakened. Initial missing APIs,a test Fraction
constructor mistake,the node-rounding defect,missing material witnesses and a
directional-callback regression are all retained as failed reports.

No original campaign was executed in this checkpoint. The production renderer
and installed wheel are unchanged. Only a fresh reviewed experiment can measure
whether the candidate resolves the original second-piece failure. All13matching
piece certificates are still required for whole-pixel acceptance. This is a
mathematical derivation with executable controls,not an external Lean proof.

## Opaque71: original-mesh test of the signed candidate

User approval D321 resolved Q18 and authorized the frozen opaque70/campaign1.
It ran exactly once and stopped at the first failure. The first piece passed
133cells/34.58884379995288reference seconds. The second used256cells and failed
after71.31408899999224seconds. Eleven pieces were not started. Both workers
stayed within their CPU limits. Total charged work,including the failed piece,
is389cells/105.90293279994512reference seconds/108.75625009997748worker seconds,
with113348608Bpeak host. No complete-pixel certificate was produced.

The last complete cover at255cells/128leaves has matte interval
[0.00656252514934138344996753030825702313724787191816412066205604,
 0.00656252515091526402774535130797131693016755020430029572822148].
Its width1.57388057777782099971429379291967828613617506616544e-12 is about2.2percent
narrower than D319,but still12.7717819779813times the allocated width. Native
accuracy comparison remains inadmissible. This failure does not prove the
renderer is inaccurate;the independent reference remains too uncertain.

The last recorded cell's two signed boundary contributions are approximately
[-9.328e-16,9.328e-16] and[-8.094e-16,8.094e-16]. Keeping their sign does not
make them narrow when their possible ranges still span both signs. This is a
last-cell diagnostic,not a full-piece decomposition or proof of a particular
root-location cause. The next analytical work should preserve more information
about a crossing's transverse variation or integrate the boundary contribution
explicitly,with independent closed-form controls before any fresh original run.
All precision,tolerance,resource and moving-gradient obligations remain intact.

The first postprocessing audit used the producer's completed-cell subtotal and
omitted the current failed piece. campaign-audit2.json corrects133 to389cells,
explicitly supersedes the retained campaign-audit1.json and binds unchanged
worker/result files. No failed cost is forgiven. D317,D319 and D321 campaigns
are terminal. The installed wheel and production runtime have not changed.


## Opaque72: condition the transverse coordinate and integrate the boundary

The development reference now explicitly selects a stronger fixed-geometry
correction. The runtime renderer and installed wheel are unchanged. The tensor
identity remains QxQy-IxIy=Qy(Ex)+Ix(Ey). Ex is now enclosed separately on each
of the five actual outer Gauss-node intervals, rather than over all y at once.
For Ey, a crossing q(x,y)=k may be integrated with y as the coordinate only
when both q_x and q_y strictly exclude zero over the complete physical cell.
This proves a single monotone graph; endpoint interval comparisons and the
mean-value root enclosure retain roots clipped to either edge of the cell.
If that condition fails, retain the previous whole-transverse bound and root
multiplicity. Curved nonmonotone cases still refuse.

For an inner-axis derivative jump J_j, j=0,1,2, the integrated error term is
h^(j+2) times the integral of J_j * abs(q_inner/q_outer) * K_(j+1)(t) dt,
where t is the normalized inner root coordinate. The spatial right-minus-left
jump owns the sign; the absolute coordinate-change Jacobian owns only measure.
Root endpoints are ordered according to -q_outer/q_inner. Physical inner and
outer scales are retained, including nonunit boxes and reversed coordinates.

The coefficient C(t) is enclosed over the complete possible support. Write
C=A+(C-A), choosing an exactly represented constant anchor A. Integrate A*K
with the exact piecewise-polynomial primitive
P_m(t)=(1-t)^(m+1)/(m+1)! - sum_i w_i max(x_i-t,0)^m/m!.
Bound the remaining term by sup(abs(C-A)) times a certified absolute kernel
mass. Primitive evaluation retains endpoint uncertainty; no cancellation is
claimed for an unbounded coefficient or a sampled root. All value, slope and
curvature jumps, ordinary third-derivative terms, Decimal60 rounding and the
16-witness limit remain. Each cell has25 field nodes, five conditional axis
bounds, one integrated-axis bound and finite endpoint/derivative evaluations;
there is no hidden adaptive subdivision in this calculation.

Forty-five controls pass: exact kernel moments and polynomial primitives,
clipped roots in both directions, uncertain endpoints and varying coefficients,
exact diagonal clipped polynomials, twelve signed oblique log-material cases,
a curved coordinate map with its Jacobian, nonunit physical boxes and refusals.
The log reference uses an independent twice-integrated log polynomial, anchored
with zero value and slope at the knot. Its rectangle difference gives the
closed double integral; another renderer or a denser sampled grid is not the
accuracy authority. These are mathematical derivations and executable controls,
not an external Lean proof and not moving-parameter transport qualification.

At unchanged1e-12 radius and256cell/depth10 limits, two named oblique log cases
need7cells and387material-input calls each. The previous signed method needs
121cells/4043calls and133cells/4439calls. Timings and exact enclosure endpoints
are preserved in opaque72-20260920/analytical-improvement1.json and material3.xml.
The new method's extra work per cell is included in those totals. Those finite
improvements do not establish original-mesh effectiveness. Initial25missing-
module failures,16missing material-API failures,and16fixture operand-order
failures are retained with the successful reports.

D317,D319,D321 original campaigns remain terminal. No original job was launched
in this checkpoint. A fresh frozen campaign requires a concrete preview and
approval; all13matching certificates and their complete summed radius must pass
before native comparison. Accuracy, moving gradients, complete frames, remaining
1080p backward cases and the final original-mesh3D showcase remain open.


## Opaque73: original trial of the transverse-integrated candidate

User approval D323 resolved Q19. The frozen opaque72/original-campaign1 ran once
and stopped at its second piece. Piece0 passed133cells/46.26163170003565reference
seconds. Piece1 hit the90-second reference deadline after231attempted cells,
90.17693710001186reference seconds and91.76618159998907supervised seconds.
The fractional overshoot is observed cooperative-stop latency,not a raised cap
or an accepted result. Eleven pieces were not started. Total charged work is
364cells/136.4385688000475reference seconds/139.99807339999825worker seconds,
peak110829568B. No complete-pixel certificate or native accuracy comparison.

The last complete cover has229cells/115leaves. Its matte width is
1.87948084387455690487705121540676399235425018520802e-12,or
15.2516778646891times the allocated width. This is a different work point from
D321's255-cell bound;it is not a matched-work comparison. The first piece uses
the same133cells as before but more measured time. The exact synthetic benefits
therefore did not establish a sufficient original-case improvement.

The saved last-cell continuous-axis boundary radius is7.11332742246771e-16,
versus6.01778884279316e-18 for the conditioned-axis boundary term. Both ordinary
remainders remain included. This is a last-cell observation,not a full-piece
error decomposition. The first32trace boxes overlap and cannot be summed as
whole-piece evidence. Current logs do not distinguish integrated/fallback
witnesses or coefficient-variation versus endpoint-location uncertainty.

The next bounded investigation should expose that per-witness accounting on a
small fixed set of saved boxes,consider both tensor orientations,and compare
new integrated enclosures with the independently valid prior uniform bounds.
Any intersection/selection repair must first pass exact analytical controls and
cost checks. Do not launch an unchanged original retry,raise a cap or claim that
a proposed repair works before measuring it. All four campaigns are terminal.
Production renderer and installed wheel are unchanged. Original accuracy,moving
gradients,complete frames,four1080p backward cases,gffit supply and final3D
showcase remain open. Evidence:opaque73-20260920/campaign-audit1.json and
saved-diagnostic1.json;all original receipts and failures remain preserved.


## Opaque74: bounded witness diagnosis and endpoint/intersection repair

The fixed diagnostic uses saved failed piece1 trace cells30,31,32 and both
orientations,at most3boxes/2048correction input calls/20seconds. Preparation is
separately timed and its closure-internal calls are not counted as correction
inputs. The three boxes overlap;they are not a full-piece decomposition.
22witnesses admit transverse integration;none falls back. Root endpoint
primitive uncertainty dominates several wide terms. The diagnostic retains
roots,coefficient intervals,anchored primitive uncertainty,variation radii,
prior uniform bounds,per-witness admission and elapsed time.

For P'=K almost everywhere,P is absolutely continuous even across K1 jumps.
For an exact represented anchor a in interval T,the enclosure
P(a)+K(T)*(T-a) therefore contains P(t) for every t in T. Certified positive or
negative K permits endpoint monotonicity bounds as well. Intersect with the
natural polynomial enclosure. Each integrated signed witness is independently
intersected with the valid uniform bound;disjoint bounds refuse. Fixed tensor
ordering,physical scales,all value/slope/curvature jumps,ordinary third terms,
Decimal60 and all acceptance/resource limits are unchanged. This is a written
argument plus executable controls,not a Lean proof.

19diagnostic,14endpoint,45material and96affected controls pass. The exact
pre-first-node identity K3=t^3/6 supplies an independent polynomial check;
closed polynomial and log-material controls remain. Same1e-12radius log cases
still need7cells/387calls. The initially insufficient endpoint tightness is
repaired by certified monotonicity,without changing its threshold.

At the same322correction calls,the original-box diagnostic takes
1.7134114999789745s before and2.156060300010722s after. Axis1 total correction
widths tighten3.0606418220,2.6343368329,1.4530638875times. Axis0 tightens
4.9964940695,2.5582601533,17.4002500373times. The tensor direction is not changed.
Do not extrapolate these box results to a whole-piece success.19missing-API,
1fixture Fraction-endpoint construction,14test-first and1tightness failures
remain preserved. Evidence:opaque74-20260920/original-boxes2-reference.json,
original-boxes3-reference.json,improvement1.json and the matching XML reports.

D324 records the candidate. No new original campaign has run;all four old
campaigns remain terminal. Fresh numerical bytes require new matching receipts,
not reuse of earlier successful pieces. A concrete frozen trial preview and
explicit approval are required. Original accuracy,moving gradients,frames,
four1080p backward cases,gffit supply and final original-mesh3Dshowcase remain
open. The native renderer,installed wheel and accepted2D image demos are unchanged.


## Opaque75: approved endpoint/intersection original trial

User approval D325 resolved Q20 and executed the exact37file candidate once.
Piece0 accepted93cells/34.22430140001234reference seconds/35.75565690000076worker
seconds/113475584Bpeak. Piece1(face5740) failed the90sreference deadline after
222attempted cells/90.14759319997393reference seconds/91.71586409996962worker
seconds/113700864Bpeak;eleven pieces were not started. Cooperative stop latency
is an observed failure,not an increased allowance. Charged totals315cells,
124.37189459998626reference seconds,127.47152099997038worker seconds. No complete
certificate or native accuracy comparison. All five original campaigns are terminal.

Last complete cover:221cells/111leaves. Matte interval width
6.3383890757955066499726358469355540089654848833390e-13 versus allocated basis
radius6.16155435667164489328157280191951655650246391818375471613732e-14,or
5.14349846555552times allowed width. Prior D32315.2516778646891times was at
229cells/115leaves;this is not matched-work evidence. The first-piece improvement
and tighter saved bound do not close original-pixel acceptance.

Last-cell ordinary radius5.4370339540674352329e-16 accounts for87.6687323425percent
of its recorded correction radius;continuous-axis ordinary alone is83.2124019475
percent. Boundary radius7.6475978560665395691e-17 remains included. These are
one depth7cell,not the whole piece. The first32trace boxes overlap and the final
111leaf cover does not retain aggregate ordinary/boundary components or dominant
leaf coordinates. Do not attribute the whole-piece failure to the last cell.

Next:bounded diagnostic/control work to distinguish the ordinary smooth-part
remainder from boundary uncertainty and evaluation cost. Qualify any tighter
remainder on independent exact fixtures and a fixed small saved-box scope,with
all material joins and resource/accuracy limits retained. Add bounded complete-
cover component accounting to future justified runs;do not repeat a completed
expensive trial just to obtain status or reset a failed campaign budget. No
new full original trial is approved. Production renderer,wheel,accepted2Dimage
supply and showcase assets remain unchanged. M1 moving gradients,frames,four
1080pbackward cases,gffit supply and final original-mesh3Dshowcase remain open.
Evidence:opaque75-20260920/campaign-audit1.json and saved-diagnostic1.json;
exact trial remains opaque74-20260920/original-campaign1 with all receipts.


## Opaque76: bounded investigation of the ordinary remainder

The three saved D325 trace regions (cells 30, 31 and 32) are inspected in both
tensor directions. They overlap and cannot be summed into a whole-piece bound.
Simple constant centering improves the current-direction total correction only
1.01 to 1.04 times. Four fixed derivative subregions give a more useful result:
the connected candidate tightens those bounds 1.7814956725, 2.1482275781 and
1.7062969340 times. The other direction improves 1.6988390861, 2.0734812616 and
1.3048564755 times. Tensor direction and quadrature nodes are unchanged.

Gauss5 exactness on cubics implies integral K3 = 0. For a constant A, the
ordinary remainder integral f^(3) K3 therefore equals integral (f^(3)-A) K3.
Symmetry gives K3(1-t) = -K3(t), so each normalized half has half the certified
absolute kernel mass. Bound the regular third derivative on two inner halves
by two outer halves. The anchor may differ between OUTER halves, but must be
shared across the INNER halves of each full kernel integral. Independently
centering inner halves would be invalid. Rounded midpoint enclosures overlap
safely while the integration weights remain exact physical halves. Each of the
four contributions uses h_inner^4 * h_outer * L1(K3) / 4. Intersect with the
old valid ordinary bound. Keep all value, slope and curvature jumps, the
conditioned remainder, Decimal60 and the complete acceptance interval.

The baseline diagnostic uses 275 correction input calls; the connected candidate
uses 299, including 24 extra third-order derivative evaluations. Its measured
time rises from 1.7142252000048757 to 1.9872991000302136 seconds. Preparation is
separately timed and its closure calls are not in this count. All 18 boundary
witnesses and both orientations remain. This is a tighter-bound result, not a
speedup or a whole-piece success prediction. Two exact log fixtures retain
7 cells at the same 1e-12 radius; charged calls rise from 387 to 403.

60 smooth/material, 9 square/accounting, 45 transverse, 14 endpoint,
19 diagnostic and 31 campaign controls pass, plus the original-box diagnostic.
The controls include independent truncated-polynomial and log antiderivatives,
physical scaling, reversed and curved maps, jumps and stop propagation. All
37 test-first missing-API failures remain. This is a written argument and
executable qualification, not a Lean proof or moving-parameter qualification.

Future reference results retain component totals over the LAST COMPLETE cover
and at most five largest retained leaves. Parent evidence is replaced by its
children; an incomplete refinement cannot masquerade as a complete cover.
Missing components remain unknown, not zero. These are conservative error
bounds, not measured rendering errors. Storage remains capped at 256 leaves.

The independent full-material reference enables the candidate. Production
renderer and installed wheel remain unchanged. All five old original campaigns
are terminal; no full original trial ran here. A new frozen trial needs separate
approval under the same 13-piece maximum, 256 cells/depth 10/90 reference seconds
per piece and first-failure stop. No prior successful certificate may be reused
with the changed numerical source. Original accuracy, moving gradients, frames,
four 1080p backward cases, gffit supply and the original-mesh SIGGRAPH-quality
3D showcase remain open. Accepted 2D image demonstrations are unchanged.
Evidence: opaque76-20260920/improvement2.json and the original-boxes1/2/3 reports.


## Opaque77: approved four-subregion original trial

D327 resolved Q21 and ran the frozen 40-file candidate exactly once in
opaque76-20260920/original-campaign1. Piece0 passed 81 cells and
32.64487799996277 reference seconds (34.00216229999205 worker seconds,
113614848 B peak). Piece1, face5740, failed the 90-second reference deadline
after 202 attempted cells and 90.16103669995209 reference seconds
(91.55686529999366 worker seconds, 107495424 B peak). The small cooperative
stop overshoot is recorded failure latency, not a higher allowance. Eleven
pieces were not started. Charge all 283 cells, 122.80591469991487 reference
seconds and 125.55902759998571 worker seconds. No complete certificate or
native accuracy comparison; all six original campaigns are terminal.

The last complete cover has 201 cells and 101 retained leaves. Its matte width
is 5.5606970536132990701727390972895191857731032500105e-13 against allocated
basis radius 6.16155435667164489328157280191951655650246391818375471613732e-14:
4.5124141829506495 times the allowed width. D325 was 5.14349846555552 times
too wide at 221 cells/111 leaves. These are different covers, not matched-work
evidence. The first piece uses fewer cells, but complete acceptance still fails.

The new complete-cover accounting is available. Ordinary correction radius is
1.48727049375982362271383698169497231005572126249493216318606e-13; boundary
radius is 1.28659494927740366447163984545738998501197702230054284417076e-13.
Those comprise 53.6172544884% and 46.3827455116% of the recorded correction
radius. The smooth base contributes 0.2331752191% of total width. The five
largest retained leaves account for only 15.3005155481% of total width; this
is not one exceptional cell. These are conservative bounds, not observed
rendering errors. In the first four largest leaves, the conditioned-axis
ordinary term exceeds the integrated-axis ordinary term; this observation is
limited to those saved leaves, not a whole-cover directional decomposition.

Next: inspect both ordinary variation and boundary uncertainty on at most three
of these saved largest-leaf boxes, in both tensor directions, under the existing
20-second/2048-input diagnostic cap. Evaluate whether applying valid cancellation
to the conditioned term, selecting a justified tensor direction, or reducing
repeated derivative work can improve the full bound at a measured cost. These
are candidates to investigate, not accepted repairs. Require independent exact
controls for any change. Do not retry an unchanged trial, increase limits or
reuse old successful certificates. Another full original trial requires a new
frozen preview and approval. Production wheel and accepted 2D demos unchanged.
Original accuracy, moving gradients, frames, four 1080p backward cases, gffit
supply and the original-mesh SIGGRAPH-quality 3D showcase remain open.
Evidence: opaque77-20260920/campaign-audit1.json and saved-diagnostic1.json.


## Conditioned centering and bounded kernel reuse — 2026-09-20

D328 qualifies a targeted repair after inspecting both terms and tensor
directions on the three largest retained D327 leaves. Keep integrated axis1:
it is tighter on all three. At every fixed outer Gauss node, cubic exactness
gives integral K3 = 0 on the complete inner interval. Subtract one constant
anchor from its certified regular-third-derivative range and intersect the
variation enclosure with the existing absolute bound. Keep positive outer
weights, physical scale, all value/slope/curvature jumps, and the integrated
ordinary term. No quadrature ordering or material evaluation is changed.
Repeated local kernel calculations use exact Decimal endpoint keys and a
128-entry LRU of immutable endpoint tuples, returning fresh interval objects.

Correction widths in the retained direction tighten1.2142044182x,
1.1459071440x and1.4391166857x. The alternative direction improves only about
1.0090x,1.0055x and1.0017x. Both bounded diagnostics charge249inputs
(96order3,141order0,12order1) and8boundary witnesses; measured times are
1.4744364999933168 and1.2861982999602333seconds. All integrated and boundary
components are exactly unchanged. These are three disjoint leaves, not a
complete cover. The previous complete-cover excess of4.5124x remains terminal
evidence; whole-piece success cannot be inferred.

263 affected controls plus the original-region diagnostic pass. Two independent
closed log integrals meet unchanged1e-12radius in5cells/311inputs rather than
7cells/403inputs. Earlier acceptance can have a wider accepted enclosure.
The timing/work benefit combines less subdivision with cache reuse; cache
capacity and fresh-object isolation are qualified separately. Preserve12red
failures and1rational-endpoint fixture failure. This is executable independent
qualification and a written mathematical argument, not a Lean proof.

Evidence: D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/
opaque78-20260920/improvement1.json and its retained before/after reports.
No fresh whole-original trial has run. All six previous campaigns remain
terminal; a new frozen trial needs separate approval. Production wheel,
accuracy/resource requirements and accepted2Ddemonstrations are unchanged.
M1 original accuracy,moving gradients,frames,remaining scale cases,gffit supply
and the final original-mesh3Dshowcase remain unfinished.


## Opaque79: approved conditioned-centering original trial — 2026-09-21

D329 resolved Q22 and ran the frozen 41-file candidate once in
opaque78-20260920/original-campaign1. Piece 0 passed in 79 cells and
32.37522270006593 reference seconds (34.51405900006648 worker seconds,
109146112 B peak). Piece 1, face 5740, failed the 90-second reference deadline
after 204 attempted cells and 90.16615240008105 reference seconds
(91.78154330002144 worker seconds, 109899776 B peak). The small cooperative
stop overshoot is recorded failure latency, not an increased allowance.
Eleven pieces were not started. Charge all 283 cells, 122.54137510014698
reference seconds and 126.29560230008792 worker seconds. All seven original
campaigns are terminal. No whole-pixel certificate or native accuracy comparison.

The last complete cover has 203 cells and 102 retained leaves. Its matte width
is 5.0851572776412475232672320687842597393123953125505e-13 against allocated
basis radius 6.16155435667164489328157280191951655650246391818375471613732e-14:
4.126521477600137 times the allowed width. D327 ended 4.5124141829506495 times
too wide at 201 cells / 101 leaves. Different subdivisions and work prevent a
matched-work improvement claim. This is conservative reference uncertainty,
not a measured rendering error.

Ordinary correction radius is
1.30051136575715913689759026241202876255775515870612699179728e-13;
boundary radius is
1.23013719290158876587490322101766396100829164613438693614528e-13.
They represent 51.3904374951% and 48.6095625049% of correction radius.
On this fixed cover, either contribution alone exceeds the allowed radius:
ordinary 2.1106871586x, boundary 1.9964721914x. Removing only one therefore
cannot establish acceptance here. The base contributes 0.4692118458% of total
width and the largest five leaves 14.9858614016%; uncertainty is distributed.

Next inspect boundary coefficient variation/root uncertainty and the cost of
ordinary derivative enclosures on at most three saved regions, under the
existing 20-second / 2048-input diagnostic cap, with preparation separately
timed. Use matched boxes and independent exact material controls to qualify
any tighter bounds or reduced derivative cost before a newly frozen full trial.
Do not retry unchanged, increase caps or reuse certificates with changed source.
No new full trial is approved after D329. Production wheel, numerical acceptance,
worker limits and accepted 2D demonstrations are unchanged. Original accuracy,
moving gradients, complete frames, four 1080p backward cases, gffit readiness
and the original-mesh SIGGRAPH-quality 3D showcase remain open.
Evidence: opaque79-20260921/campaign-audit1.json and saved-diagnostic1.json.


## Opaque80: localized jump derivatives and repeated-work removal — 2026-09-21

D330 qualifies the targeted reference repair. The three largest saved D329
leaves were inspected in both directions under20seconds/2048correction inputs.
Integrated boundary coefficient variation accounts for96.0559to100percent of
witness radii; curvature-jump variation dominates. At each crossing, evaluate
spatial jets on the certified containing interval, keeping all jump orders,
original multiplicity,orientation,absence cases,physical scale and rounding.
This does not differentiate the root. An unchanged region reuses its inputs;
existing ambiguous-root fallbacks and curved monotonicity refusals remain.
Ordinary calculation shares axis inputs instead of repeating them. Absolute
kernel bounds use the same cached signed hull, with the existing128entry cap.

288affected controls plus the matched original-region diagnostic pass. In the
current axis1 direction total correction widths tighten1.3473748655x,
1.4651433956x,1.1558449388x;ordinary bounds/ranges are exactly unchanged.
Inputs240to233 (order3 96to89,order0 132,order1 12);8witnesses retained.
One cProfile diagnostic3.3333368999883533to2.9821259999880567seconds is
bounded instrumented evidence,not whole-piece throughput. Preserve25red
outcomes and two corrected pre-worker launch failures (registration syntax,
invalid square mode). No full original trial was run or approved.

Evidence: opaque80-20260921/inspection1.json,improvement1.json and retained
before/after reports. All seven original campaigns stay terminal;D329 last
complete cover remains4.1265times too wide. Next inspect ordinary derivative
enclosure dependency and evaluation cost on matched saved regions,at most3boxes
under20seconds/2048inputs,then independently qualify any change before another
frozen full-trial preview. No cap/tolerance change or old-certificate reuse.
Production wheel,accepted2Dimage supply and all remaining M1 gates unchanged.


## Opaque81: equivalent centered smooth branches — 2026-09-21

D331 qualifies an algebraically equivalent centered cubic for the smooth
branches used only by MaterialCorrection. For Bernstein controls a,b,c,d and
z=(ln(q)-x0)/(x1-x0)-1/2, the coefficients are
(a+3b+3c+d)/8, 3(-a-b+c+d)/4, 3(a-b-c+d)/2, -a+3b-3c+d.
All operations round outward; Horner evaluation preserves uncertain controls
and spatial chain derivatives. The helper refuses tangents and orders above3.
The horizon formula, branch differences, every jump order, multiplicities,
orientation and absence cases are unchanged. A two-entry immutable coefficient
cache includes all source endpoints in its key; outputs are fresh jets. The
separate smooth-base integral, production renderer and installed wheel are unchanged.

32 new exact log-polynomial derivative/cache/refusal controls and288 affected
controls pass. Preserve32 expected missing-module RED outcomes. The3 saved D329
regions were measured in both directions within20seconds/2048inputs. Current-axis
total correction widths tighten4.400258466803937x,4.156795549646438x,
3.4300419737054444x;ordinary components4.628814150187938x,4.296361770357428x,
3.690534728508855x;boundary components2.6972092950293396x,2.306109302821249x,
2.8131897620267883x. Same233calls and8witnesses/crossing roots. One profiled
diagnostic2.9821259999880567to2.3955438000848517seconds is local instrumented
evidence only. Exact derivative controls use separate100digit formulas; the
finite uncertain-control samples are tests,not a proof of continuum inclusion.
The algebraic identity plus outward interval operations supplies that argument.

Evidence: opaque81-20260921/improvement1.json,branch-green1.json and preserved
reports. All seven prior full campaigns remain terminal;D329 last complete
cover remains4.1265214776times too wide. Whole-piece benefit remains unmeasured.
Prepare a fresh frozen full-trial preview;explicit approval required before
execution,with no old-certificate reuse,no limits changed and first-failure stop.
All remaining original accuracy,moving-gradient,frame,1080pbackward,gffit supply
and final3Dshowcase gates remain. Accepted2Dimage supply is unchanged.


## Opaque82: centered-cubic trial clears the earlier blocker — 2026-09-21

D332 resolved Q23 and ran the frozen44file candidate once in
opaque81-20260921/original-campaign1. Piece0 passed57cells/19.085244499961846
reference seconds (20.66545239998959worker seconds,113602560Bpeak).
Piece1 passed223cells/81.40174060000572reference seconds
(82.98408870003186worker seconds,114049024Bpeak),clearing the region that
failed the preceding trial. Piece2,another region of face5740,
failed the90second reference deadline at222attempted cells and90.30529539997224
reference seconds (91.74330380000174worker seconds,113995776Bpeak).
The cooperative stop overshoot is failure latency,not a raised allowance.
Ten pieces were not started. Charge all502cells/190.7922804999398reference
seconds/195.3928449000232worker seconds. All eight campaigns are terminal;
no complete original-pixel certificate or native accuracy comparison.

Piece2's last complete221cell/111leaf cover has matte width
4.570925911157971125026762385010537600969922845450e-13 against allocated
basis radius1.53403954023524508153595208998923290728525327493350389080682e-13:
1.4898331468226103497times allowed width. This is conservative reference
uncertainty,not measured renderer error. The region differs from D329's failed
piece1,so its4.1265x prior bound is not a matched improvement comparison.

Ordinary correction radius1.52451232743354337084897065189438777916578957854411038978616e-13
and boundary radius5.86863101836522330787352873009746782368129938946303980394697e-14
represent72.2047percent and27.7953percent of correction radius. Against the
full allocated radius,ordinary consumes99.3789percent,boundary38.2561percent,
and smooth-base radius11.3483percent. The base is7.6172percent of total width;
the five largest leaves11.5028percent. No contribution may be omitted.

Next inspect the ordinary correction derivative dependence and cost on at most
three largest saved piece2 regions in both directions under20seconds/2048inputs,
with preparation separately timed. Preserve all three error contributions and
qualify any repair on independent exact material controls before another frozen,
separately approved trial. Failed-result SHA256:
b236f9c4eb4279e8d340b3a9a11e6cea76d79407e985397116561ffc46064c7d.
No unchanged retry,limit increase or reuse of certificates with changed source.
Native renderer,wheel,accuracy/resource rules and accepted2Ddemonstrations remain
unchanged. Remaining original accuracy,moving gradients,frames,four1080pbackward
cases,gffit readiness and final original-mesh SIGGRAPH-quality3Dshowcase are open.
Evidence: opaque82-20260921/campaign-audit1.json and saved-diagnostic1.json.


## Opaque83: third-piece derivative investigation — 2026-09-21

D333 records the bounded investigation, not an adopted mathematical repair.
Evidence: opaque83-20260921/findings1.json, inspection1.json, regions1-reference.json
and conditioned1-reference.json. The same three largest D332 piece2 regions were
inspected in both directions; current axis1 is tighter in all three. Two passes
used 7.592870499938726 diagnostic seconds and 642 correction inputs within the
combined 20-second/2048-input allowance; preparation was separately timed.
Baseline: 291 inputs,12 witnesses,2.9849087999900803seconds. Focused:351 inputs,
12 witnesses,4.607961699948646seconds. All returned numerical bounds, components,
third ranges and witnesses exactly match the baseline. Profile costs overlap
and are not whole-piece throughput measurements.

The conditioned ordinary term accounts for 88.7508%,89.8993%,97.6810% of the
ordinary error on these regions. On 15 current-direction lines/30 halves, only
the view term has a nonzero regular third derivative; light and mixed terms
are exactly zero.21/30 halves still span two view spline segments. Hypothetical
uniform halving tightens total correction widths by 1.200312183611655x,
1.0208240068257548x,1.0402793072669732x. It is not adopted. The extra60order3
evaluations across both directions include explanatory decomposition, not a
production overhead measurement. An initial postprocessing equality assertion
used a different Decimal precision from the directed interval sum; corrected
using the original outward addition with no numerical rerun. Preserve
analysis-correction1.json alongside both original diagnostic reports.

Next qualify segment-aware signed regular-third-derivative remainder integration
on conditioned lines using certified angular-crossing intervals. Retain each
segment anchor times its signed kernel integral: a subinterval does not inherit
the full-line zero-mass identity. Bound uncertain endpoints and within-segment
variation; retain integrated ordinary/base terms and every value/slope/curvature
jump. Use conservative fallback for ambiguous or excessive crossings under
unchanged limits. Start with independent exact piecewise-cubic/log-material
controls covering increasing/decreasing directions, endpoint and uncertain
crossings, absent crossings, cancellation and resource bounds before adoption.

No complete-piece success, renderer-error claim or new full trial follows from
these diagnostics. All eight campaigns remain terminal. Original accuracy,
moving-gradient,frame,1080pbackward,gffit supply and final3Dshowcase gates remain;
accepted2Dimage supply and the installed wheel are unchanged. Any fresh full
trial needs a new frozen preview and explicit approval.


## Opaque84: signed conditioned segment remainders — 2026-09-21

D334 qualifies an optional fixed-geometry development-checker path, now enabled
by the original-material producer. Given physical domain [a,b], h=b-a, and
regular third derivative g, the ordinary error is h^4 integral K3(t)g(a+ht)dt.
For one exact full-domain anchor A, integral K3 A=0. Partition [a,b] at every
certified crossing interval endpoint. On each slab retain the signed kernel
integral times a local constant, plus an outward variation enclosure. Cover all
root uncertainty zones; no subinterval inherits the full-line zero-mass identity.
Physical slabs have exact Decimal endpoints and at most16pieces. Above that
count retain the old bound before extra inputs. Checkpoints apply to each slab.

Angular branch restrictions apply only on open slabs outside a guaranteed
strictly monotone crossing interval. Optional crossings retain all candidates;
nonmonotone witnesses retain the prior fallback. Endpoint values have measure
zero in the regular integral; every value/slope/curvature jump remains separately
charged. Intersect the conditioned signed remainder with the existing centered
conservative bound; refuse contradictory evidence. Keep the continuously
integrated ordinary term and smooth base unchanged. No native renderer,wheel,
acceptance tolerance or resource limit changed. Spatial derivatives here do not
qualify moving-geometry or knot-transport parameter gradients.

43new independent/integration checks and280affected controls pass. Retain all
test-first history:36initial RED results;34pass/2reference-precision assertions;
36pass/1optional-crossing safety regression;1original-producer binding RED.
The two precision assertions compared different outward60digit enclosures;
separate100digit Gauss algebra fixes that test requirement without changing
acceptance tolerances. The optional-crossing failure was repaired by requiring
guaranteed crossing presence before selecting a branch. Curved maps retain their
Jacobian; all four directions enclose a separate closed log-material integral.
The tests are not proof-assistant certificates. Development scripts and all
reports remain under opaque84-20260921/.

Matched3saved D332piece2regions,both directions:current-axis total correction
widths improve2.479612289896722x,2.668028498955696x,8.584468780648168x.
Integrated bounds,jump bounds and12witnesses remain exact unchanged. Diagnostic
inputs291to387;instrumented time2.9849087999900803to5.595460599986836seconds.
Candidate worker7.411632300005294seconds/100012032B. Preparation separately timed;
diagnostic stays below20seconds/2048inputs. Local per-evaluation cost increased;
fewer whole-piece refinements may offset it but that is unmeasured.

Evidence: opaque84-20260921/improvement1.json,segment-green2.json,additional1.json,
regression1.json and regions1-reference.json. All eight campaigns remain terminal.
A fresh frozen full-trial preview requires explicit approval,uses fresh
certificates,unchanged13piece/256cell/depth10/90reference-second limits and stops
at first failure. Do not restart D332 or reuse its certificates with this source.
M1,original accuracy,moving gradients,complete frames,four1080pbackward cases,
gffit supply and the final original-mesh3Dshowcase remain open. Accepted2Dimage
supply is unchanged;neither consumer implementation is in scope.


## Opaque85: segmented remainder trial exposes a cost regression — 2026-09-22

D335 resolved Q24 and ran the frozen47file candidate once in
opaque84-20260921/original-campaign1. Piece0 passed47cells/24.213282399927266
reference seconds (26.76720150001347worker seconds,112779264Bpeak).
Piece1,face5740,failed the90second reference deadline at158attempted cells
and90.1573906999547reference seconds (92.02283989998978worker seconds,
114155520Bpeak). The cooperative stop overshoot is failure latency,not a
raised allowance. Eleven pieces were not started. Charge205cells,
114.37067309988197reference seconds and118.79004140000325worker seconds.
All nine campaigns are terminal;no full original-pixel certificate or native
accuracy comparison was produced. Numerical source stayed frozen throughout.

The last complete157cell/79leaf cover has matte width
1.6337825442513454800958764994269382315019187843083e-13 against allocated
basis radius6.16155435667164489328157280191951655650246391818375471613732e-14:
1.325787658176145592423893057983395695940639410630706144340073647775646times
allowed width. This is conservative reference uncertainty,not observed image error.
Ordinary radius1.55352556774646913590442758633544217272083949823736682738655e-14
and boundary radius5.53209704276638420142623866993212601554427121087479675353581e-14
are21.9250percent and78.0750percent of correction uncertainty. Relative to the
entire allowed radius:ordinary25.2132percent,boundary89.7841percent and
smooth-base17.5814percent. None may be dropped. Five largest leaves account
for20.6842percent of total width;the base contributes13.2611percent.

D332 passed this same piece1in223cells/81.40174060000572seconds. Its adaptive
cover differs,so this is an outcome/cost regression,not a matched-box bound
comparison. Piece0previously passed57cells/19.085244499961846seconds:the new
method used fewer cells but more time. D334's local2.48x/2.67x/8.58x bound
improvements were measured on piece2,which D335 did not reach. The algebraic
enclosure qualification remains;whole-trial cost/readiness is not admitted.

Next inspect the saved D335piece1largest regions and per-evaluation cost in both
directions,at most3regions/20seconds/2048correction inputs,preparation separately
timed. Compare segmented and conservative paths on matched boxes. Determine
whether guaranteed/optional crossing support,uncertainty-zone derivatives or
kernel work dominates before qualifying selective application or exact reuse.
Retain all boundary,jump,ordinary and base contributions. Independent exact
polynomial/log-material controls precede a newly frozen separately approved
full trial. No unchanged retry,old-certificate reuse or resource increase.
Failed-result SHA23912aacbbb436d8cc229125738227a69e0b8953eac0707a4ccf7007863a4449.

Native renderer,wheel,accuracy/resource limits and accepted2Ddemonstrations
remain unchanged. Original accuracy,moving gradients,complete frames,four1080p
backward cases,gffit readiness and the final original-mesh SIGGRAPH-quality3D
showcase remain open. Evidence:opaque85-20260922/campaign-audit1.json and
saved-diagnostic1.json. Preserve every earlier trial and its failures.


## Opaque86: exact local reuse, without a throughput claim — 2026-09-22

The D335 piece1 diagnostic compares the three largest saved leaves in both
integration directions. Conservative and segmented correction bounds improve
by 1.0287–1.5962 times on matched boxes. All 39 observed crossing witnesses
are required; optional-crossing selection does not explain this sample's cost.
Segmented evaluation uses 104 slabs, including 39 root-uncertainty zones.
The instrumented baseline takes 3.596994200022891 seconds conservative then
5.820664600003511 seconds segmented. The initial diagnostic totals 748 inputs
and 9.420368400053121 seconds, including separately reported preparation.

The targeted repair reuses exact third-order input jets within one axis-bound
call, with at most 16 stored immutable endpoint tuples. Keys include every
input coefficient, order and endpoint representation. Returned hits are fresh
jets. Every call, including hits, checks cancellation. No reuse survives the
call or crosses preparation/geometry changes. Retain every ordinary, jump,
boundary and base contribution and the existing 16-slab fallback.

Twenty new controls pass, including independent exact log-material integrals,
polynomial jets, changed-input invalidation, mutation isolation, bounded storage
and cancellation; 280 affected regressions and 43 segmented controls pass.
Preserve the 20 expected preimplementation failures and the subsequent one
mixed-order test-fixture failure (19 passed) before correcting both test input
orders. No numerical acceptance condition was weakened.

The repaired saved-region diagnostic uses 387 correction inputs instead of 426,
with third-order calls reduced from 218 to 179. All saved numerical fields in
all six region/direction results match exactly. The two diagnostic executions
charge 1135 inputs and 15.633656900026836 seconds against the aggregate
2048-input/20-second limit. The repaired run takes 6.213288499973714 seconds.
Do not claim a speedup: the baseline segmented run followed a conservative run
with warmed bounded kernel caches, whereas the repaired run starts a fresh
worker, and both are instrumented. Kernel evaluation remains significant.

Next make a controlled cold-cache comparison and inspect exact kernel-work
reuse before another full trial. Keep caches bounded and test-first, preserve
all uncertainties, and freeze any new trial for explicit approval. All nine
earlier campaigns remain terminal. Native runtime/wheel, numerical acceptance,
resource limits, history and accepted 2D demonstrations are unchanged. M1,
original accuracy, moving gradients, complete frames, four 1080p backward
cases, gffit readiness and the final 3D showcase remain unfinished.

Evidence: opaque86-20260922/diagnostic1-reference.json,
diagnostic2-reference.json, improvement1.json, reuse-red1.json,
reuse-green1.json, reuse-green2.json, regression1.json and segmented1.json.


## Opaque87: crash recovery and controlled kernel-cost comparison — 2026-09-23

After the reported PC crash, no dashboard or numerical worker was running.
SQLite integrity passed and all 1056 runs/31751 tests survived. A stopped
594-file/317802708-byte database-and-assets backup was hash verified before
startup. Local and phone Tailscale URLs returned HTTP200. All864 source paths,
151runtime inputs and74installed members match the prior checkpoint; the wheel
is unchanged. Preserve the unrelated new chronobrains PLAN.md edit, giving
252devbrain dirty entries, and all1386existing source entries.

The authorized diagnostic used uncached/reuse/reuse/uncached order, clearing
the bounded local kernel cache, global kernel-bound cache and Gauss-node cache
before each pass; material coefficient and smooth-branch caches were reset too.
All four passes examined the same three saved D335piece1regions in both directions
with identical clock instrumentation and no cProfile. Preparation was charged
and separately timed. Aggregate charge:9.44550479999998seconds/1626correction
inputs, below20seconds/2048inputs. Every numerical field remained identical.

Uncached timings:2.3978441999997813 and2.4123186999995596seconds, mean
2.4050814499996704. Reuse timings:2.3277514000001247 and2.2978231000006417,
mean2.312787250000383. The measured local time reduction is3.837466710297899percent.
This is two observations per path in one worker, not a statistical throughput
guarantee or a complete-piece performance result. Calls remain426uncached and
387reused, including218versus179third-order inputs. Preparation is about0.0242s
per pass. D336exact reuse is retained; the controlled result supersedes only the
earlier inability to compare timing, not the historical measurements.

Each pass has151local kernel-range cache misses and152hits at unchanged128entry
capacity. Miss contexts:104in weighted third-order work,32in weighted first-order
work and15elsewhere. Local misses cost about0.536–0.548seconds per pass.
Weighted-kernel total inclusive time is about0.563–0.573seconds; these costs
overlap and must not be added. All152complete weighted geometry/order keys are
unique within each pass, so caching complete weighted results offers no observed
reuse here. Source inspection shows each local miss computes K1,K2,K3even where
only K3is selected. The104third-order-context misses identify a more useful
candidate: calculate only requested local orders, retaining shared all-order
callers, outward endpoint arithmetic, every kernel segment and current bounds.

No kernel optimization is implemented by this investigation. Next qualify a
requested-order helper test-first against independent exact polynomial/Peano
identities, uncertain endpoints/Gauss-node boundaries, mutation isolation,
cache identity/storage and stop behavior. Preserve the existing full tuple
interface and total128entry capacity rather than introducing an extra cache.
Then compare identical numerical fields and controlled cost under the bounded
diagnostic before previewing any full trial. All nine prior campaigns remain
terminal. No acceptance/resource relaxation, new full trial or Git publication.
M1 and the original-mesh SIGGRAPH-quality3Dshowcase remain unfinished.

Evidence:opaque87-20260923/controlled1-reference.json,controlled1.json,
findings1.json,reconciliation1.json,reconciliation-extra1.json and dashboard-backup1.json.


## Opaque88: requested-order kernel bounds with shared position storage — 2026-09-23

The D337 checker repair is now locally qualified. The original outward
polynomial/Bernstein arithmetic and global intersections are retained for each
requested order. Full-range callers retain the same tuple interface. One bounded
LRU still holds at most128positions, with at most three privately cached immutable
endpoint pairs per position, filled only when needed. Returned intervals are fresh;
failed fills retain completed slots. Weighted callers request only their order.
Runtime, installed wheel, tolerances, endpoint uncertainty and resource caps are unchanged.

TDD evidence preserves49initial missing-helper failures. The first repair passed
49controls; its broader regression had341passes and2old observer-signature
failures, repaired without changing numerical assertions. Two compatibility checks
then passed. The initial cold-cache four-pass diagnostic preserved all bounds but
exposed cache fragmentation: order-key caching raised misses151to255and reduced
hits152to48. Its source is saved under candidate-before-lazy1. The shared-position
lazy-slot repair was driven by one further red control. Final51controls and343
regressions pass, including independent100-digit closed Peano identities, uncertain
Gauss-node sides, all orders, mutation isolation, bounded storage and failed fills.

The final single pass on the same three saved regions/both directions exactly
matches every numerical field of the saved all-order comparison. Cache behavior
returns to151misses/152hits, capacity128. Polynomial order evaluations fall from
453to323. There are293fill calls, including15calls computing all three orders;
fill-call counts must not be confused with cache-position misses or order counts.
Final diagnostic time is2.2767125999998825seconds, kernel-fill inclusive
time0.4323703999962163seconds. The earlier two all-order
observations and this one final observation are insufficient for a reliable
whole-piece speed claim. Do not reuse the report's generic two-observations-per-path
label as a description of controlled2: its explicit order lists one requested pass.
All five diagnostic passes together charge12.441824200000156seconds and
1935inputs against the unchanged20seconds/2048inputs allowance.

The next action is a frozen candidate/driver/input preview for a fresh13piece
campaign, stop at first failure,256cells/depth10/90reference seconds per piece,
120worker seconds/180watchdog seconds/1GiB/one thread, aggregate3328cells,
1170reference seconds/1560worker seconds. No old certificate reuse and no new
trial without approval of that frozen preview. Original-mesh accuracy and M1
remain unfinished; all nine earlier campaigns and every failure remain preserved.

Evidence:opaque88-20260923/findings1.json,controlled1-reference.json,
controlled2-reference.json,kernel-red1.json,regression1.json,lazy-red1.json,
kernel-green2.json,regression2.json and qualification1.json.


### Opaque88 frozen preview approved and executed — 2026-09-23

The user approved the displayed53file candidate/driver/input trial explicitly.
It ran once under D338 and is now terminal. The earlier approval-pending
description above is historical and grants no permission to repeat this trial.

D338 approved opaque88 trial is terminal: piece0 passes47cells/21.13046619999932reference seconds; piece1 passes173cells/88.14543289999983seconds; piece2face5740 fails90second deadline at144attempted cells/90.2087548999998seconds;10unstarted. Total364cells/199.48465399999895reference seconds/203.79644820000067worker seconds;113340416Bpeak. Last complete143cell/72leaf matte enclosure is2.3911930121818667621times allowed width. Base,ordinary,boundary uncertainty consume respectively 0.7626587919847149818683022108553865082732354968934033642837096053823598,0.6296717102342075921498986519612539420413280607244390681888359989490603,0.9988625099629441880999782244209806421353893980132605924240100083598972 of the allowance. All ten campaigns now terminal; no complete13piece certificate or native accuracy comparison. The second piece now passes, but whole original accuracy remains unqualified. Preserve failed work and stop latency; no restart or budget increase.

The third piece's retained complete uncertainty splits into31.8945percent base
enclosure,26.3330percent ordinary remainder and41.7726percent boundary remainder.
Each term remains necessary; the boundary component alone nearly consumes the
whole allowance. The five largest leaves contribute26.2091percent of width.
This is conservative uncertainty rather than observed renderer error. The two
passed pieces are genuine progress but do not constitute a full pixel certificate.

Next inspect the three largest saved third-piece regions in both directions,
separating base/ordinary/boundary enclosure widths from evaluation cost. Keep a
bounded20second/2048input investigation, then qualify any targeted change against
independent controls before a newly frozen, explicitly approved campaign. Do not
retry this terminal campaign, reuse old certificates under changed source, remove
uncertainty terms, relax tolerances or raise limits. Native comparison was not
performed because its required complete independent certificate does not exist.

Evidence:opaque88-20260923/campaign-preview1.json,original-campaign1/terminal.json,
campaign-audit1.json,saved-diagnostic1.json,piece-00.json,piece-01.json,piece-02.json.


## Opaque89: exact arithmetic work and high-order centered smooth base — 2026-09-23

Reconciled867source paths,151runtime inputs,74installed members and all five
project-memory hashes. Concurrent chronobrains publication explains two cleaned
status entries; no gffx bytes or unrelated work were changed by that publication.
The dashboard was already running on internal storage. No Git transaction here.

The saved D338third-piece diagnostic covers three largest retained regions in both
directions. No single direction consistently wins. Profiled correction time is
6.276674999999159seconds/428inputs. Red controls then drive sign-selected interval
product extrema and skipping absent tangent recurrences. Directed60digit rounding
and every present tangent, including1e-80, remain.107arithmetic controls pass;
343affected controls and110independent smooth/moving/curved/kernel controls pass.
The same diagnostic now takes5.716701299999841seconds/428inputs and every numerical
field is exact. These are instrumented local observations, not whole-piece speed.

The base integral was still using Bernstein products. Its mathematically identical
centered cubic is now independently qualified through order10 (124controls),
including110digit log-series coefficients, uncertain control vertices, a closed
nonlinear log-antiderivative integral, horizon and refusal behavior, and producer
binding. The lower-order interface still refuses unqualified orders. Base and
correction share the same two-entry branch cache; no additive cache or raised cap.
The Gauss5 rule,order10 remainder,all knot corrections and tolerance stay unchanged.

On the same three saved regions,base matte widths fall from
1.8385955279e-14/1.0631762866e-14/4.4191979326e-15 to
1.4801444173e-24/5.9506205603e-25/1.4573875085e-25.
Old/new base times are0.2028/0.0999,0.1923/0.0961,0.2280/0.1107seconds.
This tightens the base component only; boundary and ordinary correction remain.
All diagnostic work charges12.965852999999697seconds/1019counted calls under
20seconds/2048calls, including a failed operand-order diagnostic (0.0091161s/1call).
The failed diagnostic JSON's initial status label was misleading; its failed JUnit
and incomplete variants are authoritative, preserved and superseded by the passing
base-diagnostic2 result.11arithmetic-red+122base-red+1binding-red+1diagnostic failure
are retained. A usage-limit approval-review interruption executed no command.

No native/runtime/wheel change. All ten earlier campaigns remain terminal. Prepare
a fresh frozen candidate/driver/input preview; run only with explicit approval,
same13piece/256cell/depth10/90reference-second/120worker-second/180watchdog profile,
1GiB/one thread and first-failure stop. No old certificates or allowance increases.
Original accuracy and M1 are not admitted by these local comparisons.

Evidence:opaque89-20260923/findings1.json,diagnostic1-reference.json,
diagnostic2-reference.json,base-diagnostic2-reference.json,base-final1.json.


## Opaque90: signed ordinary bounds and separate refinement guidance — 2026-09-23

Opaque89 D340 is terminal: one accepted piece, second piece deadline failure,
eleven unstarted. All275cells/133.3323719reference seconds/136.1296585worker seconds
are charged; peak114036736B. Last complete177cell/89leaf bound1.285486times allowance.
No complete13piece certificate or native accuracy admission. The tighter base
changed scheduling; on11shared early opaque84/opaque89boxes correction widths agree
exactly but6split directions change. Three current largest-region midpoint
comparisons show the current correction hint chooses the better axis on all3.
Do not reverse it globally. A failed diagnostic compared child-specific residuals
incorrectly; complete base-plus-correction comparisons repair that harness.

D341 qualifies signed ordinary integration from the same four derivative boxes.
Each inner half keeps its nonzero signed K3 mass and variation enclosure, with
physical scale included once. Intersect the prior conservative bound and retain
all value/slope/curvature jumps.41focused+280regression+167additional controls pass.
Local width gains are modest, approximately0.015/1.37/5.01percent; no complete-piece
admission follows from them.

Separate refinement guidance now evaluates the historical Bernstein derivative
scale twice per cell, retaining the centered base and complete correction bounds
for acceptance. The guide can affect priority and split direction only when its
priority is at least the actual bound width. Missing guidance falls back; stops
propagate. Seven controls include closed nonlinear integrals/tangents, zero/tiny
guidance refusal to falsely admit, invalid guidance, unchanged one-cell evidence
and original-producer binding.110numerical and167producer/branch controls pass.
The first guidance version starved an axis in2controls; that failure is retained
and repaired by the actual-priority guard. No accuracy condition was weakened.

All diagnostic work uses the same three saved regions and charges
8.143082600001435seconds/2003calls under20seconds/2048calls, including failed work.
Six additional order10guidance fields take0.2627828seconds including preparation;
the three field-pair timings total about0.24seconds. This is additional cost,
not a demonstrated whole-piece speedup.51red/harness failure outcomes remain.

No native/runtime/wheel change, new cache, allowance increase or Git transaction.
A frozen new candidate must be explicitly approved before one fresh serial trial.
All11prior campaigns are terminal; no old certificate reuse. Original accuracy,
moving gradients,complete frames and final actual-mesh3Dshowcase remain open.
Evidence:opaque89-20260923/campaign-audit1.json;opaque90-20260923/split-response2-reference.json,
signed-diagnostic1-reference.json,guidance-cost1-reference.json and qualification reports.


## Opaque91: identical arithmetic with fewer temporary objects — 2026-09-23

D343opaque90trial is terminal: two accepted pieces; third-piece90second deadline
failure;ten unstarted. Charge364cells/187.6208488000011reference seconds/
191.66851700000007worker seconds;114290688Bpeak. Piece0passes43cells/19.094108seconds,
piece1passes163cells/78.3710245seconds. Piece2attempts158cells; last complete157cell/
79leaf bound1.05805651658608times allowance. This remains a failure, not renderer
error or native admission. All12previous campaigns remain terminal.

Reduce repeated local denominator work: reuse the identical outward reciprocal
through each reciprocal/square-root jet recurrence, and use one rounding context
for the two interval reciprocal endpoints. Read existing interval operands without
constructing temporary copies. Retain zero-addition fresh-return behavior, input
immutability, Decimal60directed endpoints and every live tangent, including1e-80.
No global cache, new retained state, native/runtime/wheel or numerical-cap change.

Red controls expose12repeated-denominator and3operand-copy cases. Final147arithmetic,
225targeted,280boundary/material and110independent moving/curved/kernel controls pass.
The prior absent-tangent observer now counts multiplications instead of divisions,
because reciprocal reuse deliberately changes division counts; it still proves
that a tiny live tangent executes more recurrence work than an absent tangent.

Same three D343saved regions, both directions: every numerical field is exact.
Profiled correction time4.9311892to4.6320457seconds (about6.1percent lower).
Initial denominator-only high-order A-B-B-A comparison was3.8percent slower;
retain that unfavorable result. After operand-copy removal, full pinned-old versus
new arithmetic high-order base/guidance A-B-B-A means are0.4054375/0.3574792seconds,
about11.8percent lower. Two observations per path are not a whole-piece guarantee.
All diagnostic passes charge17.42564499999935seconds/1233calls under20/2048.

Prepare a fresh64file candidate/driver/input preview and await explicit approval
before any new13piece serial trial. All limits and first-failure stop unchanged;
no old certificate reuse. Original accuracy and M1 remain unfinished.
Evidence:opaque90-20260923/campaign-audit1.json;opaque91-20260923/findings1.json,
diagnostic1/2/3-reference.json,cost1/2-reference.json and qualification reports.


## Opaque92: qualify the alternate tensor order — 2026-09-23

D345 opaque91 is terminal: three accepted pieces (43/163/163 cells;
18.2300319/76.5224724/88.8164094 reference seconds), then the fourth stops
at its 90-second deadline. Nine pieces remain unstarted. Charge all 530 cells,
273.77766730000076 reference seconds and 279.26748340000086 worker seconds;
peak host memory 113111040 bytes. The last complete fourth-piece cover has
159 cells/80 leaves and matte width 4.069129619695478 times allowance.
No complete certificate or renderer-error conclusion follows.

Inspect both tensor orders on the three largest saved fourth-piece leaves.
QxQy-IxIy can be decomposed as Qy(Ex)+Ix(Ey), or Qx(Ey)+Iy(Ex).
Both are exact identities; their independently bounded remainders differ in width.
An explicit integrated_axis selector keeps field quadrature and physical-axis
error reporting unchanged. The independent original13 profile selects axis0 for
piece index3 only; all other pieces retain axis1. This is a profile-specific
choice based on saved evidence, not a general optimal-order heuristic.

62 analytical and control cases pass after 62 preserved red failures. They cover
both orientations, log-material rectangles, clipped value/slope/curvature jumps,
curved Jacobians, nonunit domains, callback axes, invalid-axis refusal and all13
producer selections. 280 boundary/material,225 targeted and110 independent oracle
regressions pass. On the same three saved regions, full correction width ratios
are 0.23646617081658688,0.1738726266668346,0.7265657381926348. Both-order work,
including the initial profiled diagnostic, charges 7.998966500001188 seconds and
972 inputs within20 seconds/2048inputs. Local per-region times are roughly
0.33–0.40seconds; no whole-piece cost or success is inferred.

Prepare a fresh66file candidate and exact frozen driver/input preview. Every
certificate must be fresh after this source change; all13 earlier campaigns stay
terminal. Limits, first-failure stop, numerical accuracy and native wheel remain
unchanged. Original accuracy and M1 remain unfinished.
Evidence: opaque91-20260923/campaign-audit1.json;
opaque92-20260923/diagnostic1-reference.json,order-diagnostic1-reference.json,
findings1.json and qualification reports.


## Opaque93: reduce exact arithmetic bookkeeping and zero-term work — 2026-09-23

D347 opaque92 is terminal. The first three pieces passed at 18.440997,
77.0187373 and 89.0175165 reference seconds. The fourth stopped at its
90-second deadline after 153 attempted cells; nine pieces were not started.
Its last complete 151-cell/76-leaf matte bound was 3.412435034287118 times
allowance, improved from 4.06913 but still not accepted. Charge all 522 cells,
274.62880629999745 reference seconds and 280.3050094000009 worker seconds;
peak host memory 114020352 bytes. There is no complete certificate or native
accuracy admission. All fourteen campaigns remain terminal.

The three largest saved fourth-piece leaves were checked with both midpoint
split directions using full base-plus-correction bounds. The current scheduler
chose the better split in all three cases, reducing each local width about
7–14 times. Retain the scheduler and fourth-piece tensor-order selection.

Reduce bookkeeping without changing the arithmetic: transfer internally owned
fresh jet coefficients instead of copying them again; preserve copying for
external coefficient dictionaries and for addition paths that would otherwise
alias an operand. Use one directed-rounding context for interval addition and
accept already-Decimal endpoints without redundant conversion, retaining finite
and ordered endpoint checks. Mutation-isolation, independent directed-endpoint,
legacy coefficient, sign, extreme-scale and tiny-tangent controls pass.

For angular corrections, evaluate both residual jets first. Only an empty
coefficient dictionary proves a term identically zero. Evaluate smooth branches
only for live residual terms, retaining every tiny or high-order coefficient.
Keep the same physical factors, denominator validation and live-term arithmetic
order. The change introduces no cache, raised precision, numerical relaxation,
native/runtime modification or resource allowance.

185 arithmetic, 10 zero-term, 280 boundary/material, 225 targeted, 110 independent
oracle and 62 tensor-order controls pass. Preserve 14 ownership/context red
failures and 6 zero-term red failures. Three saved full bounds remain exact after
the arithmetic repair: aggregate measured evaluation time 1.2678212 to 1.0736939
seconds, about 15% lower. One final region after both repairs also remains exact:
0.3898471 to 0.3093049 seconds, about 21% lower. These are single observations,
not a demonstrated whole-piece speedup. All diagnostic passes together charge
8.07316190000347 seconds and 2040 inputs under the unchanged 20/2048 limits.
Do not repeat or extend this nearly exhausted original-region investigation.

Prepare a fresh 69-file candidate and driver/input preview; obtain explicit
approval before a new serial thirteen-piece trial. All certificates must be
fresh. Original pixel accuracy and M1 remain unfinished; remaining moving
gradients, complete frames, scale cases, consumer-readiness and actual 3D showcase
gates remain open. Implement neither consumer.
Evidence: opaque92-20260923/campaign-audit1.json; opaque93-20260923/findings1.json,
split1-reference.json,owned-diagnostic1-reference.json,owned-diagnostic2-reference.json
and the preserved qualification reports.


## Opaque94: preserve directed arithmetic with one convolution context — 2026-09-23

D349 opaque93 is terminal: three pieces passed and the fourth reached its
90-second reference limit after 209 attempted cells. Its last complete cover
used 207 cells/104 leaves; the matte width was 1.7808848197591655 times allowance,
down from 3.412435034287118 in opaque92. Nine pieces remain unstarted. Charge all
578 cells, 234.95199280000088 reference seconds and 240.4677000000047 worker
seconds; peak host memory 114003968 bytes. This is insufficient independent
verification, not an established renderer error. All fifteen trials are terminal.

The next qualified checker change gives each jet coefficient convolution one
local Decimal context instead of opening contexts for every interval product
and addition. Each product is still rounded outward separately before its
separate outward addition; convolution order and precision60 are unchanged.
This is not Decimal fused multiply-add or reassociation. Structural zero handling,
tiny live tangent coefficients, external mutation isolation, caller context and
order-mismatch refusal remain. No new cache or persistent storage is introduced.

117 new controls pass, including exact endpoint tuples against the original
convolution through order10, independent rational polynomial products, small
live tangents, sparse/dense/uncertain coefficients and low ambient precision.
185 arithmetic, 10 zero-term, 280 material/boundary, 225 targeted, 110 independent
oracle and 62 tensor-order controls also pass. Nine historical red failures are
retained. An old/new/new/old synthetic expression comparison preserves exact
coefficient bounds; timings are recorded in findings1.json and do not establish
original-piece throughput. No original-region diagnostic was run: opaque93's
2040/2048-input diagnostic remains closed. Both time and cell limits may still
prevent admission; the new trial must fail honestly if either is exhausted.

Freeze a fresh70-file candidate and obtain approval before its full serial trial.
All limits and acceptance criteria remain unchanged; no native or wheel changes.
All13 fresh certificates and a matching native comparison are still required.
M1, moving gradients, complete frames, the remaining scale cases, gffit supply
and the final actual original-mesh SIGGRAPH-quality showcase remain unfinished.
Evidence: opaque93-20260923/campaign-audit1.json and dashboard-audit2.json;
opaque94-20260923/findings1.json and its preserved qualification reports.


## Opaque95: choose the tensor identity per fourth-piece region — 2026-09-23

D351 opaque94 is terminal: three pieces passed, the fourth reached its90-second
reference deadline at237attempted cells, and nine remain unstarted. Last complete
235-cell/118-leaf matte uncertainty was1.3135664883014955times the allowed width.
All606cells/220.6811258000016reference seconds/226.26208079999924worker seconds
are charged; peak114376704bytes. All sixteen previous campaigns are terminal.
Near the256-cell ceiling, another speed improvement alone has no demonstrated
admission guarantee. Failed verification is not an established renderer error.

Inspect the three largest new saved leaves under a fresh, separately recorded
20-second/2048-input diagnostic; the opaque93 diagnostic remains closed. Both
ordinary and boundary contributions matter. Continuous integration along axis0
is better in the first region, while axis1 is substantially better in the next
two. This refines the static fourth-piece selection of D346, not either identity.

For the fourth original piece only, compute both continuously integrated bounds
once per box, use their maximum channel widths as a selection score, then reuse
the selected rows in the complete corresponding tensor identity. Retain all25
quadrature nodes and five conditioned-line bounds. The score is a heuristic,
never an acceptance bound; the complete selected interval alone owns acceptance.
Unsupported alternatives may fall back, but cancellation always propagates.
Ties retain axis0. Other twelve pieces retain fixed axis1. No persistent cache,
native change, approximation, tolerance reallocation or higher allowance.

56 focused controls cover independent logarithmic integrals, curved Jacobians,
nonunit boxes, ordinary and full segmented/partitioned material configurations,
selected-row reuse, physical axes, cancellation, unsupported alternatives,
invalid selectors and the original13 binding. A small selection score cannot
hide a large conditioned error.280 boundary/material regressions,62 fixed-order
controls and31 certificate controls pass. Preserve28 historical red failures.

On the same saved regions the selected axes are[0,1,1]; correction-width ratios
against fixed axis0 are1.0,0.5212096213407336,0.21103063369291866. The first bound
is byte-for-byte unchanged; all corresponding intervals overlap. These compare
correction bounds on unchanged boxes, not subdivided boxes with different base
extensions. Total measured local evaluation time grows from0.8190424seconds to
0.9533947seconds, about16percent. This is a real tradeoff, not a whole-piece speed
claim. All diagnostic work totals6.002415399998426seconds/1074inputs. The full
trial must decide whether tighter bounds repay the extra work under unchanged
time and cell limits. Do not claim full-pixel accuracy from these local results.

Freeze the72-file candidate and obtain explicit approval before another fresh
serial13-piece campaign. All accuracy/resource requirements and remaining M1
moving-gradient, full-frame, scale, readiness and actual original-mesh showcase
gates remain. No consumer implementation or Git publication.
Evidence: opaque94-20260923/campaign-audit1.json,dashboard-audit2.json;
opaque95-20260923/diagnostic1-reference.json,adaptive-diagnostic1-reference.json,
findings1.json and qualification reports.


## Opaque96: direct Bernstein kernels and certified local absolute mass — 2026-09-23

Opaque95 is terminal: three pieces passed, fourth timed out with its last full
bound1.5476186376842977times allowance, nine unstarted. D354 records the user's
standing authorization for bounded qualified checker iterations through the
first accuracy item. Every candidate is frozen before a fresh serial trial;
first-failure stop, all numerical/resource limits and no Git publication remain.

For degree m<=3, (x-t)^(m-1) has Bernstein coefficients
(x-a)^(m-1-k)*(x-b)^k on [a,b]. Sum quadrature terms at degree m-1,
elevate once to degree m, then subtract (1-t)^m/m!. Directed arithmetic
encloses all coefficients without constructing general derivative jets.

The nonnegative Bernstein basis integrates to1/(m+1). Thus integral|p|
is at most length*sum|b_k|/(m+1); one de Casteljau bisection tightens
mixed-sign controls. Sum nonnegative bounds over all overlapping Gauss pieces:
node-uncertainty slivers may overcount but omit neither side of a jump.
Intersect with the existing rectangle and global absolute-mass bounds.
The selected tensor identity still charges all ordinary and boundary terms.
Reuse the same coefficients for the rectangle guard. A128-entry cache stores
immutable mass upper endpoints; fresh returned intervals prevent mutation.
No runtime/wheel change, acceptance relaxation, sampled error estimate or
higher worker/storage quota is introduced.

52direct coefficient/value checks,29mass checks,343retained material/kernel
checks,56adaptive identity checks and31campaign controls qualify this change.
All development failures remain, including an ambient-precision diagnostic
assertion corrected by auditing its saved bytes, a rational-endpoint fixture
error and cache-cost assertions. The latter now count actual polynomial builds
and permit equality when no arguments repeat; numerical conditions are intact.

Three saved same-region widths become77.17%,85.88%,79.08%of the prior widths.
All diagnostic work including preparation totals1950inputs/6.543570100002398s,
within2048inputs/20s. Do not repeat or extend this diagnostic without charging
its existing usage. Direct-only paired observations showed a modest local cost
reduction; mass tightness adds work. Whole-piece benefit is still unproven.
All17previous full campaigns are terminal. Current evidence: opaque96-20260923/
findings1.json,qualification1.json,kernel-cost1-reference.json,cost-audit1.json,
mass-cost1-reference.json and campaign-preview1.json.


## Opaque97: certified-width scheduling for the fourth piece — 2026-09-23

Opaque96 remains terminal:3passed,fourthdeadline,9unstarted; last187cell94leaf
complete bound1.3228110060458683times allowance. All530cells213.6113353999972
reference seconds219.17135879999842worker seconds and114003968peakbytes charged.
All18previous campaigns remain terminal. D354authorizes bounded qualified work.

One optional extra mean-value inclusion safely contracts a previously enclosed
material knot root using its midpoint value and derivative on the larger strip.
It retains transverse uncertainty, multiplicity, presence rules and cancellation.
69independent quadratic-root/material controls pass. A bounded old/new/new/old
comparison on3new saved regions used1600inputs4.598917600000277seconds including
preparation. Width ratios0.9957451132,0.9888239480,0.9844636091do not justify its
extra cost. Preserve this optional qualified implementation with defaultFalse;
it is disabled on all13original pieces, not adopted for the full trial.

The selected fourth-piece candidate instead omits the two optional order10
historical guidance-field calls. Its existing complete certified width now owns
refinement priority, and its complete correction components still select split
direction. All base derivatives,25nodes and correction/error terms remain.
This changes work scheduling only, not any acceptance formula or threshold.
Saved guidance priorities for the three largest leaves exceeded actual widths;
the base width itself was tiny relative to correction uncertainty. The other12
pieces retain historical guidance. No claim that the new schedule will finish
is made before a fresh full trial.14binding/exact-integral/exhaustion controls
and56adaptive identity controls pass; no full trial of root refinement was run.

Freeze the new candidate before one fresh serial trial underD354. Retain256cells,
depth10,90reference seconds,120worker seconds,180watchdog seconds,1GiB,one thread
perpiece, and unchanged13pieceaggregate caps. No old certificates, terminal
restart, source mixing, tolerance relaxation, native change or Gitpublication.
Evidence:opaque96-20260923/campaign-audit1.json,dashboard-audit2.json;
opaque97-20260923/findings1.json,roots-cost1-reference.json,qualification1.json.


## Opaque98: live angular table lookup and timing investigation — 2026-09-23

Opaque97 stopped after one passed piece and a second-piece deadline:135cells,
101.84090020000076reference seconds106.34755420000147worker seconds,110944256peak
bytes;11unstarted. The second piece had an unchanged numerical path, previously
passing151cells50.61624279999887seconds, but this run reached96cells90.4212316s.
Its cause is not established. All19terminal trials remain preserved.

A three-region old/new/new/old diagnostic reproduced exact coefficient/result
hashes and comparable unprofiled times. The one explicitly profiled sample is
excluded from speed comparison. CPU and wall time were close during this sample.
Profiling identified378163 angular-table proxy accesses in three correction
regions. Bind the existing live table once inside each material helper call;
keep every scan, comparison and arithmetic operation. No persistent table cache,
new approximation or stale replacement-table behavior is introduced.18lookup
and endpoint controls,56independent adaptive material controls and14scheduling
controls pass. The final native comparison requires BOTH interval endpoints,
using exact rational arithmetic; no near-endpoint distance-only admission.

All saved diagnostic work totals1866inputs7.062396699999226seconds, below the
2048input20second allowance. New three-region wall times.8432964,.7960069seconds
versus prior unprofiled.8748816,.8387047,.8553257seconds are modest local evidence.
The completed new measurements were followed by an empty-profiler summary error;
preserve that failed report. A pinned saved-byte audit verifies identical results
without rerunning numerical work. Full-piece CPU seconds are now observational
telemetry; all deadlines still use wall time. Fourth-only certified-width priority
and other12guidance remain asD356, root refinement remains disabled.

D354 authorizes a new frozen first-failure-stop serial trial after final checks.
Per piece256cells/depth10/90reference seconds/120worker seconds/180watchdog seconds/
1GiB/one thread and aggregate13piece limits remain. Native151runtime inputs and
74wheel members remain unchanged. No prior certificates are reused. M1 and the
original-mesh showcase remain unfinished; no Git publication.
Evidence:opaque97-20260923/campaign-audit1.json,dashboard-audit2.json;
opaque98-20260923/findings1.json,timing1-reference.json,lookup-cost1-reference.json,
lookup-audit1.json,qualification1.json.


## Opaque99: cached kernel constants and restored guidance - 2026-09-23

Opaque98terminal:3passed/fourthdeadline/9unstarted. All506cells245.83701010000004
reference seconds259.95164420000583worker seconds charged;peak114192384bytes.
The fourth used89.5625CPU seconds90.18823640000119wall seconds and165attempted
cells. Last163cell82leaf matte width2.6314792655972024times allowance. Thus
computation dominated this failure; the simpler scheduling experiment did not
improve whole-piece qualification. All20terminal campaigns remain preserved.

Restore D342historical guidance for all13pieces, superseding the D356fourth-only
omission. Retain D357live table lookup and disabled optional root contraction.
Cache immutable power coefficients of each Gauss Peano polynomial, keyed by all
node/weight endpoints and order, at most18entries (6segments times3orders).
Each requested region still receives its own Bernstein hull, including both
sides of uncertain Gauss knots. Directed Decimal60 operations enclose the exact
polynomial; no sampled estimate or unchanged-endpoint claim for the new algebra.
The cache is within the existing1GiB worker budget; no quota changes.

88exact-rational/uncertain-coefficient/jump/cache controls,343retained numerical
controls,56independent adaptive material controls and14guidance/exhaustion controls
pass. Saved3region old/new/new/old diagnostic1828inputs10.68255180000051seconds;
old3.1589052,2.0661918s,new2.4652113,1.987756s. Timing drift prevents a reliable
whole-piece speed prediction. All old/new interval enclosures overlap; width
ratios round to1.0. Passing CPU time is retained as a JUnit property; failed CPU
time remains in detailed evidence. Wall deadlines own acceptance as before.

D354authorizes one newly frozen serial first-failure-stop trial after source,
record, teaching and dashboard checks. Perpiece256cells/depth10/90reference
seconds/120worker seconds/180watchdog seconds/1GiB/one thread and aggregate13piece
limits remain. No native wheel change, old certificate reuse, terminal restart,
source mixing, relaxed accuracy, Gitpublication or coverage remeasurement.
Evidence:opaque98-20260923/campaign-audit1.json,dashboard-audit2.json;
opaque99-20260923/findings1.json,power-cost1-reference.json,qualification1.json.


## Opaque100: redistribute the same total uncertainty allowance - 2026-09-23

Opaque99terminal:3passed/fourthdeadline/9unstarted;504cells218.99445200000264
reference seconds225.44387039999856worker seconds,113844224peakbytes. Fourth
163attempted cells90.19441870000082reference seconds; last161cell81leaf complete
matte bound1.8754789468298643times its then-assigned share. All21terminal campaigns
remain failed or partial under their original allocations and source identities.

Replace D316area-only distribution for the named13piece profile with positive
exact work weights: fourth5/2,others1. Compute area times work weight, normalize
all13weights exactly, then round every allocation downward. The same aggregate
5e-13radius per specular/matte basis remains; finalRGB and constant-albedo
Jacobian still require1e-12radius. The fourth local allowance increases while
ALL twelve other shares tighten to compensate. This is an internal redistribution,
not an increase to the total numerical allowance. The default area-only helper
remains available. No accuracy admission comes from allocation or historical data.

Proof: for nonnegative exact areas a_i and positive exact weights w_i, set
r_i=floor_outward[(a_i*w_i/sum_j(a_j*w_j))*5e-13]. Thus sum_i r_i<=5e-13.
For RGB=spec+albedo*matte and0<=albedo<=1, total radius is at most
sum_i r_i+albedo*sum_i r_i<=1e-12. Constant-albedo derivative is the matte basis.
The unchanged certificate verifier also checks every piece and the final sum;
rounding, malformed shares or a counterfeit enlarged pool cannot bypass it.
Weights are bounded exact rationals in[1/16,16],never zero or floating estimates;
zero-area pieces remain exactly zero. The profile requires exactly13pieces.

30allocation controls,31campaign controls,1nonzero-width exact-integral composition,
9retained square/area controls and14production guidance controls pass. Other
numerical source remains asopaque99; no new original integration diagnostic was
run. Saved prior bounds are read only to assess feasibility, never reused as fresh
certificates or relabelled. The tighter other-piece shares remain unproven under
runtime limits until the new trial. Runtime/wheel unchanged; no Gitpublication.

D354standing authorization covers the qualified internal redistribution while
preserving FINAL numerical requirements and all resource limits. Freeze a new
candidate and run13freshserialjobs,stopfirstfailure,256cells/depth10/90reference
seconds/120worker seconds/180watchdog seconds/1GiB/one thread perpiece;
aggregate3328cells1170reference seconds1560worker seconds. CPU/CUDA native tolerance
1e-10absolute+1e-9relative and exact bytes remain. All other M1 gates stayopen.
Evidence:opaque99-20260923/campaign-audit1.json,dashboard-audit2.json;
opaque100-20260923/findings1.json,qualification1.json,campaign-preview1.json.


## Opaque101: preserve the conic cancellation before rounding - 2026-09-23

Opaque100 passed eight fresh pieces; ninth index8/face5742 stopped after11cells
at depth10 on bounded angular segment count. No complete original-pixel accuracy
certificate exists. All22 terminal campaigns and every failed report are retained.

The three saved ninth-piece regions were examined before and after factoring,
including both angle roles and both coordinate derivatives. Across the three
bounded diagnostic runs:297 input calls, including preparation, below2048 calls
and20 reference seconds. The unfactored view range crossed66,65,57segments.
Factoring reduced these to43,24,43 and identified radial direction0 for all three.
Thus the old longest-side fallback wasted depth; factoring alone is insufficient.

For the normalized conic q(t)=a*t^2+b*t+c, q(r)=0 and q'(r)=sqrt(discriminant).
Exactly q(r+delta)=delta*(sqrt(discriminant)+a*delta). Retain the original sign
factor and set delta to the signed radial width times distance from the horizon.
Exact rational coefficient equality with the packed normal/view numerator is
checked before use. No sampled root, clipped derivative or changed geometry is
introduced. Order0,1,3,10 jets, both radial sides, linear/quadratic cases, exact
horizon zero, independent90digit values and mismatched-input refusals are tested.
The runtime renderer and installed wheel are unchanged.

Reuse the four existing mean-value input evaluations to report directional
variation. If an angular segment/crossing domain guard refuses, choose the larger
variation of an over-limit angle. This changes only subdivision order. Complete
outward bounds, depth10,256cells and eight-segment guards still own acceptance.
If curved monotonicity alone cannot be proved, an optional complete order-zero
residual range times box area is a conservative fallback. It retains every
material factor and the eight-segment guard, with no invented jump-count proof.
Independent closed material integrals and cap/refusal tests qualify this path.

Initial missing-API failures and the subsequent Fraction-constructor test-fixture
failure remain historical evidence. The corrected controls pass. D354 authorizes
one newly frozen13freshserialpiece trial after source, teaching, dashboard and
record checks. Stop at first failure; do not restart terminal trials or reuse old
certificates. D359 conserved allocation remains, same5e-13basis pool/final1e-12
pixel radius,1e-10absolute+1e-9relative native acceptance and exactCPU/CUDAbytes.
All resource caps and remaining M1 gates are unchanged. No Git publication.


## Opaque102: complete curved-piece bounds remain insufficient - 2026-09-23

Four frozen, ninth-piece-only preflights were run under D354. None is an admitted
campaign certificate; none was restarted or reused. Opaque101 remains the last
full campaign: eight fresh pieces passed, ninth refused, four unstarted. All23
terminal full campaigns and all historical failures remain.

Preflight1: eighth-power map,123cells54.447072499999194reference seconds;depth10
angular-segment refusal. Preflight2: derivative-certified corner ranges,
102cells45.29534899999999reference seconds;same kind of refusal in another region.
Preflight3: complete sparse residual envelope,256cells72.43850869999733reference
seconds; complete255cell128leaf cover but matte width4.408974933378465e-7.
Preflight4: restored fourth-power map,curved certified-width priority and retained
envelope direction hint;233cells90.00202830000126reference seconds, deadline.
Exact final timings, memory and complete-cover widths:opaque102-20260923/findings1.json and
ninth-preflight4-reference.json; the receipts remain authoritative.

D361 eighth-power experiment is superseded by D364: use the previously qualified
fourth-power map. Keep the exact conic factorization. D362 narrows angular ranges
only when signed derivatives on the WHOLE rectangle prove corner extrema. Unknown
signs retain the original conservative range. Two angular roles share at mostfour
extra corner evaluations. No derivative is clipped or inferred from samples.

D363 is a separate order-zero residual proof. Each frozen cubic branch is compared
with a block's first polynomial at a common logarithmic anchor. Outward translated
coefficient hulls retain every branch and discontinuity. A sparse table of4097
blocks for511segments answers any consecutive range with at mosttwo covering
log-polynomial blocks plus the horizon branch, per angle. Multiply the complete
residual range by box area. All preprocessing and queries are charged. The older
Peano eight-segment guard remains unchanged. The envelope is a different proof,
not a claim of smoothness across knots and not a raised resource limit.

D364 uses complete certified widths to prioritize curved regions; historical
high-derivative guidance remains for noncurved pieces. D358 cached kernel constants
remain retained. The new complete bounds are still too wide under existing caps.
No original RGB/albedo-Jacobian admission, whole-frame result or moving-geometry
claim follows. Next qualify ordinary third-derivative and aggregate-jump bounds
for the complete residual, including every knot contribution, before another
frozen targeted preflight. Final5e-13basis/1e-12pixel radii and native1e-10absolute+
1e-9relative/exactCPUCUDAbytes remain, as do256cells/depth10/90reference seconds/
120worker seconds/180watchdog seconds/1GiB/one thread. No Git publication.


## Opaque103: aggregate bounds improve the ninth piece, still unadmitted - 2026-09-24

The frozen95source preflight stopped at90.00333749999845reference seconds and
239cells. Its complete237cell119leaf cover has specular width1.4866593949987313e-13
and matte width1.5065250470787650e-10. This improves the prior matte bound about404
fold but exceeds the unchanged per-piece radius1.3492836682357133e-14. No native
comparison, new13piece campaign or original-pixel admission follows. All23terminal
full campaigns and five isolated preflights remain. The latest full campaign still
has eight passed pieces, a failed ninth and four unstarted pieces.

D365's complete ordinary derivative and aggregate knot-jump proof passed13initial
independent/admission controls and14production-binding controls. Three saved bounds
were110to1992times tighter locally; the full preflight above is the relevant limit.
Later experiments add conditioned tensor order and a two-axis monotone transport
bound for curvature jumps. Fully swept knots use the exact zero K3 moment and a
complete coefficient-variation bound. Partial sweeps retain absolute kernel mass;
all value/slope jumps and their product-rule terms remain. Unknown signs preserve
uniform bounds; cancellation and inconsistent evidence are not ignored.21aggregate
controls,15transport controls and45affected tensor-remainder controls pass. These
conditioned/transport variants are qualified experimental helpers, not selected by
the original full-pixel producer. A saved three-region diagnostic used571inputs and
6.033158000002004seconds; improvements remain insufficient to justify a full trial.

Evidence: opaque103-20260923/preflight-audit1.json, aggregate-diagnostic4-reference.json,
transported-green1.json, transport-green3.json and transverse-regression1.json under
the internal continuation root. Initial fixture/import/collection failures and one
mode-registration syntax failure are retained. All renderer, wheel, numerical,
worker, depth/cell, GPU, dashboard quota and history requirements remain unchanged.
The next repair must tighten both ordinary residual and knot contributions, rather
than relying on local timing or cancellation alone. No new visual acceptance claim.


## Opaque104 — retained higher-order and signed-boundary experiments (2026-09-24)

Opaque104 remains experimental:131high-kernel/identity and44material controls,13grouped and30signed-transport controls pass;249affected controls and34teaching checks pass. Per-term diagnostic192inputs5.265262799999618seconds identifies broad knot-location uncertainty. Signed-transport diagnostic1615inputs13.2884226999995seconds: best residual widths1.054898555266771e-15,5.966503447389516e-13,3.296613307294343e-13; cost remains insufficient for a trial. Order10diagnostic failed20seconds; all red and failed outcomes retained. No producer adoption, preflight, fullcampaign or native comparison.

The helper uses complete branchwise derivatives and every finite-table jump, with signed monotone boundary transport where justified. Unknown transverse monotonicity retains the conservative bound; contradictory enclosures refuse. Original runtime and installed wheel remain unchanged. Next investigate coefficient variation along each boundary using independently checked implicit-curve and weighted-moment identities; do not infer smoothness across the other angular role or omit its residual/jumps. The current D365 producer stays selected. Evidence: `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque104-20260924`.


## Opaque105 — conditional boundary and isolated arithmetic experiments (2026-09-24)

Opaque105:41implicit/moment/root/support controls and31material/curved-coefficient/residual/refinement controls pass. Saved regionB width3.732058640276965e-14 versus old4.122958570274736e-12, but Decimal candidate5.8993694seconds/1866diagnostic inputs. Isolated python-flint0.9.0 passed20initial/21endpoint arithmetic controls and247independent material/kernel controls per profile. Ball profile diagnostic completed but REFUSED crossing monotonicity after35inputs, so its timing is not comparable. Endpoint profile preserved the saved width but took8.8930052candidate seconds/12.3670275total, slower than Decimal. Neither backend nor boundary helper selected by original producer; no new preflight/fullcampaign/native comparison.

The boundary proof uses conditional implicit-curve derivatives and exact signed weighted kernel moments. It bounds the other angular role residual separately, retains every jump order, and preserves conservative evidence on unsupported cases. Four fixed proof supports are derivative bounds, not new integration samples. The compiled dependency is isolated at `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque105-20260924/reference-arb-site1`; wheel SHA256 `1c32c16bb7119ae200a138558cd5d41e61efab8a94b2a967e7168c9e04797ec6`, allZIPmembers and186installed files verified. It is not in the renderer environment and is not selected. API and singleton-root failures, initial name collision (no overwrite/no worker), and the incorrect cubic test expectation remain historical evidence. Evidence root: `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque105-20260924`.


## Opaque106 — positive Gauss8 experiments (2026-09-24)

Opaque106 second frozen preflight terminal at90.00094269999681reference seconds/89.71875CPU seconds,213chargedunits/71regions,23high and48aggregate attempts. Last complete coverwidths1.0482569415847570684164886213749849579345955483525553E-12specular/1.01112818746770901149186707250567123499512269759890352E-9matte; ninthradius1.3492836682357133e-14 unmet. First variant stopped255units/85regions after19.0568062seconds with no high proof. Sevenisolatedpreflights/23fullcampaignsterminal;no original accuracy admission. Gauss8 controls57,material37,adaptive5,selector7,affected60passed. Historical failure from accidentally including a fixture-dependent timing test retained; corrected analytical selection60passed. Manifest collision recovered exact original bytes; D372 records path correction.

Exact rational isolation encloses all eight Legendre roots. Positive weights and exact moments through15 support the Gauss rule; outward tenth-order Peano bounds retain the original derivative cap. See https://dlmf.nist.gov/3.5.v. Each attempted region charges3units for64rather than25samples; the original256limit therefore permits85regions, with failed attempts charged. The selector keeps complete ranges and either an aggregate or order6transport proof. No selected original producer or native runtime change. Evidence root: `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque106-20260924`.


### Opaque107: analytic angular primitives, experimental only (2026-09-24)

Exact-form outward moments of the finite piecewise log cubic include the linear
horizon, discontinuities and reversed limits. Orders1..6 and integration by parts
retain every boundary term and the complete remainder. Independent controls pass
78primitive,27parts,13rectangle/line and19hybrid cases. Tiny nonzero residuals
remain; bounded call-local reuse requires exactly unchanged interval endpoints.

Saved regionB axis0 order6 line width1.0943704059413864e-15 is not an area
certificate. The complete hybrid retains outer quadrature error: width
2.1038171330207015e-12. Reuse preserved exact endpoints, reducing measured
candidate time5.0328607to2.2767238seconds. Boundary-Taylor outer proof narrows
this to2.944389953719754e-13 at5.1139455seconds; still insufficient. Axis1
order6 line width9.500033247523055e-5 is worse: more integration-by-parts terms
are not inherently better for an ill-conditioned coordinate. No producer or
original preflight is selected. All diagnostics charge preparation, obey
20seconds/2048inputs, and preserve prior failed reports.

Artifacts: D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque107-20260924.
D365 remains the original producer;23terminal campaigns and7isolated preflights
remain. Eight of13pieces passed in the last full campaign; complete original
accuracy, moving gradients, full frames and final3Dshowcase remain unadmitted.
Next investigate avoiding the unused half-vector calculation in material-only
proof calls, with exact retained cosine/Jacobian endpoint controls. Renderer,
installed wheel, numerical acceptance and all resource caps remain unchanged.


### Opaque108: unused inputs and tighter ranges, experimental (2026-09-24)

Material-only inputs omit the unused view/half-vector cosine and its two square
roots. Full smooth/specular calls remain complete; cache identity includes the
selection, and the explicit missing value stays None. Thirty input/map/cache
controls and66affected controls pass. Original production defaults remain full
inputs and the fourth-power map. Historical None-cache failure is preserved.

Alternating same-region comparison:1056inputs,7.3650014seconds including setup;
all original interval endpoints are identical. Warm B input time was about
0.193seconds full versus0.164material-only; complete B correction time about
0.555versus0.525seconds. First-use cache costs remain in the report and must not
be counted as an optimization gain. Whole-piece performance is unmeasured.

Individual-segment index27controls pass, including independently derived high
log-polynomial derivatives, exact horizon/same-branch zero, tangent refusal and
complete simultaneous jumps. Saved A/B/C correction widths become
2.7530739405683183e-16/3.9282599479145515e-12/2.5311996459082834e-12, from
1.2494213481229558e-15/4.122958570274736e-12/3.729552194357869e-12. Extra cost
and insufficient improvement prevent selection. Affine-surrogate feasibility
remains too broad (B view radius1.5224407704665387e-10,C2.8405518363682614e-10);
it is not an implemented integrator or a certificate.

The independently qualified eighth-power map combined with newer aggregate
bounds also failed to improve the saved physical regions: A smooth matte radius
4.960792726246739e-11, B correction width4.198625275892978e-12,C3.842770868287111e-12.
No original preflight, campaign or native admission was run. All limits remain.
Artifacts: D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque108-20260924.
Next bounded experiment: accelerate higher spatial derivative convolution with
explicit outward binary64 bounds while retaining lower coefficients in Decimal.
This would be a separately qualified arithmetic path, not unchanged endpoints
or relaxed acceptance; original value, slope and curvature calculations and
all final accuracy/resource criteria remain protected. No automatic selection.
