# Bounded nonlinear opaque shading reference

This development reference evaluates the actual declared Principled radiance
field over exact visible pixel regions. It is not a new production renderer,
a vertex-colour surrogate, or qualification of every material/image configuration.

## Mathematical target

For pixel P, exact affine-depth visibility partitions P into face regions R_f.
The foreground is C(P) = sum_f integral_R_f L_f(x,y) dx dy, and alpha is the
exact union area. L uses perspective weights lambda_i=(beta_i/w_i)/sum_j(beta_j/w_j),
interpolated unnormalised corner normals/view vectors/albedo, normalized shading
directions, and the complete adopted GGX correlated-Smith Schlick specular plus
reciprocal-matte Principled expression. Directional unit irradiance contributes
one positive normal/light cosine. A constant background is composed afterward.

Roughness is fixed per prepared face in this first reference profile. Stored
binary64 resource logs, slopes and mean weights are exact input constants, as in
the existing independent high-precision material-pair oracle. The reference
reconstructs the curve with outward intervals, without native queries or Torch.
It bounds integration of this declared resource model. It does not re-admit the
physical loss table, bound that table's approximation error, or prove the rounding
of production native arithmetic. Those are separate established/open gates.

## Error enclosure, not a difference between two sampled grids

The geometry partition uses the qualified Fraction polygon reference. Each convex
piece is triangulated. The Duffy map p(s,t)=a+s(b-a)+(1-s)t(c-a), for s,t in [0,1],
has Jacobian |det(b-a,c-a)|(1-s). Integration includes that Jacobian; it is also
differentiated when a triangle-coordinate tangent is explicitly supplied.

Tensor Gauss rules with n=3 or n=5 use outward-enclosed nodes and positive weights.
For an interval of length h, the error magnitude is bounded by

    h^(2n+1) * (n!)^4 / ((2n+1)*((2n)!)^2) * sup |TaylorCoefficient_(2n)|.

The Taylor coefficient is the derivative divided by (2n)!, not the raw derivative.
For a rectangle of sides hs,ht, add the s-direction bound multiplied by ht and
the t-direction bound multiplied by hs. Positive Gauss weights make this tensor
extension valid. A mixed jet encloses spatial coefficients through order 2n and
one first parameter tangent. Apply the same remainder construction independently
to radiance and that tangent. Neither passing RGB nor a small finite-difference
residual alone certifies a gradient integral.

All Decimal basic operations round outward at60 digits. The correctly rounded
Decimal sqrt/ln/exp results are widened by one adjacent representable value.
Endpoint negation uses copy_negate, never ambient-context Decimal unary minus.
Formal jet increments have an exactly zero constant coefficient; interval
uncertainty in the base value must not be mistaken for a formal increment.

Each accepted rectangle gets a share of the fixed1e-12 absolute integral budget
proportional to its parameter area. Visible triangle pieces share the pixel
budget. The result is an interval for each RGB value and directional derivative;
the midpoint is usable only with its radius. Unresolved work raises an exception
with diagnostics instead of returning the latest quadrature estimate.

## Smoothness and resource boundary

The interval field must establish a positive perspective denominator, nonzero
normal/view/light/half vectors, positive view/light cosines and one smooth loss
interpolation segment per accepted box. Known Cauchy-Schwarz unit-vector bounds
intersect the value enclosure; they do not clip the mathematical shading model.
Cosine-knot/horizon/two-sided transitions that cannot be partitioned are refused.
This stage does not support spatially varying roughness, arbitrary texture cells,
general visibility topology transitions or all scene/camera/material derivatives.
The local moving-region extension below adds separately qualified stable-visibility tangents.

The first pixel profile retains at most256 examined rectangles,depth10,32 visible
triangle pieces,90s local deadline and the enclosing120s cooperative/180s watchdog,
one thread and1GiB host cap. Geometry retains the existing exact-oracle bounds.
The higher-order rule changes numerical work placement, not these caps.

## Independent controls and provenance

Exact rational polynomial moments qualify both rules. Polynomials vanishing at
every coarse Gauss node must still be detected by the derivative remainder, not
declared zero. Additional controls cover tiny positive area, winding, mixed xy,
moving triangle Jacobians, analytic parameter tangents, transcendental jets,
invalid domains, cancellation and explicit resource exhaustion. The full point
BRDF interval must contain the separate90-digit Decimal formula. A vertex-shaded
affine surrogate must be distinguishable from the nonlinear pixel integral.

The mathematical remainder follows the Gauss quadrature error term in
[NIST DLMF3.5](https://dlmf.nist.gov/3.5). Outward arithmetic and enclosure reasoning
follow [NIST DLMF3.1](https://dlmf.nist.gov/3.1). The transcendental widening relies
on [Python Decimal's correctly rounded elementary functions](https://docs.python.org/3/library/decimal.html).
These sources justify the construction; they do not verify this implementation.
Trust remains in the reviewed algorithm, checked code, Decimal runtime and declared
smooth-domain/input premises. No Lean or external proof certificate is claimed.

Current execution evidence lives under internal opaque54-20260919. Reports retain
every failed oracle, arithmetic, capacity and time-limit attempt. PLAN.md and
NUMBERS.md determine current qualification; this document does not close M1.

## Local moving-region extension (opaque55)

`tools/opaque_moving_reference.py` represents each input vertex trajectory as a
linear polynomial in one parameter. Oriented triangle edges and affine depth-plane
differences are exact polynomial homogeneous lines. Clipping vertices are line
intersections in homogeneous coordinates, retaining polynomial identity rather
than only a value and first derivative. Quotient differentiation gives exact
rational vertex tangents at parameter zero.

Every strict branch polynomial p retains its sign on a common rational interval
[-r,r] by checking sum(k>=1,abs(p_k)*r^k) < abs(p_0). All denominator and polygon
orientation predicates enter that certificate. A predicate zero at the origin
but not identically zero is refused, including a quadratic transition with zero
first derivative. Persistent exact incidences and persistent depth ties retain
original face ordering. This is a sufficient local condition, not permission to
cross a topology transition or claim differentiability everywhere.

The transported triangle and its Jacobian enter the already qualified mixed-jet
Gauss reference together. For a material attached to a moving triangle, its screen
velocity interpolated by nominal barycentric weights is subtracted from the query
coordinate tangent before evaluating the original perspective field. Attribute
and inverse-w tangents remain separate explicit inputs. A screen-fixed field and
an attached field can have equal current colours but different correct gradients.

Controls include constant and affine fields, both diagonals and subdivision,
hidden/background contributions, subpixel geometry, simultaneous xyz motion,
independent rational finite differences inside the certificate interval, and a
translation negative control that deliberately leaves the material screen-fixed.
Full nonlinear y-independent Principled fields additionally have independent
boundary derivatives: -L(t) at a silhouette, and L_left(t)-L_right(t) at a depth
intersection. Their 90-digit evaluations must lie inside the reference intervals.
Installed-wheel Gauss16 uses a different one-dimensional rule and autodiff through
moving nodes and weights; it is compared with the existing 1e-10 absolute plus
1e-9 relative material tolerance. Native rounding is not part of the reference
integration enclosure, and agreement never supplies the enclosure.

Geometry retains 64 vertices,32 faces,128 polygon vertices,4096 intermediate
pieces,200000 counted geometry operations,8192-bit rational intermediates and90s
local guards. The symbolic representation adds a degree32 ceiling and at most64
radius halvings. Integration still permits only32 final triangles,256 examined
cells,depth10 and90s total,inside the unchanged serial worker caps. No quota or
numerical tolerance was raised. This adds finite local reference evidence, not a
production nonlinear opaque scene path or CUDA qualification.

## Material partition and varying-roughness extension (opaque56)

`tools/opaque_material_reference.py` adds exact homogeneous material cut lines,
local polynomial branch certificates, normalized bilinear clamp/repeat cells,
perspective boundary transport, exponential mixed jets, and strict interval
selection of resource interpolation branches. Neither an uncertain limiter nor
a horizon crossing is assigned a sampled midpoint branch.

For the tested roughness interval[0.826,0.828], centred mean-value enclosures bound
each roughness derivative of all512 angular resource controls. A sixth-degree
Taylor polynomial retains its seventh-derivative remainder; Gauss3 then has the
spatial order6 and mixed tangent order7 needed for separate value/gradient error
bounds. Negative controls reject a missing seventh-order contribution and a thin
texture feature missed by coarse samples. The accepted full-resource pixel uses
4 examined cells,3 accepted cells and2 original triangles in64.74246079998557s.
Maximum RGB radius3.69153540102703515e-15 and first-gradient radius
2.97361431092303449e-13 remain below the existing1e-12 budget.

A separate90-digit material formula checks roughness point derivatives and the
fundamental-theorem gradient of the spatial-roughness integral. Piecewise material
changes,UV motion and texel derivatives are checked with exact bilinear moments
and separate90-digit Principled values; installed Gauss16 or rectangular Gauss2
calls are comparison evidence,never the source of the integration certificate.
All initial branch refusals and implementation/harness failures are retained.

Native scene integration is described in OPAQUE_QUADRATURE.md. Its finite tested
profile is now connected, but no universal bound follows from fixed Gauss nodes.
Further material branches and scalable streaming must be independently qualified.


## Explicit material branches (opaque57)

`opaque_material_branches.radiance` certifies a fixed hemisphere from interval
dot products, normalizes only nonzero vector domains, applies an explicitly
requested two-sided flip, and returns exact zero for a strictly unlit branch.
Intervals crossing a horizon refuse. Angular-knot crossings retain the existing
reference refusal; no midpoint branch selection is introduced. Six moving-interface
pixel fixtures span roughness0.15..0.95,F0 endpoints and two-sided/unlit branches.
Their constant-region pixel and boundary-flux integrals are independently enclosed
and compared with90-digit formulas. This broadens finite branch evidence; it does
not certify general spatial roughness or horizon-crossing scenes.


## Angular-knot residual enclosures (opaque58)

The original scale mesh has constant roughness1 and smooth normals whose angular
cosines cross resource knots. The smooth-box reference preserves a256-cell
refusal for its256-square pixel at(160,128). Whole-face corner observations
are a preflight diagnostic, not a visible-pixel certificate.

`opaque_loss_extension.py` chooses one fixed cubic angular segment for each
view/light role in each integration cell and extends it smoothly for that cell.
The choice is frozen across all nodes, derivative remainders and residual bounds
in that cell; choosing a midpoint branch never discards the residual. In log-cosine
coordinates the difference between any actual segment and that extension is
another cubic. Outward interval power coefficients enclose this difference over
every intersected segment (at most8), including endpoint rounding discrepancies;
no exact C1 join or sampled-grid agreement is assumed.

Write actual view/light losses as A+dA and B+dB. The diffuse residual is formed
as dA*B + A*dB + dA*dB, avoiding cancellation of two independently bounded whole
BRDFs. Specular radiance is unchanged. The quadrature bounds the smooth field by
its existing derivative remainder and bounds the residual directly over each
box. The nonnegative Duffy area weight is integrated exactly. The residual's
signed value and first tangent enclosure are added to the corresponding integral
bounds, then the same acceptance test is applied. High derivatives are never
taken across the actual knot. A residual that remains too wide still refuses.

This extension supports image values and angle-independent material tangents,
such as albedo. Any nonzero angular tangent refuses: geometry, normal, camera or
light-direction derivatives need their own knot-transport treatment. Horizons,
roughness transitions, arbitrary topology and full-frame admission do not follow.
The1e-12 value/gradient target,256 cells,depth10,32 pieces and90s reference budget
are unchanged. Controls check signed residuals, an analytic nonsmooth integral,
separate90-digit cubic formulas, cancellation, unsupported tangents and the full
Principled value/albedo derivative. Current measured admission remains in records.

The residual cosine range additionally intersects its natural interval with a
two-dimensional mean-value enclosure: f(m)+Dx f(box)*(x-mx)+Dy f(box)*(y-my).
Only the smooth geometric cosine is differentiated. The piecewise angular loss
still uses its complete residual enclosure. Failed old natural-range attempts
and the overly tight nonlinear test expectation remain preserved.

The original pixel(160,128) at256-square is now admitted in the saved opaque58
reference256right8 report:128 cells,65 leaves,two visible triangles,max depth6.
Image radius <=7.290e-14 and first-albedo-tangent radius <=1.215e-13. Installed
CPU/CUDA RGB and that tangent lie inside the enclosures, and RGB/mask/full albedo
Jacobian bytes agree. Reference time is63.454s (66.588s whole supervised worker);
use the exact report for timing rather than this rounded narrative. No whole
frame, moving-angle gradient or dense-cost admission follows.


## Moving angular knots and finite horizons

The base `LossExtension` still refuses angular tangents. The development-only
`MovingLossExtension` adds the ordinary actual-minus-extension first derivative.
That is not the entire derivative of a moving piecewise integral. Separately,
`transversal_flux` bounds the jump term on any subset of a rectangle by
sum(sup|jump| * sup|q_theta| / inf|q_axis| * transverse_width).
The selected spatial partial must have fixed sign throughout the rectangle;
otherwise the crossing refuses. Actual right/left polynomials are evaluated at
binary cosine knots; exact C1 continuity is never assumed from the stored logs.
The flux uncertainty is added before the unchanged1e-12 acceptance test.

A roughness1 real-resource pixel crosses one angular knot under view motion.
Independent integral bounds agree with a90-digit fundamental-theorem endpoint
formula. Rigid projected geometry motion transports the attached view field with
the opposite derivative; installed CPU and CUDA values and derivatives lie in
these enclosures. Their geometry tangents differ by1.423e-22 at most,so this is
not an exact-byte cross-device derivative claim. Fixed visibility,constant F0,
and this finite profile do not qualify arbitrary moving geometry or materials.

`opaque_horizon_reference` treats two exact moving linear horizon profiles at
roughness1,F0=1,albedo0. A changing view has a nonzero radiance jump at n dot v=0;
a changing light is continuous at n dot l=0 after the illumination cosine.
Exact moving partitions and separate90-digit antiderivatives bound each image
and first derivative. Installed native quadrature with an explicit horizon cut
passes the existing1e-10abs+1e-9rel comparison on CPU and CUDA. Without that cut,
the view example has derivative+0.042997 instead of the correct-0.109973.
The native candidate is compared to the independent interval with the stated
native tolerance; it is not claimed to lie inside the much tighter interval.

Production scene cuts currently cover roughness and texture,not general horizons.
A varying normal dotted with a varying perspective view can have a quadratic
numerator; a linear cut cannot be substituted without proof. These finite
qualifications therefore do not admit general production horizon gradients.


Opaque60 connects the finite horizon treatment to production scene composition.
Constant-domain view horizons and directional-light horizons use homogeneous
linear cuts. Varying-normal quadratic view numerators use conservative sign
exclusion and otherwise refuse. The new actual scene reference uses a thin
plane and a moving constant normal; independent interval integration covers both
radiance and normal-motion derivative. It preserves the existing native
1e-10abs+1e-9rel and1e-12 area criteria. General curved horizon integration remains
open. The prior unpartitioned wrong-sign control and all failed reports remain.

The moving-knot exactness diagnostic distinguishes the caller reduction tree
from the native primitive: explicit channel/sample/broadcast sums retain all
contributions and restore CPU/CUDA byte equality for that fixture while retaining
the independent image/derivative comparisons. Exactness and approximation error
remain separate obligations.


## Curved view-horizon reference (opaque61)

Varying perspective normals and view vectors have a quadratic homogeneous dot
product. `opaque_curved_reference` integrates a certified conic region of an
exact visible triangle. It does not replace the curve with a straight chord.

Let A be the isolated-sign vertex and E(s) the opposite edge. In radial
coordinates p=A+t(E(s)-A), the horizon is a(s)t^2+b(s)t+c=0. Exact rational
Bernstein controls certify opposite signs at t=0 and t=1 and a strictly positive
radial derivative throughout the square. Nonidentity zeros, uncertified
transversality and unsupported sign patterns refuse. A common rational parameter
neighbourhood certifies these geometric predicates. The rationalized root
r=-2c/(b+sqrt(b^2-4ac)) also covers the linear limit without division by a.

For the outside region set t=r+(1-r)u; the area Jacobian is
abs(det(B-A,C-A))*t*(1-r). The inside region instead uses t=r*u and Jacobian
abs(det)*t*r. Differentiating this mapping and Jacobian retains both ordinary
field derivatives and the moving-boundary jump. A separate test scales the
triangle itself, so omitting its Jacobian derivative also fails. A constant-field
parabolic example has zero pointwise derivative but nonzero integral derivative;
its independent 90-digit antiderivative checks the result.

Exact polynomial reduction precedes interval evaluation. Homogeneous normal and
view fields are composed directly into radial coordinates, where their common
positive perspective denominator cancels on normalization. Reconstructing thin
triangle barycentrics from independent x/y intervals exhausted the unchanged
90-second deadline; its failure remains. The reduced form qualifies the selected
original patch in 8 cells, max depth 2, about 2.10 seconds. This is an expression
improvement, not a tolerance or worker-limit change. Reference caps remain 256
cells, depth 10, 90 seconds and radius 1e-12. Two parameter triangles represent
one curved region; they are not straight geometric approximations. A future
whole-pixel composition must account for all regions under the existing limits.

The original 128-square scene has 174 retained faces with strict corner view-sign
crossings. Pixel (101,78), face 5743, has eight candidate faces and thirteen exact
visible pieces; one selected visible triangle crosses the conic. Four earlier
candidate pixels refused the 32-piece reference cap. The selected triangle's
roughness=F0=1, albedo=0 GGX control gives radiance contribution approximately
0.00123214862176932 and normal-z-motion derivative 0.12240345303237729, each with
independent radius below 1e-12. This is the real geometry with an explicitly
different control material. It is not the full original F0=.04/RGB-albedo shader,
an entire pixel, a production quadrature result, or a frame-throughput result.

Production continues to refuse unresolved curved horizons. Connecting curved
quadrature requires the shared native CPU/CUDA and adapter surfaces, independent
image/gradient comparison, complete material and angular-knot treatment, tile and
resource qualification. No runtime input changed in this reference batch, so the
existing opaque60 installed wheel remains authoritative for production evidence.
