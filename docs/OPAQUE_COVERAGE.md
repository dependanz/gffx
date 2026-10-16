# Geometry-aware opaque coverage: qualification contract

User-approved direction, 2026-09-19 (devbrain D303). Implementation and M1 acceptance
remain open. Existing hard/soft reports retain their original model and artifact.

The target forward image integrates the nearest opaque surface's linear radiance
over a unit square pixel footprint, with background where no surface is present.
Integrate foreground color and alpha together before background composition.
Interior tessellation edges do not introduce transparency. Geometry channels such
as IDs and depth remain center samples; they are not averages of unrelated surfaces.

The first independently checkable domain uses constant RGB per triangle, affine
screen-space depth, triangles wholly inside the supported projection depth range,
and a box filter. This isolates visibility; it does not qualify arbitrary varying
shading, perspective clipping, near-plane intersections, transparency or a shader.
Both windings are included without culling in the oracle; equal depth selects the
earlier face. Pixel coordinates use x right/y down, integer pixel corners and
half-integer centers. Smaller depth is nearer. All coordinates are rationalized
exactly from the supplied values, without renderer raster outputs or torch.

An independent development-only Fraction oracle will clip triangles against pixel
rectangles, subtract regions hidden by other triangles using affine depth ordering,
and integrate disjoint visible polygon areas. This is deliberately separate from
native point rasterization and the candidate neighbor-based boundary methods.
The production geometry implementation remains C11, with the existing CUDA boundary;
the oracle is not a Python replacement runtime or an additional runtime dependency.

Oracle controls use exact rational equality: either square diagonal, subdivisions
and interior-vertex motion, boundary clipping, translated silhouettes, full and
partial occlusion, intersecting depth planes, equal-depth ties, degenerate faces,
subpixel geometry, tiled pixel windows, rejection of invalid/unbounded input and
cancellation. Analytically linear motion cases require exact unit area/color
derivatives at two rational difference steps. A one-pixel perturbation must fail.
Limit this tool to 8x8 pixels, 64 vertices, 32 faces, bounded rational complexity and
explicit operation/piece budgets. These are finite oracle bounds, not new production
limits or a performance claim.

Then compare an independently implemented nvdiffrast-style boundary treatment and
EdgeGrad's micro-edge estimator. Record forward image and gradient errors separately.
The unchanged point image used by a surrogate gradient is not the exact filtered
image, and its nonzero visibility gradient is not an ordinary derivative of hard
point sampling. No tolerance is chosen retrospectively to make a candidate pass.
Do not claim either upstream renderer was executed when evaluating an in-tree model.

The intended new explicit appearance selector is `coverage="opaque"`; its initial
absence is a red fixture. A final default/compatibility transition follows candidate
qualification. Keep the old broad `soft` coverage as a separately documented fitting
aid rather than silently reinterpreting its historical tests or images.

Before production adoption, declare error bounds for the selected filtered model
and varying-shading treatment, extend public surfaces and workload accounting, and
qualify CPU/CUDA values and derivatives in fresh installed wheels. Neighbor methods
must qualify halo width, viewport borders, tile ownership, reverse accumulation,
subpixel misses and intersections. Preserve resource caps, admission formulas,
worker serialization, every failed report and dashboard quotas. No large-scale
render is admitted by this small-scene oracle alone.

Primary mathematical references: [nvdiffrast](https://nvlabs.github.io/nvdiffrast/)
and [Rasterized Edge Gradients](https://arxiv.org/html/2405.02508v2). This contract's
exact rational oracle and tests are original project development work.


## Bounded candidate study (opaque51)

`tools/opaque_boundary_models.c` is original C11 experimental code, compiled as an
explicitly selected development DLL. No upstream library was installed or run.
It does not change the gffx wheel, CPU API or CUDA provider.

Three models isolate distinct choices on constant-colour affine-depth triangles:

* Complete geometric coverage subtracts occluded polygon regions and uses scalar
  forward-mode differentiation through those operations. It searches all faces in
  this small profile. This is a reference candidate, not a scale-qualified renderer.
* Sampled-boundary coverage first discovers surfaces at pixel centers in a one-pixel
  halo, then uses their geometry for analytical pixel filtering. This is inspired
  by the rasterize-then-antialias organization of nvdiffrast. Its explicit depth-plane
  intersection treatment is our extension; it is not a reproduction of nvdiffrast.
* The micro-edge candidate keeps point-sampled colour and applies EdgeGrad's averaged
  pixel-pair boundary derivative, with adjacency/overhang/intersection classification
  and a scalar boundary-motion derivative. It tests that principle, not DRTK's full
  fragment-scattering implementation or optimizer behavior.

The analytical fixture tolerance is fixed before candidate execution at 1e-12
absolute for float64 image channels and directional derivatives; the rational
oracle controls retain exact equality. Derivatives are checked against exact
symmetric rational differences at two steps on the declared polynomial paths.
No general convergence or arbitrary-geometry derivative proof follows from that.

The first candidate run failed subdivision-gradient cancellation. Removing a
coincident polygon vertex using only its value discarded a distinct derivative.
Keep coincident points unless both value and derivative match. The corrected
candidate passes the original unchanged controls; the failed DLL/report remain.

Both sample-discovered candidates miss the subpixel fixture. The micro-edge model
also has intentionally different forward pixels and distributed per-pixel
derivatives at moving boundaries. Good total-loss gradients do not establish
correct per-pixel derivatives of the chosen filtered image. These limitations are
recorded, not accepted as production opaque-image accuracy.

Pixel-window controls use complete global geometry and freshly discover the halo
at each window. They compare exact image and directional-derivative bytes. They
do not qualify streamed reverse VJPs, production tile memory or global reverse
accumulation. Those remain required before adoption.


## Native alpha integration (opaque52; qualification in progress)

`render.opaque_coverage(pixel_vertices, faces, height, width, origin=(0,0),
workspace_limit_bytes=67108864)` computes opaque alpha as projected triangle union
area. For alpha alone, depth ordering cannot change this union; this does not solve
filtered colour, where depth ordering and radiance integration remain essential.
The first native C11 implementation is a CPU reference with caller-owned scratch,
float32/float64 tensors and an ordered first VJP (transpose of scalar duals).
It is exposed by the C ABI and lazy Python/PyTorch rendering surfaces. CUDA requests
fail explicitly rather than transferring data. First derivatives follow the selected
clipping topology at nondifferentiable branch events; second derivatives are refused.

Conservative closed triangle bounds discover geometry per pixel, including subpixel
triangles. At most32 AABB candidates,64 polygon vertices,128 visible pieces and
2000000 clip-edge visits per individual union evaluation are allowed. Overflow
raises; partial outputs are invalid. Geometry is never truncated to fit these bounds.
Projected coordinates and image axes are limited to2**20. No scale or throughput
claim follows; the reference scans faces and recomputes derivatives deliberately.
Native workspace is charged to the caller; it is not the old32KiB stream carry.

The initial scene composition accepts only explicit `coverage="opaque"` with
`outputs=("mask",)` on CPU, unit supersampling and triangles within the projection
depth interval. It applies camera projection and culling before union integration.
Partially clipped triangles and opaque shaded images fail explicitly until qualified.
The default renderer and legacy hard/soft results retain their old semantics.
No statement that the visible shaded mesh is repaired follows from this alpha step.

Qualification: independent finite mask/JVP fixtures retain1e-12 absolute float64
and1e-6 float32 errors, with nonuniform cotangents; full-coordinate double gradcheck
uses the existing1e-6/1e-4 absolute/relative gradient limits. Global windows must
preserve values and ordered reverse sums on the declared binary-exact fixtures.
Invalid inputs, capacity/resource refusals and native C buffer contracts are separate
gates. Fresh installed-wheel identity is required; no source-tree overlay is used.


Large-origin control: an exactly representable triangle of area1/32768 must retain
the same positive area and geometry VJP at pixel origins1024,65536 and1048575.
The first global-coordinate shoelace implementation fails this case by returning
zero. The correction translates geometry to each pixel's local coordinate system
before constructing edges and integrating area. Candidate bounds remain global;
the translation is constant with respect to geometry, so it changes no derivative
target. The failed installed-wheel report is retained, not relabelled a pass.


## Depth-ordered affine radiance (opaque53; bounded CPU qualification)

`opaque_integrate(pixel_vertices, faces, corner_radiance, height, width, origin=...)`
accepts pixel vertices[V,3] and RGB radiance[F,3,3] at each face's three corners.
Depth and radiance are affine in screen space, not perspective-correct attributes.
The result[H,W,4] is the integral of the nearest field's linear RGB plus alpha.
Compose a constant background as RGB+(1-alpha)*background after integration.
There is no blur of already selected face colours and no soft interior-edge fade.

Per pixel, conservatively discover triangle bounds and subtract regions in which
another triangle's affine depth is smaller. Identical depth planes use original
face order. Integrate each remaining polygon's area and first x/y moments, then
contract its barycentric integrals with corner RGB. This is exact for the declared
degree-one radiance field apart from floating arithmetic. Geometry/depth first VJPs
differentiate those same visible pieces; radiance VJPs use their integrated weights.
Neither a depth tie nor a topology change is asserted differentiable. The selected
branch is explicit and a finite derivative there is not a smoothness guarantee.

The independent Fraction reference integrates the same mathematical field through
triangle-fan centroid quadrature, which is exact for degree one. Closed-form basis,
coordinate-field, background and constant-reduction controls precede native checks.
Image and analytical directional checks retain1e-12 float64/1e-6 float32 absolute
limits; all-coordinate double gradcheck retains1e-6 absolute/1e-4 relative limits.
Hidden owners, equal-depth tie order, geometry/depth/corner-radiance partials and
ordered pixel-window accumulation require separate tests.

The C ABI owns preallocated outputs and caller scratch. This initial CPU reference
retains32 conservative candidates,64 polygon vertices,128 pieces,2M clip-edge
visits per evaluation and the existing64MiB admitted window. Capacity failures
invalidate partial outputs rather than dropping geometry. CUDA remains explicit
unsupported; no host transfer or scale qualification is implied.

This field integration is a building block for shaded rendering. Vertex-evaluating
a nonlinear Principled shader and interpolating its answers is not generally the
integral of that shader. Perspective attributes, nonlinear radiance quadrature and
its error bounds must be independently qualified before scene appearance adoption.
The existing scene colour route and legacy hard/soft results remain unchanged.

Opaque53 checkpoint: installed wheel38e9d176 passes25 new colour checks plus41
existing alpha/scene/regression checks; the independent oracle passes24 checks.
All72 installed package members match the wheel. The direct C boundary, exported
symbols, C++ headers and structural ownership/I/O gates pass. The first C fixture
combined malformed shape with noncontiguous strides; its failed report remains
preserved, and a contiguous malformed-shape control now reaches the intended rule.
The34 teaching checks pass after rebuilding20 static textbook pages; the stale-page
failure also remains historical. Saved96x96 affine panels differ by at most
2.220446049250313e-16 across diagonals. A closed-form moving depth intersection has
zero image error and2.220446049250313e-16 maximum derivative error. These finite
checks do not establish a global floating-point error bound or qualify full shaded
meshes. The scene opaque route remains mask-only pending nonlinear integration.
