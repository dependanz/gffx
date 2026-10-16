"""Original deferred-rendering chapter, separate from numerical evidence."""

CHAPTERS=({
    'id':'learn/deferred-rendering',
    'title':'Deferred rendering: from scene geometry to an image loss',
    'prerequisites':['learn/scenes','learn/cameras','learn/barycentric','learn/shading/energy-conservation'],
    'evidence':['render/deferred'],
    'source':'src/gffx/torch/_deferred.py',
    'body':r'''
<h2>Separate finding a surface from lighting it</h2>
<p>Rasterization answers which triangle occupies a pixel and where the pixel lies
inside it. A geometry buffer, or G-buffer, stores that answer together with the
world position, normal, texture coordinates and material identity. Shading then
evaluates the selected surface. The word deferred means lighting happens after
surface selection; it does not mean the calculation is postponed to another time.</p>
<p>A scene gives object relationships, a camera projects them, and the renderer
combines their operations. Depth and a pretty colour image tell different stories:
depth measures geometry along the view, while colour additionally depends on
material and illumination. Keeping these channels inspectable helps diagnose an
incorrect normal without blaming the camera or optimizer.</p>
<h2>Why screen weights are not surface weights</h2>
<p>A perspective camera divides positions by their homogeneous clip coordinate w.
Interpolation in screen space therefore needs to undo that division. Let beta be
the affine pixel-space barycentric weights and let w_i be the positive clip w at
corner i. Corrected surface weights lambda_i divide each beta_i by its corner w_i
and normalize. World position, UVs and material attributes use these corrected
weights. The rasterizer's interpolated NDC depth keeps its separate definition.</p>
@@\lambda_i = \frac{\beta_i / w_i}{\sum_j \beta_j / w_j}@@Perspective-correct weights; beta and lambda are dimensionless, while w follows the camera projection convention.@@
<p>Worked example: beta=(1/3,1/3,1/3) and w=(1,2,4) give unnormalized weights
(1/3,1/6,1/12), whose sum is7/12. The corrected weights are(4/7,2/7,1/7).
An attribute equal to(0,0,1) at the three corners is1/7 here, not the affine1/3.</p>
<details><summary>Try it yourself: which corner dominates and why?</summary>
<p>Answer: The first corner has the smallest positive w and gets weight4/7. A screen-space
mixture does not correspond to an equal surface-space mixture when depths differ.
With all w equal, the original beta weights are recovered.</p></details>
<h2>Normal directions require care</h2>
<p>Interpolated normals must be renormalized before material evaluation. A cube
usually needs independent face normals; averaging a corner normal across its
three perpendicular faces makes the cube look inflated or dented. That can be
an attribute-authoring problem even when the shading formula is correct. Missing
authored normals use face normals in this renderer. A zero active normal is an
error rather than permission to invent a direction.</p>
<h2>Coverage is not material opacity</h2>
<p>Hard coverage is a discrete yes/no selection. Its local derivative with respect
to geometry is zero. A soft signed-distance ramp supplies a useful local gradient,
but is a declared approximation rather than physical transparency. At a candidate
cutoff, a face swap or the nearest-edge boundary, smoothness is not guaranteed.
The front-fragment approximation also does not promise a seamless union across
every shared mesh edge. These limits must be visible beside an optimization demo.</p>
@@\alpha = \frac{1}{1+\exp(d/r^2)}@@For a retained fragment, d is signed squared pixel distance and r is a positive pixel radius; absent fragments have zero coverage.@@
<p>For soft colour outside a triangle, clamping negative affine barycentrics and
renormalizing provides a defined boundary extension before perspective correction.
It prevents a legal black-to-white albedo from becoming negative through
extrapolation. The clamp is piecewise differentiable; it does not remove the
coverage approximation's discontinuities.</p>
<h2>Filter premultiplied colour, not invisible paint</h2>
<p>Store foreground RGB multiplied by coverage. Supersampling averages this RGB
and coverage together. Otherwise invisible saturated colours can bleed into edges.
For four samples consisting of one opaque red sample and three transparent ones,
the mean is premultiplied RGB=(1/4,0,0), alpha=1/4. Its straight colour is red,
not dark red. Over white, the composited RGB is(1,3/4,3/4).</p>
@@C = C_p + (1-\alpha)C_b@@Premultiplied foreground C_p plus the uncovered fraction of background C_b; all RGB values are linear.@@
<p>Do not average triangle IDs, depths from unrelated surfaces, or UVs across a
seam. Geometric output channels are sampled at final pixel centres even when
colour and coverage are supersampled. Reconstructing colour is an image operation,
not permission to reinterpret every G-buffer field as an ordinary colour channel.</p>
<h2>A practical visibility test is an approximation too</h2>
<p>Project a vertex, select its containing pixel, and compare its NDC depth with
the raster depth using an explicit tolerance in NDC units. Behind-camera,
offscreen and background probes are false. On a sloping triangle, a vertex depth
can differ from the nearby pixel-centre depth even when the vertex belongs to
the visible surface. This sampled test is useful but is not exact continuous
visibility. Increasing tolerance changes the test and must not be hidden.</p>
<h2>Differentiate the complete chain, within its limits</h2>
<p>An image loss should reach mesh positions, camera intrinsics and poses,
albedo maps and lighting. A material derivative alone does not prove that the
full render preserves these connections. Compare analytic derivatives with two
finite-difference steps away from discrete boundaries, then take a fresh small
parameter step and verify the loss improves. Never detach a fitted texture or
cache its parameter-dependent pyramid across optimizer steps.</p>
<p>Splitting an expensive shader into small calls does not bound memory if every
call's backward graph remains alive. Full-image differentiation needs a measured
execution strategy, such as verified recomputation, as well as finite worker and
allocation limits. A preflight refusal is more honest than silently omitting
gradients or claiming a large-image profile from a single shaded point.</p>
<h2>A bounded material-recovery experiment</h2>
<p>Hold a hard-covered head, camera, light and Classic highlight fixed, and fit
only its constant RGB albedo. For each covered pixel, the independent scalar
model is C=B+Sa: B is the fixed highlight, S is the diffuse coefficient and a
is the fitted channel. A mean squared image loss therefore has derivative
2 times the mean of S(C-target), independently for each channel. This gives a
separate analytic explanation of the gradient followed by the public renderer.</p>
<p>The saved64-square example takes20 fixed-rate fresh-graph steps from
(0.72,0.12,0.25) toward target(0.2,0.4,0.6). Its initial mean squared error is
0.007840132184137153 and its final error is1.10225039165403e-11. The maximum
parameter error is1.9497628087811192e-5. These are the saved Classic17 producer's
measurements, not results of running this article. The gallery labels the
initial/final images and uses the same20-times error display scale throughout.</p>
<p>This checks albedo recovery within fixed coverage. It does not fit every
camera, material or geometric parameter, qualify large frames, or establish a
consumer application's readiness. An independent material formula and a separate
Decimal coverage oracle answer different questions; neither replaces the other.</p>
<h2>Same equations, matching operations</h2>
<p>Floating-point evaluation is an ordered calculation, not exact algebra. Two
libraries can round the same elementary function differently, and a scalar divisor
can take a different implementation path from a tensor divisor. When a CPU/GPU
image differs, capture the same intermediate inputs before deciding which stage
is responsible. Increasing an image tolerance does not repair a reference-mode
byte mismatch.</p>
<p>The procedural scale-head check illustrates this distinction. Hard images matched,
but soft masks did not. Face indices, screen weights, depths and edge distances
matched exactly. A common-input comparison isolated the logistic coverage function;
a second capture found a difference in the preceding division by squared blur radius.
A shared sequence of exponential, division and multiplication operations repaired
the exact mask comparison without changing that radius or the visibility model.</p>
<p>Independent high-precision values answer whether the result is accurate. Exact
CPU/GPU values and first gradients answer whether device choice changes the
calculation. Both checks matter. Likewise, a correctly rounded native square root
can differ by one ULP from an older library reference; preserve that discrepancy
and consult an independent oracle instead of declaring either library correct by
name. Full-image gradient and resource tests remain necessary after a local repair.</p>
<h2>Larger dispatches, unchanged pixel ownership</h2>
<p>Pixels may share a material or light even when their shading calculations are
independent. A larger internal dispatch can process several ordered four-point
groups together without changing their equations. It keeps the same point/light
pair order and fits inside the existing workspace reserve. This is different
from skipping pixels, lowering precision or changing the render resolution.</p>
<p>The reverse pass must still combine the contributions to shared parameters.
Floating-point addition depends on its order: (a+b)+c can differ from a+(b+c).
The grouped renderer therefore recomputes bounded cohorts and explicitly adds
their four-point contributions in reverse group order. Within a group, shared
roughness and RGB fields use the same fixed adjacent-pair reduction on both
devices. Independent finite differences check the derivative itself; exact
CPU/GPU comparisons check the chosen calculation order.</p>
<p>The current boundary allows at most32 groups,128 shading points and1024
point/light pairs per internal call. It saves original inputs, releases temporary
backward graphs and retains the public material's smaller admission limits.
A passing small-frame batch test does not qualify a larger image's geometry,
output storage or execution time. See the
<a href="/testing/?page=render/grouped-principled">grouped-rendering tests</a>
for those distinct checks and the preserved reduction-order failures.</p>
<p>When roughness is one shared scalar,all pixels use the same material curve.
The renderer prepares that curve once during forward,releases it,and prepares a
fresh curve during backward. Updating roughness before a later render therefore
cannot reuse stale values. A roughness map still owns independent point values
and gradients. Complete input values are checked once per evaluation;each small
group retains its shape,admission and native validation checks.</p>
<p>Memory admission uses a documented inventory of simultaneous native storage.
If the largest shading cohort does not fit,the renderer selects a smaller cohort
without changing its four-point gradient order. This helps scratch storage;it
cannot make a full image fit when the requested output arrays alone occupy the
allowance. A streamed output interface has different ownership and must be
qualified separately before it can satisfy a large-frame gate.</p>
<h2>Rejecting work while preserving pixels</h2>
<p>The CPU rasterizer visits small pixel blocks and reuses each triangle's projected
coordinates inside a block. It skips a block only when conservative bounds prove
that the original rounded calculation cannot include any of its pixels. A
geometric bounding box alone is insufficient: extreme or uncertain arithmetic
keeps the original evaluation. Surviving pixels use the same equations and
ascending face order,including depth ties and soft coverage.</p>
<p>Different pixels own separate fragment lists,but fitted material parameters
may be shared across the image. Their gradients still require a fixed sum.
The bounded streamed profile keeps each original four-point material partial
and adds those partials in the original reverse order. Preparing lighting once
for an admitted cohort removes repeated setup without merging those partials.
Exact original-output checks,independent differences and full-job resource
measurements test different parts of that claim.</p>
<p>These changes improve measured execution;they do not guarantee that every
resolution passes the time forecast. The dashboard retains refused profiles
and all earlier failures. Current scale plates show actual saved pixels and
single complete-job timings,not sustained throughput or final visual acceptance.</p>
<h2>Why the largest window can be slower</h2>
<p>A row window and a shader group spend the same finite memory allowance.
Making the window taller leaves less temporary storage for a native shader
dispatch. A larger window can therefore require more, smaller dispatches.
The fixed-group measurement profile reserves the existing capacity of128 points
(32 ordered groups of four), then chooses the largest row window that fits it.
It changes scheduling, not the physical model or any numerical tolerance.</p>
<p>In the fixed64-square backward fixture, the unconstrained64-row plan admits
100 points per cohort. Requiring128 points instead selects46 rows. The complete
image still contains every pixel in its original camera coordinates. Shared
albedo gradients retain each four-point partial and the original reverse sum.</p>
<details><summary>Does independently shading each pixel make arbitrary tile-gradient sums exact?</summary>
<p>No. Shared scene parameters receive contributions from many pixels; floating-point
addition depends on order. Preserve the original partials and their reduction order,
then test exact bytes and independent finite differences. The qualified streamed
objective here is a global RGB mean differentiated with respect to constant albedo.</p></details>
<p>Timings from different group limits cannot be interchanged in a forecast.
Both calibration and target must bind the same group capacity, wheel, helper
sources, material, device and objective. Passing small controls is separate from
admitting and completing a large frame. See the <a href="/testing/?page=render/grouped-principled">grouped execution checks</a>.</p>
<h2>Changing traversal without changing a pixel</h2>
<p>A CPU raster window can visit rectangles first, checking every triangle in each
rectangle, or visit triangles first, checking that triangle against the rectangles.
Preparing the same triangle's pixel coordinates and conservative bounds once avoids
repeating that setup. A fixed pixel still sees triangles in ascending order, so its
depth ties and floating-point fragment calculations keep the same sequence.</p>
<p>This relies on separate output storage for each pixel and unchanged coverage
predicates. It does not permit regrouping shared parameter gradients. The CPU path
uses fixed-size scalar temporaries; it introduces no retained geometry cache or new
memory allowance. Frozen original raster bytes, gradients and resource checks must
pass on the newly installed wheel before any large-frame measurement counts.</p>
<p>Instrumentation answers where time is spent. Cold imports, nested timers and
asynchronous CUDA calls can distort a small frame's apparent rate, so instrumented
timings are labelled diagnostic and cannot replace the accepted forecast anchors.</p>
<h2>Computing a narrow gradient without rebuilding its graph</h2>
<p>The scale-measurement profile fits one constant RGB albedo. Its camera,
geometry, roughness and light remain fixed during the job. The original native
material already computes a full first-derivative Jacobian and vector-Jacobian
product. Calling those same operations directly can remove repeated eager graph
construction and engine setup without changing the derivative formula.</p>
<p>This requires reproducing the lighting chain rule in its original order:
apply the coverage mask to the objective cotangent, then reverse the light
integration factors. Keep every original four-point albedo sum and the global
reverse accumulation. Do not multiply factors together early or combine groups.
The original lighting forward and its checks still run, as do the complete native
Jacobian and VJP. A graph-free call still pays the backward workspace allowance.</p>
<p>Exact partial comparisons include signed cotangents, short groups, unlit and
two-sided surfaces on CPU/CUDA. Full-frame identity, independent differences,
mutation/cancellation and storage tests cover separate obligations. Cold costs
are included in complete-job timing; there is no prewarming or subtraction to
make a forecast pass. This private profile does not qualify arbitrary losses,
geometry gradients, general differentiable tiles or final scene quality.</p>
<h2>Sharing layouts and rejecting empty regions</h2>
<p>A constant material gives each original four-point group the same field shapes.
Within one graph-free call, validate each distinct group size, then materialize the
same values together. Original values are still checked across every point and the
native operation still checks full derivative arithmetic. Varying fields and eager
gradient graphs keep their original path. Nothing is cached across invocations.</p>
<p>The CPU raster can also test a large rectangle before its smaller leaves. Use
the same conservative exclusion predicate: reject only a region it proves empty;
otherwise split its larger extent until the unchanged pixel equations can run in
small leaves. Regions partition the window, and each pixel still sees faces in
ascending order. This reduces empty-region work without a new coverage rule.</p>
<p>The explicit traversal stack is4096 bytes inside the existing streaming reserve.
Thin/wide frozen-window bytes, shared gradients and resource checks must pass on
the installed artifact. These implementation changes do not themselves establish
1080p timing admission or final visual quality.</p>
<h2>Opaque coverage: integrate the visible surface</h2>
<p>A pixel footprint has area one in pixel units. An opaque image integrates the
nearest visible surface over that footprint. A diagonal introduced only to split
a square is not a translucent edge. Fading each selected triangle separately can
therefore reveal the mesh even when its normals and material are smooth.</p>
<p><strong>Worked example.</strong> A vertical boundary at x=1.25 covers all of pixel
[0,1] and one quarter of pixel [1,2]. Their alpha values are 1 and 0.25. Moving that
boundary slightly right changes only the second pixel, whose derivative is 1.
A micro-edge surrogate instead distributes a boundary derivative across neighboring
pixels. Matching the sum does not prove either individual derivative is correct.</p>
<details><summary>Try it yourself: a triangle smaller than a pixel</summary>
<p>A right triangle has legs 1/8 pixel and lies wholly within one pixel. What is its
coverage, even if no pixel center hits it?</p><p><strong>Answer.</strong> Its area is
(1/2)(1/8)(1/8)=1/128. A center-only discovery method may report zero; that is an
explicit missing contribution, not evidence that the tiny triangle has no area.</p></details>
<p>The independent development oracle uses exact rational polygon areas and affine
depth comparisons. Its seven native comparison scenes distinguish image error,
directional derivatives and per-window invariance. A coincident-vertex clipping
bug initially discarded a derivative even though the image looked correct; the
failed report is retained. Matching a second renderer or matching CPU and CUDA
alone would not have established the intended pixel integral.</p>
<p><strong>gffx boundary.</strong> These are constant-face-colour, small-scene C11
research candidates. They do not install an opaque renderer or qualify arbitrary
shading, CUDA, streamed reverse accumulation or final showcase quality. Broad soft
visibility remains a separate fitting aid. See <a href="/testing/?page=render/opaque-coverage">the independent checks</a>
and <a href="/testing/?page=render/opaque-demo">the image and gradient comparison</a>.</p>
<h3>From the reference to a native alpha primitive</h3>
<p>The next experimental implementation integrates the union of projected triangles
in native C on the CPU. Union coverage removes artificial shared-edge fading and
includes triangles too small to hit a pixel center. Its first reverse derivative
accumulates the clipped-area derivatives in a fixed order. Independent rational
areas, nonuniform pixel weights, finite differences and window boundaries check
different parts of that claim; a single attractive image cannot replace them.</p>
<p>The explicit <code>coverage='opaque'</code> scene route currently returns only
the mask. It does not yet integrate depth-ordered shaded colour. Float32 and float64
are supported on the CPU; CUDA and higher derivatives are refused. A bounded
scratch buffer and explicit candidate-capacity refusal prevent silently dropping
geometry. This small-scene reference path has no large-frame throughput claim.</p>
<p>Scene projection has its own failure modes. A triangle crossing the camera plane
can have misleading divided depths. A retained failing test shows why front/behind
classification must happen before rejecting a distant triangle. Partial clipping
is explicitly unsupported rather than returning a misleading empty mask. The
legacy hard and broad soft paths remain separately tested.</p>
<p>Global image coordinates introduce another trap: subtracting large polygon areas
can erase a small triangle. A retained test lost an exactly representable area of
1/32768 after translating it far from the origin. Evaluate clipping and area in
coordinates local to each pixel, keeping the geometry derivatives and global
candidate discovery. Exact translated-area and gradient controls then guard this
case without clamping small contributions or loosening accuracy limits.</p>
<h2>gffx boundary: what M1 does not promise</h2>
<p>This is an eager composition, not a path tracer or a new native renderer.
There is no automatic global illumination, shadowing, display conversion, physical
transparency or fully qualified near-plane clipping. SH irradiance has a labelled
diffuse path; it is not full glossy environment lighting. Numerical tests, exact
device parity, resource measurements, visual review and consumer fitting each
have their own acceptance gate. A beautiful image closes none of them by itself.</p>
''',
},)


def worked_examples():
    """Plain arithmetic for independent tests, not a rendering implementation."""
    weights=[(1/3)/w for w in (1,2,4)]
    total=sum(weights)
    return {'perspective_weights':tuple(x/total for x in weights),
            'premultiplied_red':(.25,0.,0.),'coverage':.25,
            'red_over_white':(1.,.75,.75)}

CHAPTERS[0]['body'] += """
<h2>Material transitions: integrate every visible region</h2>
<p>Opaque coverage belongs to the union of visible surfaces, not to separate
triangle opacities. A material jump changes radiance at its interface; moving that
interface contributes the radiance difference times its swept area. A continuous
bilinear texture instead has cell boundaries where its derivatives change.</p>
<p>Perspective UV and roughness thresholds become homogeneous affine lines in a
screen triangle. Split visible polygons at those lines before integration. Native
quadrature now emits face identity, screen barycentrics and area weights; the scene
then evaluates perspective attributes and full nonlinear Principled shading at the
nodes. Its reverse pass transports both node positions and integration weights.</p>
<p>The independent material reference uses exact moving partitions, outward-rounded
mixed derivatives, and a centred Taylor envelope with a seventh-order remainder
for the tested roughness interval. Image and first-gradient errors are enclosed
separately. A thin-feature negative control prevents a sampled zero from becoming
proof, and uncertain limiter branches refuse instead of picking a midpoint.</p>
<p>The scene composition is an explicit bounded CPU/CUDA float64 Principled/directional
profile with bilinear level-zero albedo and clamp/repeat addressing. Fixed native
Gauss5 geometry is not a universal error certificate: arbitrary branch crossings,
horizons, dense workloads and the final showcase still need gates. Small streamed
windows and device conformance have separate finite qualification.
Integration bounds, native floating rounding and physical loss-table approximation
are three different claims.</p>
<h2>Moving visibility: differentiate the region and the material</h2>
<p>A moving silhouette changes which part of a pixel sees the object. With a
vertical boundary at t and foreground to its right, the pixel derivative is
minus the foreground radiance at t. At a depth intersection, moving the boundary
exchanges surfaces: the derivative is the left radiance minus the right radiance.
Freezing visibility would lose these terms even if shading gradients were correct.</p>
<p>The reference now transports the exact visible polygons and their integration
Jacobians together with the shading field. Perspective attributes attached to a
moving triangle subtract its interpolated screen velocity before evaluating the
field tangent. An entirely visible translated material must not acquire a false
colour change merely because its screen coordinates changed.</p>
<p>Exact polynomial predicates certify a local interval in which visibility keeps
its combinatorial structure. A zero predicate is accepted only if it is identically
zero; a quadratic change invisible to a first-order test is refused. The two
nonlinear fixtures have a certified geometry interval of plus/minus 1/1024.</p>
<p>Sixteen exact controls and five nonlinear/installed-wheel checks pass. Independent
90-digit boundary formulas lie inside the gradient enclosures. A different Gauss16
integral using installed shader calls and autodiff agrees within the existing
floating-point tolerances. Its rounding can exceed the much tighter quadrature
interval; the integration bound does not certify native rounding. General topology
transitions, material knots, varying roughness/textures, production nonlinear scene
colour, CUDA and final showcase acceptance remain open.</p>
<h2>Nonlinear shading: an error interval for the actual pixel</h2>
<p>Shading at triangle corners and interpolating their colours changes a nonlinear
Principled image. The reference instead interpolates the surface attributes with
perspective correction, evaluates the complete material, and integrates radiance
over each exact frontmost region of the pixel.</p>
<p>A sample-grid comparison can miss a narrow feature on both grids. Our bounded
reference uses outward-rounded arithmetic and a Gauss remainder derived from
interval bounds on spatial derivatives. RGB and the chosen parameter tangent get
separate error intervals. Polynomials that vanish at every coarse quadrature node
are deliberate negative controls: their positive integrals must not become zero.</p>
<p>The first smooth, fixed-roughness pixel passed with 44 accepted cells from 87
examined. Its RGB interval radii are at most 1.185e-13, and the tested normal/albedo
tangent radii at most 2.987e-14, inside the unchanged 1e-12 reference budget.
The installed shader evaluated at 1100 weighted points agrees inside all six
intervals. This comparison is not the source of the error bound.</p>
<p>The bound concerns integration of the stored material-resource model, not its
physical-table approximation error. Spatially varying roughness, texture cells,
interpolation-knot and horizon partitioning, moving visibility, full scene colour,
CUDA and final showcase acceptance remain open. Failed depth/time-limit attempts
are retained. Unresolved work refuses a result; it never returns a last estimate.</p>
<h2>Visible colour: integrate after resolving depth</h2>
<p>The CPU <code>opaque_integrate</code> primitive integrates a declared
screen-affine RGB field over the nearest visible polygon pieces of each pixel.
It returns premultiplied RGB and covered area; a constant background is added as
<code>RGB + (1-alpha) background</code>. Shared edges between triangles representing
the same affine field do not introduce transparency. A depth intersection can
create an interior visibility boundary even when no triangle edge crosses a pixel.</p>
<p>The independent Fraction reference integrates each convex piece by exact
triangle-fan centroid quadrature; native C uses polygon first moments. Image
agreement, geometry/depth derivatives and corner-radiance derivatives are checked
separately. The existing float64 1e-12 and float32 1e-6 oracle bounds remain fixed.</p>
<p>This stage does not approximate a nonlinear Principled shader by calling it at
vertices. The bounded smooth-pixel reference above begins nonlinear and perspective-varying
radiance qualification; complete scene integration and derivatives remain open. The current CPU candidate cap is32;
overflow is an explicit refusal. CUDA, dense-scene throughput and final shaded
showcase acceptance remain open.</p>
"""

CHAPTERS[0]['body'] += """
<h2>Bounded windows: one image, one ordered derivative</h2>
<p>Opaque colour windows keep the original camera coordinates and resolve every
pixel's visible surfaces. The optional <code>opaque_tile_pixels</code> caps window
width; the planner chooses a fitting width automatically when it is omitted.
Outputs, cotangents, input adjoints and temporary work still count against64MiB.
A window cannot make an oversized retained output disappear.</p>
<p>Backward saves inputs and recomputes one pixel at a time in a fixed order,
independent of the forward window width. This avoids retaining every window's
shading graph and keeps accumulation predictable. The initial CPU/CUDA scene
comparison exposed differing three-term perspective sums; explicit normalization
and its VJP repaired it. The failed report remains visible.</p>
<p>Independent piecewise-constant pixel tests now include roughness0.15 through0.95,
metallic and dielectric endpoints, two-sided orientation and unlit surfaces.
Exact region motion supplies the interface gradient. A region whose interval
crosses an unresolved horizon or angular knot is refused, not assigned a sampled
branch. These finite fixtures and the small CUDA windows are engineering evidence;
they do not yet establish the original mesh or the final SIGGRAPH-quality showcase.</p>
"""

CHAPTERS[0]['body'] += """
<h2>Returning to the original mesh: admission is not accuracy</h2>
<p>The original 5,002-vertex, 10,000-triangle fixture has constant roughness 1,
F0 0.04 and albedo (0.2, 0.4, 0.6). Conservative projected-box counts stay below
the 32-face cap: maxima are 30 at 128 square, 20 at 256 square, and 18 at both
512 square and 1080p. These counts establish candidate capacity only; they do
not establish visible-piece capacity, nonlinear accuracy or frame time.</p>
<p>At 1080p the retained output/cotangent reserve alone is 200,179,712 bytes,
above the unchanged 64 MiB allowance. Smaller temporary tiles cannot remove
retained full-frame storage. No full original-mesh opaque frame is admitted.</p>
<p>Original pixel (160,128) in the 256-square camera exposed a reference problem:
angular-table knots break the smoothness assumption of a derivative remainder.
The new reference freezes a smooth cubic extension per integration cell, then
adds an outward enclosure of the actual piecewise expression minus that extension.
A separate mean-value cosine bound reduces interval dependency. Neither a
midpoint branch nor agreement between sample grids is an error certificate.</p>
<p>Image and angle-independent material derivatives are separate obligations.
Moving-angle and horizon crossings still require qualification. Cell, depth,
time and numerical limits are unchanged; incomplete attempts remain explicit
refusals in the preserved reports. This is engineering evidence on the path to
the requested SIGGRAPH-quality demonstration, not showcase acceptance.</p>
"""

CHAPTERS[0]['body'] += """
<p>The first original pixel is now independently certified: two visible triangles,
128 examined cells and 65 accepted leaves provide separate RGB and albedo-tangent
radii below 1e-12. Installed CPU and CUDA values lie inside those enclosures;
their RGB, mask and full albedo Jacobian bytes also agree. This is one pixel with
fixed geometry, not a whole image or a moving-angle gradient certificate.</p>
"""


CHAPTERS[0]['body'] += """
<h2>Moving knots and horizons: the missing boundary term</h2>
<p>Changing geometry or view direction moves the angular-table boundary through
an integration region. The reference now adds the ordinary piecewise derivative
and a separate conservative jump-transport bound. It certifies a crossing only
when a spatial derivative has fixed sign throughout the cell; uncertain tangencies
refuse. One real-resource knot and rigid geometry motion pass independent bounds
on CPU and CUDA. This does not certify every moving scene.</p>
<p>At a view horizon,pointwise differentiation can miss a radiance jump entirely.
The analytical test gives a positive unpartitioned gradient of0.042997,while the
correct pixel gradient is negative,-0.109973. An explicit moving horizon cut
recovers the boundary contribution and passes on both devices. A light-horizon
case is checked separately. General production horizons remain open: varying
normals and view directions can make the boundary curved.</p>
<h2>Full-frame storage is not full-frame speed</h2>
<p>The internal-disk development store preserves RGB,mask and their cotangents,
rejects incomplete frames and checks every block before replay. A1080p synthetic
frame uses132710400 disk bytes with at most32768 payload bytes per block.
A real3x3 textured scene preserves each device's resident pixels and ordered
vertex gradient exactly. Neither result is a1080p render timing.</p>
<p>The original32/64-square startup frames have maximum candidate counts68/44,
above the unchanged32-face cap. No complete matching opaque timing anchors exist,
so a full-frame forecast is refused. Old hard/soft timings and one-pixel crops
cannot qualify this new execution profile. Production horizon integration,
external-storage wiring and a bounded complete-frame cost path remain required
before the original mesh and final SIGGRAPH-quality demonstration are admitted.</p>
"""


CHAPTERS[0]['body'] += """
<h2>Production horizons, exact arithmetic and host-owned windows</h2>
<p>The production opaque scene now inserts linear horizon cuts. Constant or
uniform normals give a linear view boundary; directional lighting gives a linear
light boundary. A varying normal dotted with a varying view can be quadratic.
Conservative coefficient signs can exclude a crossing; unresolved curved and
grazing boundaries refuse. The shared cut limit and numerical criteria stay.</p>
<p>An off-centre3x3 camera exposed a scalar-division rounding difference before
shading. Tensor dimension division and an explicit RGB coverage-weight VJP
restore all captured CPU/CUDA stage and final-gradient bytes. The smaller knot
case separately needs fixed caller broadcast/channel/sample reduction order.
Exact agreement is checked independently from the reference error bounds.</p>
<p>The public opaque_window operation uses the original camera coordinates and
reserves only the requested rectangle, plus inputs, cut/horizon storage and
scratch. A host driver writes complete ordered internal files and replays one
pixel at a time. Input mutation, cancellation and corruption refuse. File I/O
and scheduling remain outside the renderer. Finite complete-frame reports name
their actual scene and resolution; they never stand in for the original mesh or
the requested final SIGGRAPH-quality demonstration.</p>
"""


CHAPTERS[0]['body'] += """
<h2>Following a curved view horizon</h2>
<p>The opaque61 checkpoint below established the independent reference; the following native section records its implementation.</p>
<p>The original mesh has174 retained faces whose normal/view dot product changes
sign. Because both interpolated vectors vary, their homogeneous dot product is
quadratic. A straight cut generally changes the forward integral and its gradient.
The independent reference now follows a certified curved boundary on one visible
triangle. It maps the curved region to a square and differentiates both the
mapping and its area weight. That preserves the boundary contribution which a
pointwise shading derivative would miss.</p>
<p>Exact polynomial signs certify one transverse crossing along each radial ray.
Tangencies and unsupported topologies refuse. Closed parabolic coverage and
moving-triangle formulas test the reference separately. Direct radial composition
of perspective fields avoids interval cancellation on a thin triangle without
relaxing accuracy. The real selected patch passes in8 cells with image and motion
radii below1e-12; the earlier90-second refusal remains historical evidence.</p>
<p>This uses actual original geometry with a declared roughness1,F0=1,albedo0
control material. Full original material integration,shared native CPU/CUDA curved
quadrature and complete-frame admission remain open. The new diagram explains
this gate; it is not a replacement for the requested final rendered showcase.</p>
"""

CHAPTERS[0]['body'] += """
<h2>Native curves: images,gradients and complete frames</h2>
<p>The shared native CPU/CUDA path now follows the conic root and differentiates
both sample locations and area weights. Linear cuts keep their existing path.
A strict topology guard is separate from a shader integration-error bound;
unsupported tangencies and over-cap pixels still refuse.</p>
<p>The first native curved pixel had an accurate image but an inaccurate gradient.
Two angular strips per side repaired that pixel,then a2-by-2 frame exposed a
larger gradient error after clipping. Four strips per side consume eight of the
existing32 pieces and200 of800 nodes. Neither the numerical tolerance nor the
memory allowance was raised. These failures remain visible in the test history.</p>
<p>Complete-pixel RGB and normal motion are checked against an independent
outward reference. CPU/CUDA bytes,host-storage replay,frame-integral accuracy
and complete execution timing are separate gates. The tests name their small
control scene and resolution. Actual original-mesh full-material integration and
the requested final SIGGRAPH-quality demonstration are still required for M1.</p>
"""


CHAPTERS[0]['body'] += """
<h2>The original material: execution and accuracy are separate</h2>
<p>The complete silhouette pixel now evaluates every visible piece using the
original blue material,F0=.04 and roughness1. A compact angular quadrature rule
uses120 nodes per crossed fan instead of200,while keeping the800-node ceiling.
Constant material bindings also allow the planner to reserve fewer impossible
cut arrays; unknown material-cut profiles retain the conservative reservation.
The initial full-pixel memory and capacity failures remain in the test history.</p>
<p>CPU and CUDA produce identical colour,mask and albedo-Jacobian bytes for this
pixel. Their agreement does not establish the correct integral. The independent
reference has to follow the actual piecewise angular-loss table as well as the
curved visibility boundary. Its current residual bounds exhaust256 cells on the
first visible piece. Neither accuracy nor the work budget was relaxed.</p>
<p>This leaves complete original-frame qualification and the final SIGGRAPH
showcase open. The next numerical work is a tighter certified treatment of
angular joins,including boundary transport for moving-angle gradients. A small
successful control image is not the requested final original-mesh demonstration.</p>
"""


CHAPTERS[0]['body'] += """
<h2>Certifying one piece does not certify the pixel</h2>
<p>The next reference integrates the signed correction around angular lookup
boundaries and bounds its remaining error. A Peano kernel measures how a
quadrature rule responds to changes in the integrand. Here the bound includes
ordinary third derivatives and explicit jumps in value,slope and curvature.
It never assumes that stored binary knots join perfectly. Polynomial convex
hulls provide the kernel bounds; sampled maxima are not certificates.</p>
<p>Exact rational dot products reduce repeated-normalization uncertainty without
changing the shading formula. Closed-form tests check the correction,its spatial
derivatives,box scaling and refusal of unsupported moving-angle tangents.
The checker now propagates an external deadline immediately; its earlier
subdivision retry is retained as a failed regression.</p>
<p>This certifies the first of13 exact visible pieces of the original blue-material
pixel. Installed CPU and CUDA shading and albedo derivatives agree with that
piece's independent interval. The whole-pixel experiment still reaches its
shared90-second deadline on the second piece,so there is no whole-pixel pass.
Continuing each piece with a fresh timer would not satisfy this test.</p>
<p>The remaining work needs a more efficient partition that follows the angular
boundaries while retaining the same error and work limits. Complete original
frames and the final SIGGRAPH-quality showcase remain open. The earlier gallery
continues to show finite engineering evidence,not a completed showcase.</p>
"""

CHAPTERS[0]['body'] += """
<h2>Bounding an approximate partition</h2>
<p>An affine classifier L may approximate a nonlinear cosine q while retaining a
rigorous bound |q-L| at most delta. If their branch choices disagree at knot k,
the point lies inside |L-k| at most delta and |q-k| at most delta. Rational
clipping gives the area of that band. A Taylor bound on the adjacent branch
difference, including stored value and slope jumps, then bounds the omitted
integral. Adding that interval preserves the original material rather than
silently replacing it with the classifier's appearance.</p>
<p>A closed example uses q=x+epsilon*y squared and L=x over a unit square.
The omitted curved strip has an exactly integrable polynomial contribution.
That independent result checks the integrated error bound, not just sampled
classifier points. Parameter derivatives require separate moving-boundary
transport and are deliberately outside this fixed-geometry certificate.</p>
<p>The first original piece now certifies much faster, but the refined whole
pixel stops at256 combined cells on its second piece. A diagnostic charging
only classifier visits and one root per subregion also exhausts256 there.
Thus faster arithmetic alone cannot admit this partition. The next method must
reduce the number of regions while retaining the same summed error bound.
The complete original pixel, frame and final3D showcase remain open.</p>
"""


CHAPTERS[0]['body'] += """
<h2>Choose refinement work without changing acceptance</h2>
<p>The residual checker now keeps separate certified error contributions for
its two integration directions. When one direction dominates,it subdivides that
direction. A closed clipped-quadratic example checks both orientations: cutting
an independent constant coordinate wastes cells but cannot improve the bound.
The hint chooses work;the complete outward interval still decides acceptance.</p>
<p>Exact triangle areas also divide the existing total error allowance unevenly,
with downward rounding. If the specular and matte basis integrals each have
summed radius at most5e-13,RGB=specular+albedo*matte has radius at most1e-12
for albedo between zero and one. The finalRGB and albedo-Jacobian intervals
are checked separately. Directed Decimal60 endpoint arithmetic is unchanged;
sharing its rounding context reduces overhead without dropping any endpoint.</p>
<p>The latest original-pixel attempt reaches the256-cell limit after64.332seconds,
with only its first of13pieces certified. Its earlier90-second failure remains.
The improvement exposes a work-budget bottleneck;it does not qualify the whole
pixel. A separately bounded offline campaign has been proposed for review.
It has not been executed or accepted,and neither a changed deadline nor a
passed helper test can relabel these failures. Original moving gradients,
complete frames and the final3D showcase remain open.</p>
"""

CHAPTERS[0]['body'] += """
<h2>Approved offline campaign: a useful failure</h2>
<p>The user approved a separate offline reference campaign for the thirteen exact
visible pieces of one original pixel. Each piece retained256cells,depth10,
90reference seconds and1GiB host memory;the aggregate allowed3328cells and
1170reference seconds. Renderer limits and the1e-12 pixel radius did not change.</p>
<p>Thirty-one finite contract and routing checks pass. They check ordered complete
coverage,exact input/source/result identity,resource enforcement and conservative
interval composition. An early false rejection came from outward summation of
already rounded error shares;exact rational budget accounting fixes that while
numerical integrals still use outward arithmetic. These tests qualify the
checker,not the original pixel. All earlier failed checks remain in the history.</p>
<p>The first original piece passed141cells in34.483reference seconds. The second
stopped at256cells after64.860seconds,below its90-second limit. The campaign
stopped immediately;the other eleven pieces were never started. Including the
failed work,it consumed397cells and99.343reference seconds. No complete-pixel
certificate was assembled. Its last recorded cell is dominated by the angular
boundary correction;that diagnostic is not the total remaining error.</p>
<p>The next numerical work must improve the independently tested boundary-error
bound or integration method. Another unchanged run or a larger aggregate budget
would not resolve this per-piece failure. Original moving gradients,complete
frames,the four1080p backward cases and the final3D showcase remain open.
Current2D demonstrations stay available as their own finite evidence.</p>
"""

CHAPTERS[0]['body'] += """
<h2>Use the crossing location without weakening the proof</h2>
<p>The previous campaign remains stopped. A new analytical checker improvement
uses the mean-value theorem to bound where each angular knot can cross a cell.
When the derivative interval includes zero,it keeps the entire cell and the
existing root-count bound. Curved nonmonotone crossings still refuse.</p>
<p>Jump-error kernels are piecewise polynomials. Their outward Bernstein hulls
can be bounded over the certified crossing interval instead of the whole cell.
Every overlapping quadrature-node segment participates,including both sides of
uncertain nodes. Value,slope and curvature jumps all remain charged. The
ordinary third-derivative error is separate and still present.</p>
<p>Twenty new controls pass against closed integrals and a piecewise-log material
antiderivative. Named examples improve uncertainty by about1170-fold near a
cell endpoint,136-fold for an oblique crossing,and36-fold for the log-material
fixture. These are finite error-bound improvements,not renderer speedups.
Sixteen prior Peano and seven square controls also pass. Earlier reds and
fixture mistakes are preserved. A failed refinement now retains the last
complete-domain enclosure and separate error components.</p>
<p>A new original-pixel campaign has been prepared for explicit review with the
same per-piece and aggregate limits. It has not run. Analytical improvement
does not promise that every original piece fits its allowance. Original moving
gradients,frames,scale completion and the final3D showcase remain open.</p>
"""

CHAPTERS[0]['body'] += """
<h2>The original case measures the limit of the improvement</h2>
<p>The user approved the fresh localized-bound campaign. Its first piece passed
133cells in35.093reference seconds,using eight fewer cells than before. Its
second piece still exhausted256cells after72.451seconds. The campaign stopped
immediately and never started the remaining eleven pieces. Failed work remains
charged:389cells and107.544reference seconds in total.</p>
<p>The new complete-enclosure diagnostic makes the gap explicit. Before its last
refinement,the second piece had128leaves and matte width1.610e-12. Its allowed
width is1.232e-13,so the bound is13.063times too wide. This is a complete-domain
bound. The last cell also records angular-jump uncertainty larger than its
ordinary correction and smooth quadrature error;that component split applies
to the last cell only.</p>
<p>The finite analytical tests remain valid,but their improvements did not make
the original case fit. Both campaigns are terminal failures for full-pixel
acceptance. There is no complete certificate and no new native-accuracy pass.
A materially tighter treatment of boundary error is needed before another
reviewed original run. All tolerances,renderer limits and history remain intact.
Moving gradients,complete frames,scale completion and the final3D showcase stay
open. The existing2D demonstrations retain their separately scoped acceptance.</p>
"""

CHAPTERS[0]['body'] += """
<h2>Keep the direction of a certified boundary correction</h2>
<p>The next checker retains signed value,slope and curvature jumps instead of
always charging their absolute magnitude as a symmetric error. Polynomial
kernel hulls still include both sides of uncertain Gauss nodes. The integral
is the quadrature result minus the signed error,with the ordinary remainder
and every unresolved crossing still enclosed conservatively.</p>
<p>A material crossing gets this treatment only when its derivative has a
strictly certified direction. Endpoint bounds establish whether it is present
on every transverse line;otherwise zero stays in the contribution interval.
Nonmonotone crossings keep their earlier multiplicity bounds,and curved
nonmonotone cases still refuse. No stored binary discontinuity is discarded.</p>
<p>Exact oblique and curved polynomial controls give roughly616-fold narrower
bounds. Two closed logarithmic-material cases reach the unchanged1e-12 radius
in3cells instead of7. These are finite reference-checker measurements,not
renderer speedups. A separate near-endpoint fixture improves only about3percent
overall because its ordinary remainder dominates;the earlier2x prediction
failed and stays visible alongside the rounding and regression failures.</p>
<p>Seventeen signed controls,twenty-five localized/material controls,sixteen
Peano controls and seven square controls pass. The original material has not
been rerun with this candidate. Both earlier campaigns remain terminal,and
original accuracy,moving gradients,complete frames,the four1080p backward
cases and the final SIGGRAPH-quality3D showcase remain open.</p>
"""

CHAPTERS[0]['body'] += """
<h2>The signed method still needs a tighter original bound</h2>
<p>The approved signed-bound campaign ran once. The first piece passed133cells
in34.589reference seconds. The second exhausted256cells after71.314seconds,
and the remaining eleven were never started. Including failure,the work totals
389cells and105.903reference seconds. All historical failures remain visible.</p>
<p>The complete second-piece matte bound is about2.2percent narrower than the
previous campaign,but still12.772times too wide. The last recorded cell's signed
boundary intervals remain nearly symmetric about zero. Retaining a sign cannot
remove uncertainty when both signs remain possible. This describes that cell;
it is not a measured decomposition of the complete piece.</p>
<p>The next method needs to retain how a crossing varies across the cell or
integrate its contribution more directly. Exact analytical controls must qualify
that method before a new original experiment. There is no complete original
pixel certificate,no new native-accuracy pass and no final3D-showcase acceptance.
The renderer,tolerances,worker limits and accepted2D demonstrations are unchanged.</p>
"""


CHAPTERS[0]['body'] += """
<h2>Integrate how the crossing moves across the cell</h2>
<p>A tighter checker now evaluates one directional error at each actual outer
quadrature node. For the other direction, it integrates the boundary contribution
along the crossing itself. This coordinate change is admitted only when both
coordinate derivatives have strictly certified signs over the complete cell.
Clipped endpoints, reversed crossings and all stored material jumps remain.</p>
<p>The error kernel has a known polynomial primitive. A constant part of its
material coefficient can be integrated exactly; the coefficient's remaining
variation still receives a conservative absolute-error bound. Uncertain roots
and nonmonotone cases cannot borrow that cancellation. Ordinary derivative
remainders, outward rounding and every resource cap remain in force.</p>
<p>Forty-five analytical controls pass. On two exact oblique log-material cases,
the new checker uses7cells and387material evaluations, compared with121cells/
4043evaluations and133cells/4439evaluations for the previous method. Both methods
must enclose the independently integrated answer at the same1e-12radius. These
are checker costs on named fixtures, not rendering benchmarks.</p>
<p>The new path is connected to the development reference, but has not been
run on the original mesh. Three original campaigns remain failed and terminal.
Their history, the new test-first failures and fixture corrections stay visible.
An approved fresh experiment must still certify all thirteen original pieces
before any complete-pixel accuracy claim. Moving gradients, complete frames,
remaining scale cases and the final SIGGRAPH-quality3D showcase remain open.</p>
"""


CHAPTERS[0]['body'] += """
<h2>The transverse method still misses the original acceptance gate</h2>
<p>The approved original trial stopped on its second piece. The first passed
133cells in46.26reference seconds. The second attempted231cells before the
90-second reference deadline;its cooperative stop was recorded at90.177seconds.
The remaining eleven pieces were never started. All364attempted cells and
136.439reference seconds remain charged.</p>
<p>The last complete cover,at229cells and115leaves,still has a matte interval
15.252times too wide. It cannot certify the original pixel. The bound is at a
different work point from the previous trial,so those widths are not a
matched-work comparison. Large improvements on exact analytical examples did
not establish sufficient improvement on this original material.</p>
<p>The last saved cell retains most boundary uncertainty in its continuous-axis
term. That is one cell,not a decomposition of the whole piece. The logs do not
yet distinguish uncertain crossing endpoints,coefficient variation and fallback
terms. A bounded witness diagnostic is the next step before another justified
repair. All four original campaigns remain terminal;no limit was relaxed.</p>
<p>The production renderer,installed wheel and accepted2D demonstrations remain
unchanged. Original accuracy,moving gradients,complete frames,remaining scale
cases and the final original-mesh3D showcase are still required for M1.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Tighter endpoint bounds, with the uncertainty still included</h2>
<p>The next diagnostic inspected three saved regions from the failed second
piece, in both integration directions. All22boundary witnesses admitted the
coordinate change;none used a fallback. Uncertain crossing positions made some
polynomial-primitive intervals much wider than the older valid bounds. The
large terms in the primitive cancel at a point,but direct interval evaluation
loses that dependence across a range.</p>
<p>The repair encloses the primitive through an exact constant anchor and a
bound on its derivative over the entire endpoint interval. Where that derivative
has a certified sign,monotonicity gives endpoint bounds too. The result is
intersected with the previous enclosure;each integrated boundary contribution
is also intersected with its independent uniform bound. Contradictory bounds
are an error. This retains uncertain roots and every stored material jump.</p>
<p>Fourteen endpoint controls,19diagnostic controls,45material controls and96
affected regressions pass. On the three saved regions,the currently used
orientation gives correction bounds3.06,2.63 and1.45times tighter. The diagnostic
uses the same322correction input calls;fixed preparation is timed separately.
Its total time rises from1.71to2.16seconds. These overlapping regions do not
constitute a complete-piece bound,and this is not a rendering speedup claim.</p>
<p>No fresh full original trial has run. Its frozen scope needs approval before
execution. All earlier failures remain visible. Original accuracy,moving
gradients,complete frames,remaining scale cases,gffit readiness and the final
original-mesh3D showcase remain open;the accepted2D demonstrations are unchanged.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>The tighter endpoint trial still needs a complete accuracy certificate</h2>
<p>The approved original-pixel trial passed its first visible piece with93cells
in34.22reference seconds,compared with133cells and46.26seconds previously.
The second piece reached the90-second reference deadline after222attempted
cells;its cooperative stop was recorded at90.148seconds. The remaining eleven
pieces were not started. Every attempt remains charged and preserved.</p>
<p>The last complete cover has221cells and111leaves. Its matte interval is
5.1435times wider than allowed. The prior trial was15.2517times too wide at
229cells:these are different subdivision states,not a matched-work comparison.
The improvement is insufficient for a complete original-pixel certificate;
it does not establish either renderer accuracy or renderer inaccuracy.</p>
<p>In the final saved cell,ordinary smooth-part remainder accounts for87.67percent
of the recorded correction radius. That describes one cell only. The saved
cover does not report aggregate components across all111leaves,so a whole-piece
cause cannot be inferred. The next bounded investigation should distinguish
those contributions and qualify any tighter smooth-part bound on independent
exact controls before proposing another full original trial.</p>
<p>All five original campaigns are terminal. No budget was raised,no failed
history was removed,and no old certificate was reused with changed source.
M1 still needs original accuracy,moving gradients,complete frames,the remaining
scale cases,gffit readiness and the final original-mesh3D demonstration.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>Four fixed regions for the smooth remainder</h2>
<p>The bounded investigation inspected three saved regions from the failed
second piece. Simple constant centering barely helped the total bound. Dividing
the derivative enclosure into four fixed regions gave correction bounds
1.78, 2.15 and 1.71 times tighter in the current integration direction.</p>
<p>The rule already integrates cubics exactly. Its third-order error kernel
therefore has zero total mass: subtracting one constant from the third
derivative leaves the integral unchanged. Symmetry divides the absolute kernel
mass equally between the two inner halves. The bound uses two inner halves by
two outer halves, with one shared inner anchor per outer half. Independent
inner anchors would incorrectly discard real variation. Every material jump
and the other directional remainder remain included.</p>
<p>This costs 24 extra third-order evaluations across the three regions and
two directions: 275 becomes 299 input calls. Diagnostic time rises from 1.71
to 1.99 seconds. These overlapping regions do not form a whole-piece cover,
so the improvement cannot establish that the next original trial will pass.</p>
<p>178 analytical, diagnostic and evidence-handling controls pass, alongside
the bounded original-region diagnostic. Exact polynomial and piecewise-log
integrals independently check the bound. All 37 test-first failures remain
visible. Future results also retain component totals across the last complete
cover and at most five largest leaves, distinguishing conservative uncertainty
from an observed rendering error.</p>
<p>No new full original trial has run. All five old campaigns are terminal;
the candidate needs a separately approved frozen trial with unchanged limits.
Production rendering and the installed wheel are unchanged. M1 still needs
original accuracy, moving gradients, frames, remaining scale cases, gffit
readiness and the final original-mesh SIGGRAPH-quality 3D showcase.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>The four-region trial: two remaining sources of uncertainty</h2>
<p>The approved trial passed its first visible piece in81cells and32.64reference
seconds. Its second piece reached the90-second reference deadline after202
attempted cells, and the remaining eleven pieces were not started. The last
complete cover contains101leaves; its matte interval is still4.5124times wider
than allowed. No complete original-pixel certificate was produced.</p>
<p>The previous trial was5.1435times too wide, but at a different refinement
state:221cells versus201in the latest complete cover. This is not a matched-work
comparison. All283attempted cells and122.81reference seconds remain charged.</p>
<p>We can now inspect the entire retained cover. Ordinary derivative remainder
accounts for53.62percent of its correction-radius bound and boundary uncertainty
for46.38percent. Neither can be ignored. The five largest regions together
account for only15.30percent of total width, so this is not one exceptional
region. These percentages describe conservative uncertainty in the independent
reference, not measured errors in the renderer.</p>
<p>The next investigation should use saved region coordinates to compare the
two integration directions, ordinary derivative variation and boundary terms
at bounded cost, before qualifying another repair. All six campaigns are
terminal. Accuracy thresholds, worker limits, the production wheel and accepted
2D demonstrations are unchanged. M1 original accuracy, moving gradients, frames,
remaining scale cases, gffit readiness and the final3Dshowcase remain open.</p>
"""


CHAPTERS[0]['body']+=r"""

<h2>Three saved regions: center the conditioned remainder</h2>
<p>The three largest retained regions from the failed second piece were checked
in both integration directions. The existing direction remains tighter on all
three. The complete saved cover still has both ordinary derivative uncertainty
and boundary uncertainty; this repair retains every boundary contribution.</p>
<p>At each outer Gauss node, the full inner third-order error kernel has zero
integral because the quadrature integrates cubics exactly. Subtract one constant
anchor from the certified inner derivative range, then bound its remaining
variation. Keep one anchor for the entire inner interval, the positive outer
weight, physical scale, all jumps and the other directional remainder.</p>
<p>The current-direction correction bounds tighten by1.2142,1.1459 and1.4391
times. Both diagnostics use249material inputs and8boundary witnesses. Time was
1.47seconds before and1.29seconds after on this single bounded measurement.
Integrated and boundary components are exactly unchanged. Repeated kernel
calculations now use an exact-endpoint cache capped at128entries, storing
immutable endpoints and returning fresh intervals.</p>
<p>263 analytical, material, diagnostic and evidence-handling controls pass,
alongside the original-region diagnostic. Two exact log fixtures meet the same
1e-12radius in5rather than7cells, using311rather than403input calls. This combines
less subdivision with kernel reuse; the observed gain is not attributed solely
to the cache. Earlier stopping can leave a wider, still accepted final interval.
Twelve expected test-first failures and one corrected rational-endpoint fixture
failure remain historical evidence.</p>
<p>No full original trial has run with this repair. Three regions cannot predict
whole-piece success; the previous complete cover remained4.5124times too wide.
All six old campaigns are terminal. Accuracy and resource limits, the installed
wheel and accepted2Ddemonstrations are unchanged. Original accuracy, moving
gradients, frames, remaining scale cases, gffit readiness and the final original
mesh SIGGRAPH-quality3Dshowcase remain open.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>The centered trial still needs both error terms</h2>
<p>The approved trial passed its first visible piece in 79 cells and 32.38
reference seconds. The second piece reached its 90-second reference deadline
after 204 attempted cells, and the remaining eleven pieces were not started.
The last complete cover has 102 leaves; its matte interval is still 4.1265 times
wider than allowed. No complete original-pixel certificate was produced.</p>
<p>The preceding trial ended 4.5124 times too wide on a different subdivision:
201 complete cells versus 203 here. This is not a matched-work comparison.
All 283 attempted cells and 122.54 reference seconds remain charged.</p>
<p>The saved cover splits correction uncertainty into 51.39 percent ordinary
derivative remainder and 48.61 percent boundary uncertainty. Each component
alone exceeds the entire allocated radius on this cover: 2.11 times for the
ordinary part, 2.00 times for the boundary part. Omitting either cannot prove
acceptance here. These are conservative bounds, not observed image errors.</p>
<p>The next bounded investigation should examine boundary coefficient variation,
root uncertainty and the cost of derivative enclosures on matched saved regions.
Any repair still needs independent exact controls. No unchanged retry is planned.
All seven campaigns are terminal, all failures are preserved, and accuracy,
worker limits and the production wheel are unchanged. M1 original accuracy,
moving gradients, frames, scale cases, gffit readiness and the final original-mesh
SIGGRAPH-quality 3D showcase remain unfinished.</p>
"""


CHAPTERS[0]['body']+=r"""

<h2>Localize the jump calculation and remove repeated work</h2>
<p>The bounded investigation revisited the same three largest saved regions in
both integration directions. For integrated boundary witnesses, 96 to 100
percent of uncertainty came from variation in the coefficient multiplying the
error kernel. The curvature-jump contribution dominated; refining root
endpoints alone would leave most of this uncertainty.</p>
<p>Every crossing already has a certified containing interval. We now evaluate
its spatial derivatives on that smaller interval before forming the jump bound.
All value, slope and curvature jumps, orientation, absence cases and possible
crossing counts remain included. This is a fixed-geometry argument; it does
not differentiate the moving boundary or qualify moving-geometry gradients.</p>
<p>The checker also reuses axis inputs instead of calculating them twice.
Absolute and signed kernel bounds share the same polynomial hull through the
existing 128-entry cache. These savings offset the extra crossing evaluations:
240 inputs become 233, with all eight boundary witnesses retained.</p>
<p>Total correction widths in the current direction are 1.35, 1.47 and 1.16
times tighter on the matched regions. Ordinary derivative bounds are exactly
unchanged. One profiled diagnostic takes 3.33 seconds before and 2.98 after;
profiling overhead and the small scope prevent a production speed claim.</p>
<p>288 affected analytical, material, diagnostic and contract controls pass,
alongside the original-region diagnostic. Preserve 25 expected test-first
failures and two corrected launch failures: a diagnostic registration syntax
error and an invalid test-mode name. Neither launch failure ran a numerical
worker. The previous original trial remains terminal at 4.13 times the allowed
width; this repair does not establish whole-piece acceptance.</p>
<p>Next examine the ordinary derivative enclosure and its evaluation cost on
matched saved regions before preparing another full trial. All accuracy and
resource limits, the production wheel and accepted 2D demonstrations remain
unchanged. M1 and the final original-mesh SIGGRAPH-quality 3D showcase are open.</p>
"""


CHAPTERS[0]['body']+=r"""

<h2>The same polynomial with tighter uncertainty bounds</h2>
<p>A smooth material branch is a cubic polynomial in the logarithm of the
view or light cosine. The checker previously repeated its Bernstein expression.
With interval arithmetic, repeated appearances of the same variable can produce
unnecessarily wide bounds. Rewriting that same cubic around its midpoint and
evaluating it with Horner's rule reduces this dependency and the work required.</p>
<p>This changes only the independent fixed-geometry correction checker. Every
coefficient is rounded outward, including uncertain input controls. All material
segments, horizon formulas, value/slope/curvature jumps and spatial chain
derivatives remain. Moving-angle tangents and orders above three are refused.
The two-entry coefficient cache stores immutable endpoints, detects changed
controls and returns fresh jets. The separate smooth-base integral and native
renderer are unchanged.</p>
<p>32 new analytical, cache and refusal controls pass after 32 expected
test-first failures; 288 affected reference controls also pass. The same three
saved D329 regions were then inspected in both integration directions. In the
current direction, total correction widths are 4.40, 4.16 and 3.43 times tighter.
Ordinary derivative contributions improve 4.63, 4.30 and 3.69 times; boundary
contributions also improve. All 233 inputs and eight witnesses remain.</p>
<p>One profiled diagnostic took 2.40 seconds compared with the saved 2.98-second
baseline. This small, instrumented measurement is not whole-piece throughput.
The benefit across the complete original piece is still unmeasured. A fresh,
frozen full-trial preview is prepared for explicit approval, with unchanged
accuracy, cell, time and memory limits and a stop at the first failure.</p>
<p>All seven earlier original campaigns and their failures remain historical
evidence. The last complete cover of the failing piece was 4.13 times too wide;
that result has not been replaced by this local diagnostic. M1 and the final
original-mesh SIGGRAPH-quality 3D showcase remain open. Existing accepted 2D
demonstrations remain available.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>The earlier blocker passes; the next region reaches the deadline</h2>
<p>The approved centered-cubic trial advances beyond the previously failing
piece. All accuracy and resource requirements remain unchanged.</p>
<table><thead><tr><th>Visible piece</th><th>Outcome</th><th>Cells</th>
<th>Reference seconds</th></tr></thead><tbody>
<tr><td>First (index 0)</td><td>Passed</td><td>57</td><td>19.09</td></tr>
<tr><td>Second (index 1)</td><td>Passed: earlier blocker cleared</td><td>223</td><td>81.40</td></tr>
<tr><td>Third (index 2)</td><td>90-second deadline</td><td>222 attempted</td><td>90.31</td></tr>
</tbody></table>
<p>The trial stops at that first failure; ten pieces are not started. All 502
cells and 190.79 reference seconds remain charged. The small cooperative stop
overshoot is failure latency, not an increased allowance.</p>
<p>The third piece's last complete cover contains 111 leaves. Its matte
uncertainty interval is 1.4898 times wider than allowed. This is a new region,
so it must not be compared directly with the prior failed piece's 4.1265 ratio.
These bounds measure what the independent checker has established, not observed
image error or an accuracy concession.</p>
<p>The ordinary derivative remainder accounts for 72.20 percent of correction
uncertainty; boundary terms account for 27.80 percent. The separate smooth-base
bound also contributes. Relative to the entire allowed radius, the contributions
are 99.38 percent ordinary, 38.26 percent boundary and 11.35 percent base. Their
sum still exceeds the allowance; none may be discarded.</p>
<p>Next inspect ordinary derivative dependence and cost on the new region's
saved subregions, with both integration directions and independent exact
controls. All eight campaigns and their failures remain preserved. No further
full trial is approved. The native renderer and wheel are unchanged. Whole-pixel
accuracy, moving gradients, frames, remaining scale cases and the final actual
original-mesh SIGGRAPH-quality 3D showcase remain open.</p>
"""


CHAPTERS[0]['body']+=r"""

<h2>Why the next piece needs a segment-aware bound</h2>
<p>The earlier blocker now passes, but the next piece reached its 90-second
reference deadline. Its last complete uncertainty interval is 1.49 times the
allowed width. This means the independent checker has not yet established the
required accuracy; it does not measure an error in the rendered image.</p>
<p>We inspected the three largest saved regions in both directions. The current
direction is already tighter in all three. Two bounded diagnostic passes used
7.59 seconds and 642 correction inputs against a combined 20-second, 2048-input
allowance. Preparation was measured separately. The original returned bounds,
components and crossing witnesses are unchanged.</p>
<p>The ordinary remainder along the conditioned lines dominates. On the thirty
half-lines in the current direction, its nonzero third derivative comes from
the view-angle material term; the light and mixed terms are exactly zero.
Twenty-one half-lines still span two material spline segments. Splitting at
the midpoint therefore leaves much of the same loose interval dependency.</p>
<p>The hypothetical total correction-width improvements are only 1.20, 1.02 and
1.04 times, with extra derivative evaluations. These are local bound estimates,
not production speed measurements. Uniform halving has not been adopted.</p>
<p>The next repair to qualify is a signed remainder bound for each actual
material segment, using certified crossing intervals. It must retain the
signed kernel mass on every segment, uncertain endpoint contributions and all
existing value, slope and curvature jumps. Exact independent piecewise-cubic
and log-material controls must pass before adoption, with a conservative
fallback when crossings are ambiguous or exceed the existing limits.</p>
<p>All eight full campaigns and their failures remain preserved. No new full
trial was launched and no mathematical repair was adopted in this investigation.
The installed renderer, accuracy requirements, resource limits and accepted
2D demonstrations are unchanged. Original-mesh accuracy, complete frames and
the final SIGGRAPH-quality 3D showcase remain open.</p>
"""


CHAPTERS[0]['body']+=r"""

<h2>A qualified segment-aware uncertainty calculation</h2>
<p>The independent checker now splits conditioned lines at the endpoints of
certified material-boundary intervals. It includes the uncertain boundary zones
and retains each segment's signed kernel contribution. Only a constant shared
over the complete line is cancelled. All existing value, slope and curvature
jumps, integrated terms and smooth-base contributions remain.</p>
<p>Branch selection requires a crossing guaranteed on every represented line.
If a crossing might be absent, every candidate branch remains. Nonmonotone
crossings retain the old fallback. More than sixteen slabs also triggers the
old bound before extra derivative evaluations. A contradictory new bound is
refused rather than silently accepted.</p>
<p>43 new analytical and integration checks and 280 affected reference checks
pass. Tests use closed cubic and log-material integrals, curved coordinates and
their Jacobians, signed physical scaling, uncertain endpoints and refusal cases.
Historical test-first failures, two reference-precision assertion failures and
the optional-crossing regression failure remain visible. The precision checks
now use separate 100-digit algebra; no acceptance threshold was loosened.</p>
<p>On the same three saved regions, the current-direction correction intervals
are <strong>2.48, 2.67 and 8.58 times tighter</strong>. Integrated bounds and all
jump bounds are exactly unchanged. The diagnostic used 387 inputs and 5.60
seconds, compared with 291 inputs and 2.98 seconds previously; all twelve
witnesses remain. This is an instrumented local measurement. Whether fewer
refinements offset the extra per-evaluation cost over a complete piece remains
unmeasured.</p>
<p>The method is connected to the original-material reference producer. The
installed renderer and wheel are unchanged. A fresh frozen full-trial preview
awaits explicit approval: at most thirteen serial pieces, unchanged accuracy
and resource limits, no old certificate reuse and a stop at the first failure.
All eight historical campaigns remain terminal. M1 and the final original-mesh
SIGGRAPH-quality 3D showcase remain open; accepted 2D demonstrations remain
available.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>Tighter local bounds have not yet paid for their evaluation cost</h2>
<p>The approved segmented-remainder trial stops at the second piece. The
independent calculation remains qualified on its analytical controls, but the
complete workload has not met the unchanged accuracy and resource requirements.</p>
<table><thead><tr><th>Visible piece</th><th>Outcome</th><th>Cells</th>
<th>Reference seconds</th></tr></thead><tbody>
<tr><td>First (index 0)</td><td>Passed</td><td>47</td><td>24.21</td></tr>
<tr><td>Second (index 1)</td><td>90-second deadline</td><td>158 attempted</td><td>90.16</td></tr>
</tbody></table>
<p>Eleven pieces were not started. All 205 cells and 114.37 reference seconds
remain charged. The small cooperative stop overshoot is failure latency, not
an increased allowance. The earlier trial passed this same second piece in
81.40 seconds; the added work per evaluation regresses whole-trial progress.</p>
<p>The last complete cover has 79 leaves and remains 1.3258 times wider than
allowed. Boundary terms now account for 78.08 percent of correction uncertainty,
and ordinary derivative remainder for 21.92 percent. Against the complete
allowed radius, the charges are 89.78 percent boundary, 25.21 percent ordinary
and 17.58 percent smooth base. These are conservative bounds on what the checker
has established, not measured rendering error.</p>
<p>The earlier 2.48, 2.67 and 8.58 times tighter local bounds concerned the third
piece, which this trial never reached. The adaptive covers also differ between
trials, so their cell counts are not matched-box accuracy comparisons. Fewer
cells alone do not establish lower total cost.</p>
<p>Next inspect the saved second-piece regions and cost in both directions.
Compare the segmented and conservative paths on identical boxes before choosing
selective application or exact reuse. Keep every boundary and base contribution,
and qualify any repair with independent analytical controls. No unchanged retry,
new full trial or higher limit is approved.</p>
<p>All nine campaigns and their failures are preserved. Native renderer and
wheel bytes remain unchanged. M1, original accuracy, moving gradients, complete
frames, remaining scale cases and the final original-mesh SIGGRAPH-quality 3D
showcase remain open. Accepted 2D demonstrations remain available.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>Reuse exact work without claiming an unmeasured speedup</h2>
<p>A matched investigation of the three largest saved second-piece regions
finds 39 repeated third-derivative inputs. A cache limited to sixteen entries
within one axis calculation now reuses those exact values. Every coefficient
and endpoint belongs to the identity; returned objects are independent, and
cancellation is checked even on cache hits.</p>
<p>The repair reduces correction input calls from 426 to 387 and third-order
calls from 218 to 179. Every saved numerical bound across both directions and
all three regions is exactly unchanged. Twenty new independent controls,
280 affected regressions and 43 segmented-reference controls pass.</p>
<p>This establishes less repeated work, not a measured speedup. The earlier
segmented measurement followed a conservative run with warmed kernel caches;
the repaired run started a fresh worker. Its 6.21 seconds cannot be compared
fairly with the earlier 5.82 seconds. Both were instrumented. Together the
diagnostics used 15.63 of 20 seconds and 1135 of 2048 permitted inputs.</p>
<p>Next compare cold-cache costs under the same conditions and inspect exact
kernel-work reuse before proposing another full trial. All previous failures
remain available. Accuracy thresholds, resource limits and the installed
renderer are unchanged. M1 and the final original-mesh 3D showcase remain open.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>A controlled comparison finds a modest local improvement</h2>
<p>After the PC crash, the dashboard database passed its integrity check and
all 1056 runs and 31751 test results were preserved before restarting.</p>
<p>We now compare uncached, reuse, reuse, uncached with empty kernel and material
caches before every pass. The same three saved regions and both directions
produce exactly identical bounds. Mean local time falls from 2.405 to 2.313
seconds, about 3.8 percent. This is two observations per path in one worker;
it does not predict complete-piece runtime or qualify the original image.</p>
<p>The four passes consume 9.45 of 20 allowed seconds and 1626 of 2048 inputs.
Each pass still incurs 151 local kernel-range cache misses. Of these, 104 occur
in third-order work, where the helper computes three orders although the caller
selects only the third. The 152 complete weighted-kernel keys are all distinct,
so another cache for complete results would not help this sample.</p>
<p>The next candidate is to compute only the requested kernel order while
retaining shared all-order callers and existing cache capacity. This remains
unimplemented: independent polynomial and endpoint tests must establish it
before another local cost comparison or a separately approved full trial.
All failed campaigns, accuracy criteria and resource limits remain unchanged.
M1 and the final original-mesh 3D showcase remain unfinished.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>The checker now computes only the kernel bounds it needs</h2>
<p>The independent accuracy checker was calculating three polynomial orders even
when a caller needed just one. We now fill those bounds on demand, keeping a
single cache of128positions. Each position can retain the same three orders;
there is no extra cache or larger resource allowance.</p>
<p>The first attempt split entries by order and increased cache misses from151
to255. We preserved that result and repaired the storage design. The final
version restores151misses and152hits, while reducing computed polynomial orders
from453to323 on the saved regions. All numerical bounds match exactly.</p>
<p>Final qualification has51 focused controls and343 regressions passing.
Independent100-digit formulas, uncertain Gauss-node boundaries, mutation
isolation, failed fills and capacity are covered. All52 earlier failing test
results remain in history. The final saved-region diagnostic took2.277seconds;
one observation cannot establish complete-piece speed. All five diagnostic
passes used12.442of20seconds and1935of2048inputs.</p>
<p>Original-mesh accuracy is still open. The next gate is a fresh, separately
approved13piece certificate campaign with unchanged limits, stopping at its first
failure. Then the native CPU and CUDA image and albedo derivatives must agree
with that independent certificate. This checker qualification is not the final
SIGGRAPH-quality3D showcase and does not close M1.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>The approved trial passes two pieces and stops at the third</h2>
<p>The frozen candidate was explicitly approved and executed once. The first
piece passes in21.13reference seconds. The previously failing second piece now
passes in88.15seconds, within the unchanged90second limit. The third piece
reaches that deadline before its error bound is small enough; the remaining
ten pieces were not started. This trial is terminal, not awaiting approval.</p>
<p>The third piece's last complete bound is2.391times the allowed width.
About31.9percent comes from the base enclosure,26.3percent from the smooth
remainder and41.8percent from boundary uncertainty. None can be dropped.
These are conservative error bounds, not measured errors in rendered pixels.</p>
<p>All364attempted cells,199.48reference seconds and203.80worker seconds are
charged, including failed work. Peak sampled host memory is113340416bytes.
The next investigation must separate these three uncertainty sources from
their computation costs on the saved regions. A repaired candidate needs its
own qualification and newly approved trial. There is still no complete pixel
certificate, no new native accuracy admission and no final3Dshowcase acceptance.
M1 remains unfinished; earlier failed trials stay visible in history.</p>
"""


CHAPTERS[0]['body']+=r"""
<h2>A tighter expression for the same smooth material curve</h2>
<p>The third-piece investigation found no consistently better integration
direction. Exact arithmetic work reductions preserve every saved bound and
reduce the instrumented local time from6.28to5.72seconds.107new arithmetic
controls include extremely small live tangents; none are discarded.</p>
<p>A stronger improvement comes from expressing the smooth cubic material curve
around its center. This is the same mathematical curve, independently qualified
through the tenth derivative needed by the integration error bound.124controls
cover independent110digit series, uncertain controls, horizon behavior, the
actual producer binding and a closed nonlinear integral.</p>
<p>On three saved regions the base uncertainty falls from roughly10^-14to10^-24,
while that portion runs in about half the time. Boundary corrections still
apply. We keep the same shared two-entry cache, precision, quadrature rule,
tolerances and resource limits. All diagnostic work, including the repaired
harness failure, stays below20seconds and2048counted calls.</p>
<p>These are local qualification results. A fresh approved13piece campaign must
still certify the complete selected pixel before native CPU/CUDA accuracy can
be admitted. All historical failures remain; M1 and the final3Dshowcase are open.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Separating accuracy bounds from refinement guidance</h2>
<p>The centered-base trial stopped at its second piece after the90second limit.
One piece passed and eleven were not started. Its tighter local bounds did not
guarantee cheaper subdivision. The failed trial and all its costs remain recorded.</p>
<p>Three saved regions show the existing correction hint chose the better split
direction. A new signed remainder keeps the nonzero kernel contribution of each
half interval, using the same four derivative boxes and retaining all jumps.
Independent exact-integral controls pass, but local gains are modest.</p>
<p>We now separate the guide that selects work from the bound that proves accuracy.
Historical Bernstein derivative scales may guide refinement; centered bounds plus
every correction term still decide acceptance. A weak guide cannot displace the
actual error priority. Zero or tiny guidance cannot falsely qualify a result.</p>
<p>Guidance adds two derivative evaluations per cell. Six such evaluations on the
three saved regions cost about0.26seconds including preparation. All diagnostic
work remains within20seconds/2048calls. A fresh explicitly approved trial must
determine whether its subdivision choices repay that cost. No whole-pixel or
native accuracy admission follows yet; M1 and the final3Dshowcase remain open.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>A near miss and exact arithmetic work reduction</h2>
<p>The guided trial passed its first two pieces in19.1and78.4reference seconds.
The third stopped at90seconds; its last complete uncertainty was5.8percent above
the allowed width. It is still a failure. Ten later pieces were not started.</p>
<p>The next repair keeps the same numbers while doing less bookkeeping. Each jet
recurrence reuses its identical denominator reciprocal, and interval arithmetic
reads existing operands without temporary copies. Tests protect exact outward
endpoints, tiny live tangents and mutation isolation.</p>
<p>The final saved correction diagnostic is about6percent faster. A controlled
old-new-new-old comparison of high-order base and guidance fields is about12percent
faster after copy removal. An earlier denominator-only comparison was slower and
remains in history. These local timings cannot guarantee a full trial pass.</p>
<p>All diagnostic work stays within20seconds/2048calls. A fresh explicitly approved
trial is still required; neither the original pixel nor the final3Dshowcase is
accepted yet. M1 remains unfinished, with accuracy and resource limits unchanged.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Three pieces certified; choosing the fourth piece’s integration order</h2>
<p>The fresh arithmetic trial certified its first three pieces. The previously
failing third piece passed in88.8reference seconds. The fourth reached its
90second limit with a bound about4.1times too wide; nine later pieces were not
started. All results remain in history.</p>
<p>A two-dimensional integral can be computed in either coordinate order.
The exact answer is the same, but conservative boundary estimates can differ.
Both orders now pass analytical controls. For the fourth piece only, the checker
uses the alternate order; the other twelve keep their previous order.</p>
<p>On three saved regions the alternate full correction bounds are about4.2,
5.8and1.4times tighter. The complete diagnostic costs8.0seconds/972inputs under
the20second/2048input limit. These small-region results do not certify the whole
piece. A fresh approved thirteen-piece trial is required. Accuracy, resource
limits and the native renderer are unchanged; M1 and the final3Dshowcase remain open.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>The fourth piece still needs less checking work</h2>
<p>The alternate-order trial passed three pieces again. The fourth reached its
time limit with uncertainty3.41times the allowed width, improved from4.07but
still a failure. Nine later pieces were not started. This does not show that
the rendered pixel is wrong; its independent certificate is incomplete.</p>
<p>Both midpoint split directions were tested on three saved regions. The
scheduler already chose the better direction in all three, so it stays unchanged.
The next repair removes repeated copies of fresh arithmetic results and avoids
smooth calculations that multiply a provably zero correction. Tiny nonzero
derivatives, outward rounding, input isolation and denominator checks remain.</p>
<p>The saved bounds are exactly unchanged. Arithmetic-only local evaluation
was about15% faster across three regions; one final region with both repairs
was about21% faster. Those observations do not predict a whole-trial pass.
All diagnostic work totals8.07seconds/2040inputs within20seconds/2048inputs.</p>
<p>A fresh approved thirteen-piece trial remains necessary. Accuracy, resource
limits and the native renderer are unchanged. M1 and its final3Dshowcase remain open.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Less setup work, the same outward arithmetic</h2>
<p>The latest approved trial passed three pieces. The fourth reached its deadline
after 209 attempted cells; its last complete uncertainty bound was 1.78 times the
allowed width, improved from 3.41. Nine later pieces were not started. This remains
an incomplete accuracy certificate, not a demonstrated error in the rendered image.</p>
<p>The next repair sets up a rounding context once for a complete coefficient
multiplication. Every product and addition still rounds outward separately, in
the original order. Independent rational polynomial checks, exact endpoint
comparisons and material integration controls pass. The native renderer and
all accuracy and resource limits remain unchanged.</p>
<p>No saved mesh-region diagnostic was repeated. Synthetic timings cannot predict
a whole-piece pass, and the cell cap can still stop the trial even if it runs
faster. A newly approved thirteen-piece trial is required before any original-pixel
accuracy claim. M1 and the final original-mesh 3D showcase remain unfinished.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Choose the integration direction for each region</h2>
<p>The latest trial still stopped at the fourth piece: 237 attempted cells and
the time limit, with the last complete uncertainty bound 1.31 times the allowed
width. Three pieces passed and nine remain unstarted. The 256-cell ceiling is
also close, so faster arithmetic alone may not finish the proof.</p>
<p>Both integration directions were examined on three newly saved regions.
The current direction remains better in one; the other direction gives tighter
bounds in two. The checker now estimates both continuous contributions and
chooses a direction separately for each region of this fourth piece. It then
computes the complete selected formula, including every remaining error term.</p>
<p>The three local correction widths become 100%, 52% and 21% of their previous
widths. Local checking time increases about 16%. Whether fewer subdivisions
outweigh that extra work remains a question for the next approved full trial.
Independent mathematical controls pass, but these local results do not certify
the whole original pixel. All accuracy and resource limits remain unchanged;
M1 and the final original-mesh showcase are still unfinished.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Bound the kernel area more closely</h2>
<p>The previous trial still stopped at the fourth piece: three passed, nine
unstarted, and a last complete uncertainty bound1.55times the allowance.
Choosing both directions on every region did not repay its extra cost.</p>
<p>The next checker computes small kernel polynomials directly. It also bounds
their area using polynomial coefficients, rather than only a rectangle as tall
as their peak. This is an analytical upper bound, not sampled convergence.</p>
<p>On three saved regions the complete correction widths fall to about77%,86%
and79%of their previous widths. Every preparation input is charged. Independent
exact-integral controls pass. The full trial must still establish whether the
tighter bounds repay their cost; the original pixel and M1 remain unfinished.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Spend checking time on certified uncertainty</h2>
<p>The tighter kernel-area bound reduced work on the first three pieces, but the
fourth still reached its deadline with1.32times the permitted uncertainty.
An additional boundary-location refinement was correct yet offered only about
0.4–1.6%tighter bounds on saved regions while adding cost. It remains disabled.</p>
<p>The fourth piece already computes a complete certified error bound. The next
trial uses that bound to prioritize subdivisions and omits two extra derivative
calculations used only as a work-order heuristic. The other twelve pieces keep
their previous strategy. Every numerical acceptance check remains in place.</p>
<p>This is a qualified scheduling experiment, not a completed accuracy claim.
The original-mesh showcase and M1 remain open until their full gates pass.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Separate checker work from timing variation</h2>
<p>A previously passing second piece reached its deadline in the latest trial.
The bounded old/current comparison produced identical numerical results and did
not reproduce the slowdown. Its cause remains unresolved. The next trial records
CPU time alongside wall time; wall-time limits remain authoritative.</p>
<p>The investigation also found repeated angular-table lookups. Resolving the
same live table once per call removes overhead while retaining every interval
endpoint, knot comparison and table replacement. Tests pass, but modest local
timing gains do not prove that the complete original pixel will be certified.</p>
<p>All earlier failures remain available. The final original-mesh showcase and
M1 are still unfinished; no tolerance or resource quota has been relaxed.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Reuse polynomial constants, keep the complete proof</h2>
<p>The cheaper refinement schedule did not succeed: the fourth piece again
reached its deadline, with a wider remaining uncertainty. Its CPU time nearly
matched elapsed time. We restore the earlier guidance that chose better regions.</p>
<p>The next checker caches only the fixed coefficients of eighteen small kernel
polynomials. Every spatial region still gets its own complete outward bound.
Independent rational, uncertain-coefficient and mutation checks pass. Local times
improved unevenly; only the fresh full trial can establish whole-piece success.</p>
<p>No rendering accuracy, memory quota or deadline has changed. All failed trials
remain visible, and the final original-mesh showcase remains unqualified.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Redistribute uncertainty without increasing it</h2>
<p>The fourth piece still exceeded its assigned share at the deadline. Area alone
need not distribute checking effort well. The next trial gives that difficult
piece a larger share of the same fixed uncertainty pool and requires tighter
bounds from every other piece.</p>
<p>Exact rational normalization and downward rounding keep each complete basis
channel below5×10<sup>−13</sup>radius. The final pixel still must satisfy
10<sup>−12</sup>radius. Independent tests reject any increase to the pool,
even with matching hashes. The renderer's acceptance thresholds are unchanged.</p>
<p>Every piece needs a fresh certificate under its new share. Old failures stay
failed, old passes are not reused, and the same time, memory and cell limits apply.
This is a qualified allocation strategy, not an original-pixel accuracy result.</p>
"""

CHAPTERS[0]['body']+=r"""
<h2>Follow shading boundaries in the accuracy proof</h2>
<p>All 13 reference pieces pass. Opaque122 repairs radial sampling: original pixel RGB/albedo endpoints and CPU/CUDA bytes pass. Spare capacity preserves the horizon-motion control. Limits and failures remain. Moving fields, full frames and final showcase are open.</p>
"""
