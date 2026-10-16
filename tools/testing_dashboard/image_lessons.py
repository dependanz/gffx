"""Original image-mathematics chapters; root textbook owns reading-order registration."""


def _chapter(slug, title, prerequisites, body):
    return dict(id='learn/'+slug, title=title,
                prerequisites=['learn/'+item for item in prerequisites],
                evidence=['image/contract'], source='src/gffx/torch/image.py', body=body)


CHAPTERS = (
_chapter('image-compositing', 'Images · Linear light, alpha and compositing',
         ['vectors','autodiff','numerics'], r'''
<h2>Colour and coverage are different quantities</h2>
<p>An RGB triple describes colour. Alpha, a dimensionless number between zero and one,
describes how much a layer contributes. Straight RGBA stores colour c and alpha a separately;
premultiplied RGBA stores p=a c and a. These are representations of the same contribution,
not two different artistic blend modes. At zero alpha, colour is invisible and cannot be
recovered by division. gffx chooses canonical zero colour and zero colour derivatives there.</p>
@@p=a c@@Premultiplied colour p includes the layer's alpha a; each RGB channel obeys this equation.@@
<h2>The over operation</h2>
<p>Place source s over destination d. Source contributes its premultiplied colour; only the
uncovered fraction of the destination remains. Layer order matters. Alpha over is not an
arithmetic average, and an opaque top layer completely hides what is below it.</p>
@@p_o=p_s+(1-a_s)p_d@@The output colour combines source contribution and uncovered destination.@@
@@a_o=a_s+(1-a_s)a_d@@Output alpha obeys the same coverage accounting.@@
<p>Worked example: half-opacity red has premultiplied RGBA (0.5,0,0,0.5). Over opaque blue
(0,0,1,1), the result is (0.5,0,0.5,1). The test named half_red_over_blue_closed_form checks
these exact dyadic values. For fixed source alpha, the derivative to destination colour is
1-a_s: fully covering a pixel removes the derivative to what lies behind it.</p>
<h2>Why coloured fringes appear</h2>
<p>Imagine an invisible red texel beside opaque blue. Averaging their straight RGB creates
purple, even though no red should be visible. Premultiply first: the invisible texel becomes
black with zero alpha. Average premultiplied colour and alpha together, then unpremultiply;
the answer is blue at half alpha. The transparent-edge warp fixture checks this explicitly.</p>
<h2>Linear light is not an encoded display value</h2>
<p>sRGB encoding is nonlinear. Averaging encoded black and white gives 0.5 encoded, whereas
averaging their linear-light intensities gives 0.5 linear, approximately 0.735 encoded. The
latter represents half the light. Decode source RGB before linear compositing, never decode
alpha, and encode only at an explicit output boundary. Negative and HDR working values need
a declared policy; gffx retains them rather than silently clipping to a display range.</p>
<p>The transfer functions are piecewise. Their low branches are linear, and gffx takes that
branch at the exact threshold. Rounded sRGB constants introduce a tiny boundary mismatch;
central-difference checks at that threshold do not establish a smooth derivative.</p>
<h2>Masks and exposure</h2>
<p>A source-space mask m and opacity o combine with existing alpha: a'=m o a. Premultiplied
colour receives the same factors. Exposure is measured in stops: +1 doubles linear RGB and
-1 halves it. Additive brightness b in straight colour becomes b a in premultiplied colour,
which keeps transparent pixels transparent.</p>
@@p_{out}=2^e p+b a@@Exposure e is in stops; brightness b is in linear colour units. Alpha is unchanged.@@
<details><summary>Try it yourself: combine coverage factors</summary><p>A pixel has alpha 0.8,
mask 0.5 and layer opacity 0.25. What alpha reaches the compositor?</p><h3>Answer</h3>
<p>0.8 times 0.5 times 0.25 = 0.1. Each premultiplied RGB component is multiplied by 0.125.</p></details>
<h2>gffx boundary: what these tests do not prove</h2>
<p>Correct alpha mathematics is not full colour management, physically accurate transparency,
display calibration or an editor's undo system. Unpremultiplication near zero alpha is
ill-conditioned. Discrete layer order is not differentiable. Read the current exact-artifact
test results before making CPU/CUDA, performance or compilation claims.</p>
'''),
_chapter('image-sampling', 'Images · Inverse mapping, filtering and fitting',
         ['transforms','textures','autodiff','image-compositing'], r'''
<h2>Ask where each output pixel came from</h2>
<p>Forward mapping a source grid can leave holes in a destination image. Inverse mapping
visits every destination pixel centre and asks which source coordinate to sample. gffx pixel
centres are (column,row), with y downward. A source-to-destination affine transform A and
translation t therefore require solving s=A inverse times (d-t). A singular A cannot be
inverted; near-singular transforms can have very large derivatives.</p>
@@d=A s+t@@Source point s is mapped to destination d by the affine transform.@@
<p>Translation and pivot are in pixels; rotation is in radians, clockwise on the y-down
display. Scale is dimensionless. To rotate about a pivot, subtract that pivot before the
linear map and add it afterward. If a source pixel at x=0 is translated one pixel right,
destination x=1 reads source x=0. The integer-translation test makes that direction explicit.</p>
<p>For fitting, each sampled pixel contributes a derivative to the same six affine matrix
entries. The inverse transform sends an image-coordinate derivative backward through the
inverse transpose of its linear map, with a minus sign for translation. Multiplying that
translation derivative by the sampled source position gives the linear-map derivative.
These contributions must then be added across pixels.</p>
<p>The addition order matters in finite precision: two mathematically equivalent parallel
reductions can differ in their last bit. gffx's candidate inverse-map backward pairs adjacent
contributions in a fixed tree and carries an unpaired last contribution unchanged. That makes
the intended arithmetic order explicit without building a huge per-pixel matrix Jacobian.
Independent finite differences still check the mathematics; exact CPU/GPU comparisons check
the implementation separately. A visually identical result is not a substitute for either.</p>
<h2>Pixels are samples, not continuous squares</h2>
<p>Bilinear interpolation combines four neighbouring pixel centres. In one dimension, with
fractional coordinate u between two samples, use (1-u)c0+u c1. Its derivative with respect
to u is c1-c0. A constant region supplies no translation signal. At a grid knot the selected
cell changes; the implementation publishes a branch convention, not a claim of smoothness.</p>
@@c=(1-u)c_0+u c_1@@Interpolation is a weighted sum; its coordinate derivative is the local colour difference.@@
<p>Nearest sampling makes a discrete choice and has zero coordinate derivative by convention.
At a half tie, gffx picks the increasing integer index. Transparent borders supply zero
premultiplied RGBA; clamped borders repeat edge values. These choices change both pixels and
derivatives, so an optimizer and an export path must agree on them.</p>
<h2>Why downscaling can shimmer</h2>
<p>A destination pixel may cover many source samples. Merely choosing one bilinear sample
can miss alternating stripes or fine patterns. That is aliasing. The first affine warp
does not secretly add mipmapping or EWA filtering. Supersampling and explicit box reconstruction
provide a declared alternative, not perfect frequency reconstruction.</p>
<p>Worked example: a 2-by-2 block with values 0,1,2,3 resolves to 1.5. Its output
derivative to each input is one quarter. gffx sums each block in a fixed row-major order,
preserves channels independently, and rejects dimensions that are not exactly divisible.</p>
<h2>A Gaussian has a width and a support</h2>
<p>Sigma is the Gaussian width in pixels. Radius is the integer support: how many neighbours
are included on each side. They are not interchangeable. Hold the radius fixed while
optimizing sigma; choosing ceil(3 sigma) silently changes the discrete kernel as sigma moves.</p>
@@w_i=\frac{\exp(-i^2/(2\sigma^2))}{\sum_j\exp(-j^2/(2\sigma^2))}@@Normalize weights over the explicitly chosen finite support.@@
<p>A separable Gaussian applies a horizontal one-dimensional filter then a vertical filter.
The impulse fixture compares this with a separate two-dimensional scalar convolution.
Transparent padding loses mass outside the image; clamped padding preserves a constant image.
Blur acts on premultiplied colour and alpha together to avoid transparent-edge fringes.</p>
<details><summary>Try it yourself: find the translation signal</summary><p>Samples have values
0.2 and 0.8. What is their interpolation at u=0.25, and its derivative with respect to u?</p>
<h3>Answer</h3><p>The value is 0.75 times 0.2 plus 0.25 times 0.8 = 0.35.
The derivative is 0.8 minus 0.2 = 0.6. A negative image gradient would push u upward.</p></details>
<h2>gffx boundary: fitting requires a full connected chain</h2>
<p>A useful fitting recipe combines exposure, mask/opacity, blur, affine sampling, compositing
and reconstruction. Isolated derivative tests are necessary but not sufficient: the full-chain
fixture checks nine fitting parameters through every stage: translation, rotation, two scales, opacity, exposure, brightness and blur width. Even a correct gradient does
not imply a unique inverse problem. Repeated patterns, hidden layers and constant regions can
produce ambiguity. A recovered image may match while its parameters differ.</p>
<p>First-order smooth-region tests do not prove higher derivatives, arbitrary convergence,
full-frame speed or real Compositron revision safety. Cancellation, locked properties and
undo remain future consumer responsibilities. M1 establishes gffx-side mathematical and
resource readiness to start those projects; it does not implement their document adapters.</p>
'''),
)

CHAPTERS[0]['body'] += """
<h2>A saved fit is evidence with an identity</h2>
<p>The nine-parameter example adjusts translation, rotation, scale, opacity,
exposure, brightness and Gaussian width against an independently computed
synthetic target. The original two solves retain their artifact,79 closure
counts, locked-input checks and repeatability evidence. Reusing those parameters
on a newer wheel requires fresh evaluation of the independent objective; it does
not turn the old optimization into a new solve.</p>
<p>The current replay gives initial MSE0.00013223659472628245 and final
MSE2.5634514115042626e-10, within the original one-percent requirement. Independent
stage and derivative tests and exact CPU/CUDA comparisons remain separate gates.
The image matches closely although correlated physical parameters are not
uniquely identified. The recovery board keeps the same error scale before and
after fitting rather than amplifying each image independently.</p>
<p>The bounded CI-01 through CI-07 supply prepares Compositron development.
It does not implement a document editor, image decoding, undo, job cancellation
or an arbitrary optimization basin. The CHROMA poster and edge/sampling studies
use actual public image operations. Their finite visual acceptance does not
close the original-mesh3D renderer or all of M1.</p>
"""
