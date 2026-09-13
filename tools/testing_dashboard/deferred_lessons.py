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
