"""Original deformation chapter; examples are mathematics, not run-result claims."""
import math


def worked_examples():
    """Scalar arithmetic for the chapter; deliberately independent of production kernels."""
    template = (1., 2., 3.)
    displacement = (2., 0., -1.)
    triangle = ((0., 0., 0.), (3., 0., 0.), (0., 3., 0.))
    return {
        "unit_corrective": tuple(a + b for a, b in zip(template, displacement)),
        "negative_corrective": tuple(a - 2*b for a, b in zip(template, displacement)),
        "translated_child": tuple(a+b for a, b in zip((0., 3., 0.), (2., 0., 0.))),
        "triangle_centroid": tuple(sum(point[c] for point in triangle)/3 for c in range(3)),
        "quarter_turn_x": (math.cos(math.pi/2), math.sin(math.pi/2), 0.),
        "corrective_bytes": 5023*400*3*8,
    }


CHAPTER = dict(
    id="learn/animation-deformation",
    title="Deformation · from corrective shapes to a moving skeleton",
    prerequisites=["learn/vectors", "learn/transforms", "learn/autodiff", "learn/barycentric"],
    evidence=["animation/contract"],
    source="native/core/animation.c",
    body=r'''
<h2>A character is a composition, not one magic operation</h2>
<p>A mesh starts with vertex positions. Corrective displacements change its shape, joint rotations change its pose, a hierarchy expresses which joints move with other joints, and skin weights distribute those transformations across the surface. Landmarks finally turn selected points on that surface into quantities a fitting loss can compare. gffx provides these mathematical building blocks. A model supplies its own assets, joint offsets, corrective basis and bind-pose conventions.</p>
<p>Coordinates have the caller's length unit. Rotation vectors use radians. Coefficients and skinning weights are dimensionless. A displacement is a vector, not an absolute replacement position.</p>
<h2>1. Blend shapes are linear displacements</h2>
@@v_b=t+\sum_{k} w_{bk}d_k@@The template t and corrective displacements d are shared; frame b supplies its coefficients w.@@
<p>For one vertex, take t=(1,2,3) and d=(2,0,-1). Weight one produces (3,2,2); weight minus two produces (-3,2,5). The template is still present in both answers. Weight zero returns the template. There is no reason to force these coefficients to be positive: negative amounts simply move in the opposite direction along the corrective.</p>
<p>A shared corrective receives gradient contributions from every frame that uses it. This is why backward is not just the forward loop in reverse: the implementation needs a well-defined reduction order and may avoid allocating a corrective gradient entirely when the corrective is fixed.</p>
<h2>2. Axis-angle rotation must work at zero</h2>
<p>A rotation vector r points along the rotation axis and its length is the angle. Let K be the skew matrix whose action is Kp=r cross p. It is zero when r is zero, without requiring us to divide by the vector's length.</p>
@@q=r_x^2+r_y^2+r_z^2@@The squared angle q avoids taking a square root on the small-angle branch.@@
@@R=I+A K+B K^2@@The coefficients A and B smoothly approach one and one half as the angle approaches zero.@@
@@A=1-q/6+q^2/120-q^3/5040+q^4/362880@@This small-angle polynomial approximates sin(theta)/theta without a division by zero.@@
@@B=1/2-q/24+q^2/720-q^3/40320+q^4/3628800@@This companion approximates the second Rodrigues coefficient.@@
<p>At exactly zero, R is the identity, but its derivative is not zero. An infinitesimal positive z rotation moves the x-axis toward the y-axis. Detaching the axis or replacing the zero-angle gradient with zero would make a model initialized at its neutral pose unable to begin learning a rotation correctly.</p>
<p>The implementation uses these polynomials when q is at most 2 to the power minus8. It compares both sides of that switch with a separately implemented matrix exponential. Rotation near pi is not a singularity of this forward formula; choosing a unique inverse rotation vector is a different problem, left to a later inverse-map operation.</p>
<h2>3. Forward kinematics keeps track of coordinate frames</h2>
@@W_j=W_p L_j@@A non-root joint's world transform is its parent's world transform times its local transform. A root simply has W=L.@@
<p>Think of a wrist attached to an elbow: the wrist's local movement is relative to the elbow's frame, so the elbow's movement affects it too. A stored parent list need not put parents first. The evaluator derives an order, rejects cycles, and permits several independent roots. It does not rely on recursion, so a deep hierarchy does not exhaust a call stack.</p>
<p>Worked example: a parent translates three units along y, and its child translates two units along x. With no rotations, the child's world translation is (2,3,0). Add a parent rotation and the child's local x direction rotates with it; multiplication order now matters.</p>
<h2>4. Skinning blends transformed points</h2>
@@y_v=\sum_{j} s_{vj} T_j h_v@@Here h is the homogeneous point with fourth component one; only the first three output components are used.@@
<p>A vertex attached wholly to one joint follows that joint exactly. With two normalized weights, it moves between two transformed versions of itself. This is linear blend skinning, not a simulation of flesh or elastic energy. Twisting can collapse volume: that is a property of this deformation model, not a promised physical effect.</p>
<p>Conventional skin weights are nonnegative and sum to one. The primitive nevertheless accepts any finite signed coefficients and does not normalize them: explicit normalization belongs to the caller, and unconstrained coefficients have ordinary derivatives. A row of all zeros therefore produces zero. The fourth matrix row does not trigger perspective division and has exactly zero derivative in this operation.</p>
<p>The caller normally constructs a skinning transform from the current world transform and an inverse bind transform. Forgetting that correction can apply the rest pose twice. gffx never guesses the model's bind convention.</p>
<h2>5. Landmarks live on triangles, not image pixels</h2>
@@p=b_0 v_0+b_1 v_1+b_2 v_2@@A selected triangle and three coefficients define a point for each frame.@@
<p>For vertices (0,0,0), (3,0,0) and (0,3,0), equal coefficients one third give the centroid (1,1,0). A face index is discrete, but vertex positions and coefficients are differentiable. No renderer or camera is needed to evaluate this point. Values outside the triangle are allowed as explicit linear extrapolation; coefficients are neither clipped nor silently renormalized.</p>
<h2>The complete gradient chain</h2>
<p>A loss on projected or 3D landmarks can flow backward through barycentric evaluation, skinning, forward kinematics and Rodrigues to pose parameters, while simultaneously flowing through blend shapes to the template, coefficients and even learned corrective displacements. Each primitive can be correct in isolation yet wired incorrectly in this chain; therefore the chain receives its own finite-difference test.</p>
<details><summary>Try it yourself: which inputs need gradients?</summary><p>A model's template and corrective basis are fixed. You fit only the per-frame coefficients. Must backward allocate gradients for the template and every corrective displacement?</p><h3>Answer</h3><p>No. It needs those values to compute coefficient derivatives, but their own derivative buffers are unnecessary. At 5023 vertices, 400 correctives, three coordinates and float64, one corrective tensor occupies 48,220,800 bytes; allocating an unused gradient of equal size can exceed a small device-memory budget.</p></details>
<details><summary>Try it yourself: what changes when the root moves?</summary><p>A forest contains one root with two descendants and a separate root. Which world transforms change when only the first root's local transform changes?</p><h3>Answer</h3><p>The first root and its descendants change. The separate root remains unchanged. A parent relationship expresses transformation dependence, not merely grouping in a user interface.</p></details>
<h2>gffx boundary</h2>
<p>These operations use native C11 mathematics behind thin PyTorch bindings. Inputs do not silently change precision, device or layout. Only first-order reverse-mode gradients are required here. Native CUDA bit-identity, compiler/export support, larger-model memory costs and real-time claims require their own evidence; a teaching example does not grant them. The linked animation contract and test pages show the current qualification state. No licensed character model is bundled, and authored motion is not physics.</p>
''')
