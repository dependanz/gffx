# Deterministic Gaussian weights - design for review

Status: proposed, 2026-09-10. No implementation or acceptance claim. This is a
gffx-only image-lane repair, not a generic exponential library or consumer feature.
Root owns adoption, shared registration/build changes and numerical-worker leases.

## Evidence and unchanged requirements

Installed source14 wheel `7f0b04abd9c5b3346c55758142789bf27631115b5cb37b1803bf70dca9850379`
passed CPU88, the focused odd-extent CUDA2 and44 broader CUDA cases before the first
float32 odd/batched blur failed. `build/image-20260909/source14-blur-staged1.json`
isolates the first discrepancy to `exp` with identical operands: values differ by
2.9802322387695312e-8, while matched-weight convolution values and source/weight VJPs
are exact. Literal .83f32 and the previously tested .23f32+.6 differ by one ULP.
The saved test/guide copies and XML retain both input bytes and all stage results.

Do not weaken CPU/CUDA byte identity, independent accuracy tolerances, the public
sigma/radius domain, finite/error rules, dependency policy or resource limits.
No CPU fallback, hidden device allocation, libm exp/pow acceptance shortcut or new
external dependency. Evidence is exact-artifact and device specific; a sampled
envelope is not exhaustive correct-rounding proof.

## Options and recommendation

| Option | Benefit | Remaining risk / cost |
|---|---|---|
| Internal bounded negative-exp helper; keep tensor tap construction and normalization | Smallest apparent numerical replacement; reusable within Gaussian construction | Leaves division, normalization and shared-sigma autograd reductions dependent on adapter evaluation choices. Exposes an artificial exp derivative boundary and still needs scaled derivatives near underflow. Many tiny dispatches or another wrapper would be required. |
| Native normalized Gaussian taps plus analytical sigma VJP | Owns one bounded mathematical operation, exact tap/reduction order and explicit first-order contract; one forward/backward dispatch per weight set | Three additive C functions, one private family and two adapter operations; more ABI validation and hardware tests. |

Recommend the second option, subject to root review. Keep the exp approximation a
private shared scalar detail of this family, not a public callable. Existing image
convolution and its fixed `_WeightedMultiply` spatial reductions remain unchanged.
Do not expand to arbitrary kernels, FFT blur, arbitrary exp or higher derivatives.

## Mathematical and tensor contract

For integer radius R in[0,32], taps i=-R,...,R and finite represented sigma in[.1,32]:

    a_i = i^2 / sigma^3
    q_i = exp(-0.5 * (i/sigma)^2)
    Z = sum_i q_i
    w_i = q_i / Z
    mu = sum_i w_i * a_i
    dw_i/dsigma = w_i * (a_i - mu)
    grad_sigma = sum_i incoming_i * dw_i/dsigma

These are mathematical definitions, not permission to compute derivatives from
already-rounded/underflowed output weights. Incoming cotangents are independent
finite signed values, not probabilities. Output weights are nonnegative and
symmetric; represented summation need only meet the existing accuracy contract,
not equal exactly1. The centre raw tap is exactly1, so Z>=1 and never needs an epsilon.
Do not clamp, renormalize again in the adapter, or detach tiny derivatives silently.

Sigma input is rank0 or rank1 `[B]`, dense f32/f64 on one device. Output is `[K]` or
`[B,K]`, K=2R+1. The VJP returns the original sigma shape. B=0 is legal and produces
empty outputs. Public image APIs retain scalar/shared or per-batch sigma semantics;
one shared sigma evaluates one tap set and receives the existing fixed spatial
weight cotangent before the native tap VJP. Per-batch sigma evaluates independent
sets. No new generic sigma broadcasting. Radius0 returns exactly[1] and first
derivative +0 after normal structural/domain validation; no exp work is needed.

Branch/domain bounds compare in the input dtype, including represented .1f32.
Compute f32 inputs in f64 internally, round output weights once to f32. For f32 VJP,
round each analytic tap derivative once to f32, then use prescribed f32 products and
fixed reduction of represented incoming cotangents. For f64 use f64 throughout.
This explicit VJP convention must be independently checked; it is not the derivative
of IEEE rounding, and need not reproduce the former PyTorch graph's last bits.

## Bounded scalar algorithm proposal

Use one host/device scalar implementation with strict FP, no contraction, gradual
underflow, and explicit round-to-nearest divisions. Never rely on Python wrapped
scalars or compiler fast-math. Reuse compatible controlled-division/bit helpers only
after review; do not entangle the colour-transfer API with Gaussian semantics.

1. Evaluate each nonnegative |i| once; mirror the same bits into negative taps.
   Ratios are at most320, so exponent x lies in[-51200,0]. Compute q for centre
   exactly1 and a for centre exactly0.
2. Candidate range reduction: choose integer n nearest to x/ln(2), with an explicit
   tie rule, and form r=(x-n*ln2_hi)-n*ln2_lo. Keep the split constants as generated
   hexadecimal literals with independently recorded Decimal source/error. The
   candidate domain is |r|<=ln(2)/2 plus a conservatively derived rounding margin.
3. Candidate exp(r): fixed degree18 Taylor/Horner polynomial with independently
   rounded coefficients, no FMA, no convergence-dependent loop. The analytic
   truncation remainder alone is bounded by exp(|r|)*|r|^19/19! (approximately
   2e-26 at the ideal interval endpoint); coefficient/evaluation/range-reduction
   error is separate and dominates. This is not a finished error bound.
4. Retain exp as `(mantissa,n)` until the required value or derivative has been
   normalized and scaled. Use a shared explicitly rounded binary scaling routine
   that handles normal/subnormal/tie-to-even cases; no host/GPU ldexp equivalence
   assumption without evidence. Large negative n is handled by bounded integer
   comparisons, not a loop of n multiplications.
5. Use a fixed ascending tap order and adjacent-pair reduction with the existing
   odd-tail rule for Z, mu and the final cotangent reduction. The implementation
   must describe precisely which unrounded/scaled quantities feed each reduction.
   A maximum of65 taps bounds all work; neither sigma nor values determine loop count.

The proposed polynomial and range-reduction scheme is provisional until independent
controls establish error bounds. In particular, large |x| magnifies ln2 split error;
an early-zero rule may only remove regions where BOTH weight and derivative are
provably below their respective rounding thresholds. Do not merely assume every
weight-underflow region has a zero derivative.

### Underflow, zero and overflow

For f64, q can round to zero while q*a or the normalized derivative still rounds to
a nonzero subnormal. Derive `dw` through scaled mantissa times `(a-mu)/Z`, then apply
the binary exponent; do not multiply a rounded-zero q or w. For f32 the same rule
applies despite wider intermediates. Scaled intermediates keep the fixed exponent
range explicit without constructing an unrepresentable exp value first.

Negative tap symmetry includes derivative bits. Zero raw/weight outputs are +0;
derivative zero uses the sign implied by the analytic branch except the explicitly
defined radius0 +0. A first implementation must choose and test the sign for tiny
negative centre derivatives rounded to zero rather than letting platform behavior
choose it. No NaN is accepted as an underflow result.

Weights and analytic unweighted derivatives are bounded on the accepted sigma/radius
domain; derive usable bounds before coding. Arbitrary finite incoming cotangents can
overflow a VJP product or sum. Preserve native typed overflow/error translation;
overflowed output is invalid, not clipped. Structural/domain errors preserve output.
Do not change existing intermediate-overflow policy to accept cancellation cases
without a separate contract decision.

## Proposed native and adapter surface (not yet adopted)

Candidate additive C family in `include/gffx/image.h`:

    gffx_image_gaussian_weights_workspace(batch_count, dtype, radius, context,
                                         required_bytes, required_alignment, diagnostic)
    gffx_image_gaussian_weights(sigma, radius, context, output, workspace, diagnostic)
    gffx_image_gaussian_weights_backward(sigma, grad_weights, radius, context,
                                         grad_sigma, workspace, diagnostic)

Use existing status/tensor/execution/workspace types and integer conventions. Root
must approve exact signatures, family count and append-only private provider tail.
Old providers reject without tail reads. CPU may use bounded stack arrays, no heap;
CUDA requires caller-owned validation/status workspace, with a query, not an assumed
zero size. Reject aliases, invalid dtype/rank/shape/device, out-of-range radius and
nonfinite/out-of-domain sigma or cotangents before output writes. Empty shapes still
validate descriptors. No new context creation: use the established scoped owner
context and caller stream with checked restoration.

Adapter proposal: `torch.ops.gffx.image_gaussian_weights(sigma, radius)` and backward
counterpart; `_GaussianWeights` saves only sigma/radius, recomputes bounded scalars,
and declares first-order-only with `once_differentiable`. Avoid retaining a hidden
tap Jacobian. Public `gaussian_blur` consumes the returned weights with its existing
convolution path; capabilities accurately mark the new custom-VJP boundary. No new
top-level public Python API is required. One native thread per sigma is a simple
bounded initial CUDA implementation; speed claims wait measured costs.

## Test-first sequence and remaining design obligations

Before production implementation, author and run independent oracle controls:

1. A stdlib Decimal oracle from exact represented input bytes: raw exp, normalized
   taps, sigma derivatives and VJP. Check known radius0/centre/symmetry cases and
   independently rounded f32/f64 conversions, including zero/subnormal tie cases.
   Oracle precision must be increased to demonstrate stable reference rounding.
2. Certify approximation-domain reduction and scaling controls, polynomial remainder
   plus coefficient/Horner/range-reduction error bounds. Generate constants with
   reproducible provenance. Do not claim exact rational or correctly rounded exp.
3. Enumerate every radius0..32; include sigma endpoints/neighbors, .83 and its
   historical neighbor, centre/outer taps, range-reduction transitions, normal/
   subnormal/zero transitions for BOTH weights and derivatives, logarithmic sigma
   sweeps and deterministic interior samples. Use original accuracy thresholds;
   add meaningful relative/ULP screens for tiny values, not zero-valued allclose passes.
4. Independently finite-difference sigma away from endpoints and numerical transition
   knots. Check scalar/shared/per-batch/empty shapes and random signed cotangents;
   test analytic derivative cancellation and representable gradients after weight
   underflow. Define finite-difference step/error bounds before interpreting results.

Then preserve a genuine missing-operation red before implementation. Native validation,
radius0/empty, old-provider tails, workspace/aliases/sentinels and typed overflow tests
precede integrated hardware qualification. Exact CPU/CUDA values/VJPs/repeats use
identical transferred input/cotangent bits, both dtypes, multiple radii and streams.
Retain the existing public blur failure unmodified as the integration regression.
Run CPU independent image controls and the whole remaining image matrix, not only
the reproducer. No qualification claim until these layers agree.

Unresolved before implementation: complete error derivation; exact signed-zero
derivative convention; precise scaled quantities/reduction rounding for Z and mu;
subnormal conversion proof and tests; exact C signatures/workspace cost. These are
design obligations, not hidden permission to weaken the domain. All compiler/Decimal/
CUDA work remains serialized under root leases with existing resource caps. No new
performance threshold, asset generation, dashboard upload or consumer work is included.

## Adopted direction / independent-oracle checkpoint

Root adopted the normalized-tap family direction, not production arithmetic or ABI
details. Source-only `tools/gaussian_weights_oracle.py` now separates:

- mathematical Decimal weights/analytic derivatives/VJP from exact IEEE input bits;
- direct nearest-even Decimal-to-IEEE rounding without a binary64 intermediate;
- an ideal derivative-rounded-once, rounded-product, adjacent-pair VJP convention;
- two-precision agreement of final bits, explicitly not a universal rounding proof.

`tests/python/test_gaussian_weights_oracle.py` has16 prospective parameterized
controls, including all33 radii, endpoint neighbors, central differences, tiny
representable gradients after weight underflow, cancellation and overflow. Authored
dashboard explanations do not make these unexecuted controls acceptance evidence.
No native implementation has been added. A bounded stdlib-only lease is requested.

### Constant generation and error-budget proposal for review

Generate ln2 with Decimal precision180 and independently240. Decode every chosen
binary64 constant back to exact Decimal. Proposed ln2_hi is the downward-truncated
binary64 ln2 with its lowest32 fraction bits cleared (at most21 significant bits);
ln2_lo is nearest-even binary64 of the exact high-precision residual. Generate1/k!
for k=0..18 independently from integer factorials and round each once. Record both
precision results, residual bounds and literal hex encodings; no guessed decimal
constants or production libm result is a source of truth.

A conservative early-zero region can simplify scaling without losing derivatives:
`a_i <= 1024000 < 2^20`, `0 <= mu <= max(a_i)`, and Z>=1. If the exact exponent
is below-799, then q < 2^-1100 (since799 > 1100*ln2), hence both q/Z and
`|q*(a_i-mu)/Z|` are below2^-1080, smaller than half a binary64 minimum subnormal.
A rounded exponent threshold of-800 would therefore be safe IF a derived input
arithmetic bound is below1. Under ordinary correctly rounded division/square/
half multiplication, the standard gamma3 bound over the full51200 magnitude is
far below1. These assumptions require explicit tests/strict-compilation evidence.
This rule zeros only a provably negligible region; it does not zero every tap
whose output weight underflows. Positive tiny derivatives near the rounding
transition still require the scaled path. Radius0 and centre remain exact branches.

For the retained region[-800,0], nearest-integer reduction has |n|<=1155.
The integer times truncated ln2_hi is exact in binary64 because their combined
significands need fewer than53 bits. Bound low-constant residual error separately
from low multiplication and the two subtractions; do not treat an exact high
product as an exact reduction. Require a certified |r|<=.347 envelope, including
nearest-integer selection/rounding; the exp polynomial remains valid at either
neighboring n if selection falls across a near-half boundary.

At |r|<=.347, the degree18 analytic remainder is below3e-26. Independently bound
coefficient rounding plus18 multiplication/addition pairs using a Horner gamma
bound on the absolute coefficient polynomial, then propagate range-reduction
error through exp. This approximation error is distinct from the original
`-.5*(i/sigma)^2` floating evaluation error: that error can be around1e-13 for
retained large exponents and must not be omitted from comparison with exact sigma.

Final normalized-weight error must propagate each tap error, the fixed-tree Z
reduction (depth at most7), division, and final dtype rounding. The derivative
bound additionally propagates a, mu and cancellation; a blanket relative bound
is inappropriate at a zero derivative. VJP error then separates this analytic
approximation from the explicitly rounded derivative/product/tree convention.
Those final derivative/cancellation bounds and precise scaled Z/mu quantities
remain unresolved: the rough global2^20 slope bound is useful for zero-region
safety but too pessimistic to serve as the final accuracy budget. Independent
oracle envelopes will test the eventual tighter derivation, not replace it.

## Complete conditional error budget - normalized weights and derivatives

This section bounds a specified arithmetic candidate, not arbitrary future code.
It closes the propagation beyond Taylor truncation while stating exactly which
primitive/reduction/scaling obligations must be proved against the implementation.
Let u=2^-53, b_i=i^2/(2*sigma^2), a_i=i^2/sigma^3 and w_i the exact mathematical
normalized weight. Maximum radius32 gives at most64 noncentre taps. All sums in
the mathematical bounds below are real sums; runtime uses the declared fixed tree.

### Uniform weighted moments (avoid the pessimistic maximum slope)

For t=|i|/sigma and integer |i|>=1:

    q_i*a_i = t^3 exp(-t^2/2)/|i| < 1.16/|i|
    q_i*a_i*b_i = t^5 exp(-t^2/2)/(2|i|) < 2.3/|i|
    q_i*b_i <= 1/e < .368

Differentiate these one-variable functions: maxima occur at sqrt(3),sqrt(5),sqrt(2)
respectively. Z>=1 and the exact rational harmonic number H32<4.06 therefore give:

| Quantity | Conservative bound |
|---|---|
| w_i | 1 |
| w_i*a_i | 1.16 |
| w_i*b_i | .368 |
| w_i*a_i*b_i | 2.3 |
| mu=E_w[a] | 2*1.16*H32 < 9.42 |
| E_w[b] | 64/e < 23.55 |
| E_w[a*b] | 2*2.3*H32 < 18.68 |
| absolute derivative | w_i*a_i+w_i*mu < 10.58 < 11 |

The numeric upper bounds can be certified without libm: use exact Fraction H32
and positive exponential series lower bounds for e,e^3,e^5. For example the
squared maxima are27/e^3 and3125/(4e^5). These certificate controls are still to
be authored/run; the calculus identities are separate from hardware evidence.

### Raw exponential envelope required of the candidate

For non-subnormal raw taps in the retained region, require:

    qhat_i = q_i*(1+eta_i), |eta_i| <= u*(100+5*b_i)

This is a budget to derive/check, not an observed result. Components:

- Rounded ratio/square/exact half: |xhat-x|<=gamma3*b_i. At b_i<=800 plus
  rounding margin, exp of that perturbation fits5*u*b_i including second order.
- Split reduction and fixed degree18 polynomial: the absolute coefficient
  polynomial divided by exp(r) is bounded by exp(2*.347)<2.01. A gamma36 Horner
  bound plus one rounding per coefficient contributes fewer than75u relative.
  The proposed exact high product, low product and subtractions must together
  contribute at most8u relative after exp. Final normal scaling and second-order
  terms must fit the remaining17u. Coefficient precision and the .347 reduction
  interval are explicit prerequisite proofs, not inferred from Decimal samples.
- Degree18 Taylor remainder below3e-26 is included but is not the dominant term.

If any implementation primitive exceeds this envelope, this complete bound does
not apply. Refine the arithmetic or derivation rather than relabeling an excessive
error as acceptable. A tiny positive q with an underflowed materialized value is
handled separately below, not forced into a relative-error formula.

### Propagation through exact normalization

Write A=100,B=5 and E=E_w[b]<23.55. Weighted mean raw relative error is bounded by
u*(A+B*E). It is less than1e-12; denominators remain strictly positive. For the
normalization perturbation, the first-order absolute bound is

    |delta w_i|/u <= 2*A*w_i + B*(w_i*b_i+w_i*E).

The exact quotient identity has a factor1/(1+E_w[eta]); using its lower bound and
retaining products of perturbations increases the displayed first-order bounds
by less than1%. This deliberate1% allowance is vastly larger than those actual
second-order factors, but small enough to keep the budget useful.

For d_i=w_i*(a_i-mu), the corresponding raw-weight perturbation satisfies

    |delta d_i|/u <=
      2*A*(w_i*a_i+2*w_i*mu)
      + B*(w_i*a_i*b_i + mu*w_i*b_i + E*w_i*a_i
           + 2*E*w_i*mu + w_i*E_w[a*b]).

Substitute the uniform moments, not maximum a_i. The right side is below6480.
This accounts for changing the output weight AND every weight contributing to
mu; it does not omit the normalized-derivative cancellation term.

### Runtime normalization/derivative arithmetic and tails

In addition to that raw-weight perturbation, budget1024u absolute per derivative
for the following bounded operations. This margin is not an arbitrary number:

| Rounding source beyond raw q | Conservative contribution in units of u |
|---|---:|
| two products for sigma^3 and division for a (relative error below4u), both direct a and mu | 43 |
| Z fixed-tree depth7 and weight division; propagate additional8u through local weight and mu using the same moments | 320 |
| products and depth7 fixed tree forming mu | 76 |
| final subtraction and multiplication/scaling, with two extra divisions if the scaled formulation needs them | 64 |
| second-order/cross terms and final rounding margin | 521 |

The implementation must actually use this operation count/order or supply an
updated bound. For a scaled representation, algebraically equivalent code is not
automatically covered: root's native contract must specify its Z/mu/scaling path.
The total derivative bound, including a1% perturbation allowance, is below8192u
(about9.095e-13). The analogous normalized-weight bound, with its tree/division
rounding and second-order allowance, is below512u (about5.685e-14).

Subnormal raw taps/associated products require absolute accounting. Even if each
such raw term changes by a minimum-subnormal quantum, at most65 terms times the
global slope bound2^20 gives a total far below2^-990. This conservatively covers
their influence on Z and mu. Add2^-990 to the preceding absolute bounds instead
of applying a false normal relative-error model. The proven early-zero region
contributes a still smaller amount. Crucially this small *global* error allowance
does not authorize dropping representable subnormal tap derivatives: the scaled
path and transition-specific oracle controls remain mandatory, independently of
the global absolute error screen.

For f32, wider f64 evaluation inherits those errors, then rounds weights and each
analytic derivative once. Weight rounding adds at most2^-24 (a conservative bound
for[0,1]); derivative rounding adds at most2^-21 because |d_i|<11. These fit the
existing f32 numerical accuracy requirements, but do not predict CPU/CUDA identity;
the latter still requires shared arithmetic and byte tests. No full correct-rounding
claim follows for either dtype.

### Metrics and evidence needed

Keep all existing absolute/relative acceptance checks unchanged. Add the following
diagnostics and conservative checks; they do not replace stricter existing tests:

1. Weight absolute error, symmetry and normalization residual. Optional relative
   error applies only where the reference magnitude is meaningfully above the
   dtype rounding floor; report the excluded tiny region explicitly.
2. Derivative absolute error against the bound above; report relative error only
   away from a zero. At exact zero, cancellation and underflow transitions, report
   signed-zero classification, ULP distance and nonzero representability separately.
   A blanket global allclose that accepts zero everywhere is insufficient.
3. Mathematical VJP and separately rounded-contract VJP are distinct references.
   For finite nonoverflowing cotangents, analytic approximation contributes at
   most sum(abs(g_i))*max(abs(delta d_i)). Product/tree rounding contributes its
   own standard dot-product bound (depth7 plus product rounding, absolute terms).
   Report norm-scaled VJP error alongside existing checks; it is not a new weaker
   tolerance. There is no uniform absolute VJP bound for arbitrary finite g.
4. Build independent directed interval checks for underflow/rounding transitions.
   Precision agreement is necessary reference evidence, not an exhaustive proof
   or a guarantee that a near-midpoint approximation picks the correctly rounded bit.

Unresolved choices sent to root: whether to implement the proposed split/Horner
scheme exactly or refine it; exact scaled arithmetic matching the counted operations;
and how much interval certification is needed around subnormal transitions beyond
the preserved byte/accuracy gates. Native signatures and sign/rounding contract live
in the separately root-owned Gaussian native contract, not in this draft.


## PC conditional certificate checkpoint - 2026-09-13

The eighteen-control independent oracle now passes on danziepc. Its original sixteen
controls missed the adopted radius-zero positive-zero VJP rule for negative incoming
cotangents; a new red regression preceded correction. Finite-cotangent validation
still occurs. This is reference qualification, not native acceptance.

`tools/gaussian_weights_bounds.py` and its five prospective controls turn the
conditional inequalities above into exact Fraction computations. ln(2) is bracketed
by 120 positive atanh-series terms plus a geometric remainder; both rigorous endpoints
round to the same split/reciprocal constants. Independent Decimal precisions 180/240
agree. The high product uses at most 32 significant bits. Certified reduced argument
bound is below .347; the degree-18 remainder is below 3e-26 and full mantissa error
below 100u. Input arithmetic contributes below 5bu separately. Exact positive series
and H32 certify the weighted moments without libm.

For the adopted native a=RN(RN(RN(j/sigma)^2)/sigma), the slope error uses gamma4,
not the older sigma-cubed operation count. The certificate separately charges slope,
Z tree, weighted-slope products/division/tree and final subtraction/product/division,
retaining the declared one-percent perturbation margin. The resulting conditional
weight and derivative budgets are below 512u and 8192u; no acceptance tolerance changes.
A separate absolute tail allowance remains 2^-990. The certificate assumes correctly
rounded primitive/scaling behavior: it does not prove compiled operations, every
subnormal rounding decision, native workspace cost or CPU/CUDA identity. Directed
transition and candidate-accuracy tests still precede native implementation.

Exact source hashes, constant hex encodings and measured test reports are in the
mapped PC output directory `m1-continuation-20260913`; NUMBERS.md owns observed values.
The five-certificate missing-module red and all previous failures remain preserved.


## Resolved subnormal interpretation - PC 2026-09-13

The user explicitly retained the existing accuracy and requirements. The adopted
bounded approximation, exact CPU/CUDA byte parity, scaled derivative evaluation,
signed-zero conventions and resource limits remain unchanged. No extra guarantee
of mathematically correctly rounded zero/nonzero classification is introduced.
The previous recommendation to revise the arithmetic for that stronger guarantee
was premature and is superseded. All three interval-certified differences below
remain diagnostic evidence; no compiled or native acceptance follows from this
clarification. Continue qualification against GAUSSIAN_NATIVE_CONTRACT.md.

## Historical subnormal decision proposal - superseded 2026-09-13

Directed-interval attribution now certifies 35 selected mathematical rounding
decisions from seven dtype/radius/quantity controls. Intervals use outward-rounded
basic Decimal arithmetic, positive exponential series with a geometric tail,
reciprocal and bounded squaring; they are independent of the candidate polynomial
and Decimal.exp. Their passing status certifies these selected intervals, not the
candidate's acceptance. Three sampled classifications disagree:

| dtype / radius / outer quantity | represented sigma bits | mathematical result bits | interpreted candidate bits |
|---|---:|---:|---:|
| float64 / 4 / weight | 4592130761096662846 | 1 | 0 |
| float64 / 8 / derivative | 4596590174398369896 | 0 | 1 |
| float64 / 16 / derivative | 4601097691735874510 | 1 | 0 |

A result of 1 above is the smallest positive binary64 subnormal (2^-1074), not
the floating value one. The global absolute bounds still hold, but those bounds
cannot decide a zero/nonzero preservation requirement. For the radius-4 weight,
the discrepancy persists with ideal exp evaluated at the prescribed rounded
exponent and exact mathematical normalization; increasing polynomial degree alone
cannot repair that example while preserving the adopted input evaluation order.

The existing contract simultaneously specifies a rounded operation order, disclaims
full correct rounding, and requires preservation of tiny representable derivatives.
It does not resolve how that preservation applies at the mathematical rounding
boundary. Before native code, obtain a concrete decision: retain the adopted bounded
approximation with explicitly documented possible boundary classification differences,
or require mathematical zero/nonzero preservation and revise/recertify the arithmetic.
The recommendation is to retain the preservation requirement and refine the arithmetic;
no accuracy criterion is silently weakened. This is not a request to raise a resource
limit, change convolution or implement a public generic exponential.

The interpreted candidate's original five controls passed; a sixth failed because
the authored test selected a positive exponent outside its declared domain. That
failed report is preserved. The corrected negative-boundary control and explanation
inventory pass in a focused retry. None of these interpreted results qualifies a
native symbol, compiled scaler, CUDA operation, VJP reduction or installed artifact.
Exact runs, hashes and measured costs remain in NUMBERS.md and the mapped PC outputs.
