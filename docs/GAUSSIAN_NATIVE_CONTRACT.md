# Gaussian weights: native boundary and evaluation contract

Adopted boundary decisions, 2026-09-10. This is a gffx-only library repair.
The scalar approximation remains gated on the independent bounds and tests in
GAUSSIAN_WEIGHTS_DESIGN.md; this document does not claim an implemented operation,
correctly rounded exponential, tested device support or measured stack cost.

## Additive native boundary

Add these declarations to `include/gffx/image.h` only after the missing-operation
test baseline. Use the existing export and calling-convention macros:

```c
gffx_status gffx_image_gaussian_weights_workspace(
    int64_t batch_count, gffx_dtype dtype, uint32_t radius,
    const gffx_execution_context *context,
    uint64_t *required_bytes, uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic);
gffx_status gffx_image_gaussian_weights(
    const gffx_tensor_view *sigma, uint32_t radius,
    const gffx_execution_context *context, gffx_tensor_view *weights,
    const gffx_buffer *workspace, gffx_diagnostic_buffer *diagnostic);
gffx_status gffx_image_gaussian_weights_backward(
    const gffx_tensor_view *sigma, const gffx_tensor_view *grad_weights,
    uint32_t radius, const gffx_execution_context *context,
    gffx_tensor_view *grad_sigma, const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic);
```

Sigma has rank zero or rank one `[B]`; rank zero means one independent weight set,
not a length-one vector. With K=2R+1, forward output is `[K]` or `[B,K]` respectively.
The incoming weight cotangent has exactly that shape. Grad-sigma exactly matches
sigma's shape. Supported dtypes are float32 and float64; every tensor matches dtype,
device and device index. Radius is 0..32. Sigma lies between dtype-represented 0.1
and 32 inclusive. All scalar inputs and cotangents must be finite.

Descriptors follow the existing dense C tensor ABI, including byte offsets, flags,
rank-zero null shape/stride pointers, alignment and checked byte extents. Do not
add native arbitrary-stride support in this family. The adapter materializes dense
inputs while preserving autograd and reports its allocations through existing
resource accounting. Input-to-input aliasing is harmless and permitted; no output
may alias an input, and caller workspace may not overlap any tensor storage.

The query rejects negative batch counts, invalid dtype/radius/context, null result
pointers and overflow in B*K or its byte extent. Both element count and byte count
must fit the existing signed-count/addressable-storage conventions, including
INT64_MAX elements. It computes costs without reading sigma or allocating memory.
CPU reports zero workspace bytes and alignment one. The proposed CUDA family uses
eight caller-owned status bytes aligned to eight, including an empty batch. Both
values are explicit API costs to test, not an assumption that device work is free.
CUDA must reject an absent or old provider without reading beyond its declared tail.

Append private forward/backward function pointers and a new operation identifier;
do not repurpose the transfer modes or alter historical table field offsets. Query,
capability reporting, missing-family diagnostics and old-provider tests must agree.
No public generic exp function, global state, persistent cache or runtime dependency.

## Validation and failure atomicity

Validate descriptors, shapes, dtype/device, count overflow, workspace, aliases and
all domain values before writing any output. CUDA uses a validation pass and checks
its caller-stream status before the computation pass. It must not copy sigma or
cotangents to CPU for numerical evaluation. The existing checked owning-context
scope and caller stream are required; no new context, dummy warmup or default-stream
shortcut. Host reading of the small diagnostic status follows existing conventions.

Structural/domain failures preserve output sentinels. Arithmetic overflow returns
the existing typed overflow status, and the output is invalid (possibly partially
written). Every prescribed product and tree sum is checked: a later cancellation
does not retroactively make an overflowing intermediate valid. Do not clip.

For B=0, validate all metadata, context and workspace without dereferencing an empty
data pointer. For radius zero, validate sigma and every incoming cotangent normally,
return weight +1 and derivative/VJP +0. Do not infer forward versus backward from
a null data pointer; the two entry points determine the operation explicitly.

## Exact operation order to implement and test

For each independent sigma, widen float32 exactly to float64; float64 stays unchanged.
All following intermediates are float64 with round-to-nearest, gradual underflow
and no contraction. Controlled device division must match host division. Evaluate
only j=0..R and mirror the same bits to the ascending taps i=-R..R.

1. For j>0 compute t=RN(j/sigma), b=RN(t*t), x=-0.5*b and a=RN(b/sigma).
   The centre uses q=1 and a=0 exactly. Multiplication by -0.5 is exact in this
   domain. Obtain the bounded exp approximation as mantissa/exponent `(m,n)`;
   do not materialize an underflowed q and reuse it for derivatives.
2. Compute each raw q using correctly rounded binary scaling of `(m,n)` to float64.
   Form Z by adjacent-pair sums in ascending tap order, carrying an odd tail
   unchanged at each level. Z is at least one; no epsilon is allowed.
3. Compute the weighted-slope contribution using
   `scale(RN(RN(m*a)/Z), n)` for each tap; centre contribution is +0. Form mu with
   the same tree. Scaling occurs after multiplication and normalization, retaining
   contributions that would be lost by first rounding q to zero.
4. Compute weight as `scale(RN(m/Z), n)` and the analytic derivative approximation
   as `scale(RN(RN(m*RN(a-mu))/Z), n)`. Mirror their bits. Return each weight after
   the final conversion to the input dtype. Weight values are nonnegative; zero
   weight is +0. An exactly zero derivative coefficient is +0; a nonzero coefficient
   rounded to zero retains its sign. Radius-zero VJP is explicitly +0.
5. For backward, convert each computed derivative to the input dtype first. Round
   each multiplication with its represented incoming cotangent in that dtype.
   Reduce these products using that dtype and the same ascending adjacent-pair
   tree. No native cross-batch reduction: shared-sigma spatial reductions remain
   the existing convolution's responsibility.

The independent Decimal oracle separates the mathematical derivative from this
rounded derivative/product/tree convention. Approximation error is measured against
both; exact agreement with the old PyTorch exp graph is not promised. CPU/CUDA bytes
for this new evaluation order must agree. Near-zero derivatives need an absolute
condition-aware error bound; unqualified relative-error claims are inappropriate.
The interval/range-reduction/polynomial and propagated error bounds remain separate
pre-implementation obligations, not satisfied by specifying an operation order.

## Binary scaling proof obligations

Use a dedicated finite-value bit scaler, not the colour helper's unqualified libm
subnormal fallback. Decompose a finite binary64 input into sign, integer significand
and exponent. Normalize a subnormal with at most 52 bounded shifts, preserving zero.
Adjust the exponent with checked integer arithmetic. A normal result changes only
exponent bits. For a subnormal result, right-shift the integer significand and use
the discarded bits for round-to-nearest, ties-to-even; a carry may produce the
minimum normal. Excess shifts produce signed zero, not an undefined C shift.
Overflow is explicit. Do not lose the sign when normalizing or rounding to zero.

Independent exact-integer/Decimal controls must cover both signs, zero, minimum
subnormal, subnormal-to-normal carry, midpoint ties, ordinary normal scaling and
overflow. Compile this same scalar source for host and CUDA with strict flags, and
test the final float32 conversion as well. Passing an interpreted model alone is
not compiled-scaler qualification.

## Adapter and bounded storage

Register `torch.ops.gffx.image_gaussian_weights` and its backward counterpart.
Use a first-order-only custom autograd boundary saving sigma and radius, not a tap
Jacobian. The existing public `gaussian_blur` signature and image/border/alpha rules
stay unchanged. Its parameter helper currently reshapes per-batch sigma to
`[B,1,1,1]`: reshape to `[B]` for this native call, and reshape each returned tap
back to `[B,1,1,1]` for `_WeightedMultiply`. Keep scalar sigma rank zero throughout.
Do not silently broadcast a length-one batch or lose empty-batch semantics.

Keep the current horizontal-then-vertical convolution and fixed spatial reductions.
Use one native CUDA lane per independent sigma with bounded per-lane local storage
and no hidden heap allocation. CPU stack use and compiled CUDA register/local-memory
costs must be measured before scale claims. Outputs and adapter temporary storage
still count against caller budgets; eight workspace bytes do not describe the
whole operation's footprint. Existing worker, GPU, time and single-worker limits
remain unchanged.
