# Reference-mode normalization: bounded square-root composition

Prospective decision, September 13, 2026. The packed-head diagnostic found154
CPU/CUDA sqrt differences on identical float64 operands, after fixed channel sums
and divisions already matched. Do not waive the public reference-byte gate.

Unit vectors first divide by their largest absolute component. Their squared
length x is in [1,3]. For this private, validated range only, evaluate six Newton
steps in float64, starting y=1.5, each exactly y=(y+x/y)*0.5, then round once to the
original dtype. Every tensor operation is separate; there is no reassociation,
input-dependent iteration count, native ABI addition or general-purpose sqrt claim.
Use the analytic first derivative 1/(2*y); retain graph connectivity for existing
higher-derivative tests. Invalid inputs and zero-vector semantics stay unchanged.

Independent forward bound: let u=2^-53. Starting absolute error is at most1/2.
For a current bound E<1, y>=1-E. Exact Newton error is at most E^2/(2*(1-E)).
A conservative16u bounds the local roundoff of the division/addition; multiplying
by one half is exact on this normal range. Iterating that rational upper bound
six times gives E<32u. The intermediate interval stays positive and bounded; no
underflow/overflow occurs. Final float32 rounding adds at most u32*(sqrt(3)+32u),
so absolute error is below2*u32. These bounds preserve the existing independent
normalization/gradient tolerances; they are not a claim of correctly rounded sqrt.

Before adoption: certify the recurrence using exact fractions, compare endpoint,
interior and captured-input neighborhoods with independent high-precision Decimal
sqrt, check first/second derivatives, and require exact CPU/CUDA bytes. Requalify
real-head normals/shading and affected finite physical/device/resource gates after
an installed-wheel rebuild. The ordinary Torch sqrt remains for unrelated domains.

## Exact native fusion (opaque193)

The C11/CUDA vector primitive performs exactly the same six binary64 divisions,
additions and half-products, with explicit round-to-nearest CUDA intrinsics and
separate volatile C intermediates. It is not a replacement by platform sqrt.
The [1,3] accuracy proof above is unchanged. Raw inputs may lie in (0,4] to
retain endpoint-neighbour differentiation; no expanded accuracy claim follows.
Counts are bounded by1048576. Larger eager arrays retain their previous path.
The output replaces the old iteration result; contiguous input, cast and output
fit within the previous simultaneous iteration-temporary inventory. Caller
scratch is8bytes. Domain errors precede writes; aliases and short scratch refuse.
First and higher-order Python derivative expressions and float32 casts stay the
same. The provider slot is append-only and size-checked before CUDA dispatch.


Complete vector fusion (opaque194) extends those same separately rounded steps
with max-absolute scaling, three scaled divisions, component squares and the
left-associated three-term sum, followed by final divisions. It applies only to
float64 forward execution under no-grad; differentiated graphs and float32 retain
their existing composition. Finite nonzero vectors include subnormal and maximal
inputs; no magnitude cutoff is introduced. At most1048576 vectors are admitted.
Output and any contiguous copy replace existing scaled/output temporaries;8bytes
of caller scratch remain inside the original conservative storage inventory.
