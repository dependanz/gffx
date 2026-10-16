# Private positive-unit power for Classic reference arithmetic

Prospective, 14 September 2026. The existing Classic lobe accepts a real finite
nonnegative exponent and both floating precisions. Its fixed-square repair does
not qualify other exponents. The retained noninteger contract operands expose a
first-gradient device difference and require a broader repair, not a restriction
to integer exponents or a tolerance change.

Candidate value: evaluate exp(n log(c)) in float64 using the separately qualified
positive-unit logarithm and the degree18 split-ln2 exponential arithmetic. Keep
exact zero/first/square powers for those exponent values. The public c=0 endpoint
continues to use its original explicit value and zero-derivative convention.
Internal higher precision does not change input/output dtype or device ownership.

First derivatives use the analytic powers in the log domain, including n*c^(n-1)
and c^n*log(c), to avoid prematurely overflowing a reciprocal or underflowing a
power before its scale is applied. Each result is cast once to the input dtype.
The positive exponential is formed from the reciprocal of the same negative
exponential; positive-power overflow remains explicit mathematical nonrepresentability.
No queried value is sent to CPU, and no new runtime dependency is added.

Reuse the degree18 reduced-interval remainder and split-ln2 certificate. The
private negative exponential may extend its input floor from-745 to-746; this
changes exponent scaling, not the reduced polynomial interval. Prove with high
precision that exp(-746) is below half the least float64 subnormal before using
zero beyond that floor. Existing loss-resource domain/admission remains[-745,0].
Do not claim general correctly rounded pow, exp or log.

Acceptance retains the existing lobe algebra and derivative criteria. Compare
both dtypes against independent90-digit Decimal values and analytic first partials,
including noninteger exponents, normal/subnormal cosines and endpoints. Require
exact CPU/CUDA value and first-VJP bytes on the declared grids and original failing
operands, plus first and mixed-second finite differences in the interior. Check
that finite representable reference derivatives remain finite and nonzero when
required by their existing acceptance. Keep all prior failures. This finite
qualification does not prove every possible floating input or higher derivative.
