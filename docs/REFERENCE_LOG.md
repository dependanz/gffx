# Private reference logarithm for bounded material resources

Prospective repair,2026-09-13. Captured identical view cosine0.6378118574468598
produces CPU/CUDA Torch.log values differing by5.551115123125783e-17 and a public
view-loss difference. Keep strict byte and independent physical criteria unchanged.

Use only positive finite float64 x<=1. Scale subnormals exactly by2^52, extract
binary exponent/mantissa, and move the mantissa into[1/sqrt(2),sqrt(2)] with one
exact power-of-two adjustment. For z=(m-1)/(m+1), evaluate
log(m)=2*z*(1+z^2/3+...+z^38/39) in separate fixed-order float64 operations.
Add the integer exponent times the same split-ln2 constants used by the already
qualified private exponential. This is no public log API or correctly-rounded claim.

Independent analytic remainder: abs(z)<.18, so the omitted tail is bounded by
2*(.18)^41/(41*(1-.18^2))<1e-31. The numerator subtraction is exact by Sterbenz;
positive-series Horner operations have no cancellation. Standard roundoff bounds
for the division,19 Horner stages, final product and split-exponent addition are
conservatively covered by128*u*(1+abs(log(x))),u=2^-53. The exact-rational test
certifies the tail and100-digit Decimal verifies the split constant error times
1074 below1e-19. The high split times the bounded integer exponent is exact.
Near1 uses exponent0, avoiding subtraction of two log(2)-sized terms; x=1 gives
zero and x<1 stays negative. Subnormal inputs are not flushed or clamped.

The first VJP is gradient/x; existing higher-derivative connectivity remains.
Apply the same composition to setup-derived table logs and runtime coordinate
logs, preserving exact node reuse and immutable raw payload hashes. Requalify
independent physical/curve, derivative, device and resource gates after rebuilding.
A fixed multiply tree for the integer fifth power separately replaces generic
pow at Schlick's unchanged polynomial; its value/gradient tests precede the edit.
