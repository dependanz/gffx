# Soft coverage reference sigmoid

The scale-head32/64 soft render exposed different CPU/CUDA masks. Captured native
face indices, affine weights, depth and distance are byte-identical; the same
sigmoid input produces differing device values and first gradients. The32-square
mask difference is1.1102230246251565e-16. The failing reports remain preserved.

Use the existing certified degree18 split-ln2 exponential in float64 on
`-abs(x)`, then `1/(1+e)` for nonnegative inputs and `e/(1+e)` otherwise. Cast only
the completed value back to the input dtype. The -746 floor uses the existing
underflow certificate; it does not cut off a representable float64 value. Infinite
scores keep their limiting0/1 values. The existing rounded-mask first derivative
`gradient * y * (1-y)` remains, including saturation; this change does not assert
nonzero mathematical derivatives once the stored mask rounds to one.

Independent100-digit values and first-gradient errors are bounded by the existing
normalization reference allowance32*2^-53 for float64 and2^-23 for float32 on523
fixed samples. The original1e-10/1e-9 finite-difference gate remains. CPU/CUDA
values and VJPs must be exact, and the captured full soft-mask assertion remains.
Only installed-wheel results qualify the renderer. Large allocations still need
the complete planner and worker checks; this is not a new blur or visibility model.


Soft30 passes both independent523-sample dtype/device controls but the scale mask
still fails. A second capture isolates the preceding division: Python-scalar
`-distance/(soft*soft)` produces different scores on CPU/CUDA despite identical
distance bytes. Soft31 materializes the unchanged squared-radius constant in the
input dtype/device and uses tensor division, as already required for GGX pi.
No radius, threshold, epsilon, precision or mathematical equation changes.
The original full-mask assertion and every failed report remain preserved.
