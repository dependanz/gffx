# Environment image-loss adjoints

## Whole-image extension

The original selected-pixel preflight below is retained as measured history.
Contract 8.38 extends it to four complete fixed 128x128 profiles: two materials
under uniform and coloured affine world lighting. Each image owns 6,536 covered
pixels and a 32x32 RGB source texture. Geometry, UVs and material/light parameters
stay fixed. Only source-texture colour is fitted.

The bounded implementation computes each angular chunk once, obtaining its full
3x3 RGB-to-albedo Jacobian with three reverse calls. After summing complete pixel
radiance and Jacobians, it applies the loss cotangent and performs one native
texture VJP. A non-diagonal colour-mixing control prevents an accidental assumption
that red output only depends on red albedo. Both-material small comparisons also
agree with the original two-pass implementation.

An image loss is the sum of squared covered-pixel RGB residuals divided by
`3 * 6536`. Tile means and gradients are weighted by each tile's pixel count.
The two light-sampling offsets are checked separately. Only after the entire
first image is assembled do we update the shared texture by `T -= 5 * gradient`.
All pixels are then freshly evaluated; both losses must strictly decrease.
This is not a collection of separately fitted tiles or differentiation of saved
public images. Independent closed forms/moments and a scalar bilinear transpose
remain the reference, with unchanged tolerances and two-scale finite differences.

Each profile has 104 supervised workers: 52 partitions at each of two iterations.
All four total 948,128 public material calls and 970,883,072 sampled directions.
These declared work counts are not performance targets. Workers retain 1 GiB
sampled host memory, one CPU thread, 120-second cooperative and 180-second hard
limits; each profile has a 6,000-second aggregate ceiling. No whole image's
autograd graphs are retained. Source/wheel/corpus/texture/report identities and
every pixel's ownership are checked before declaring completion.

Current completion and artifact boundaries are in [the shading status map](SHADING_M1_STATUS.md).
The finite gallery is `shading/environment-image-adjoints-demo`; its replay tests
are on `shading/environment-image-evidence`. Neither page claims whole-image
CUDA, UV/geometry/material/light derivatives or M1 completion.

## Original two-pixel preflight

The original test qualified **two selected surface pixels per profile**, not
complete-image fitting. It extends the direct-image source-texture derivative
checks to uniform and coloured affine environment light under both complete
materials. Geometry, UV coordinates and material/light parameters are frozen.

## Why two angular passes?

A pixel's radiance is a sum over light directions. Its squared error must be
computed after that sum; summing the squared errors of partial light chunks
would be a different objective.

The fixture first computes the complete radiance without retaining graphs. It
then recomputes small angular chunks and differentiates each with the fixed
incoming sensitivity `2 * (complete_RGB - target_RGB) / (3 * covered_pixels)`.
Each chunk rebuilds the native texture sampler graph and releases it after
obtaining its source-texture derivative. This trades recomputation for bounded
graph memory; no full angular graph is kept.

Independent scalar material integration supplies the initial radiance. Exact
classic hemisphere moments, or independently converged principled loss moments,
supply colour sensitivity. A scalar bilinear transpose and three two-scale
finite differences check the source gradient. A tiny complete autograd graph
also checks the streamed method under both materials.

The two sampling offsets are validated separately. A fixed texture update uses
their average gradient, and both second-measurement losses must decrease. RGB
and source-gradient tolerances are unchanged. The scalar loss tolerance is
prospectively propagated from the integrated RGB error allowance; it is not a
claim that numerical environment integration has exact direct-light arithmetic.

## Measured preflight, 2026-09-09

All four workers passed, with fixed pixel IDs 2904 and 8536 from the shared
128x128 buffer. They represent the most grazing and middle selected view ranks.
Orders 128 and 32 converged independently. The experiment used 2,176 public
material calls, 2,228,224 public direction evaluations, 741,376 independent
reference directions and 57,344 radial moment samples.

The four-worker cohort took 35.1076 seconds. The largest sampled whole-worker
working set was 565,596,160 bytes; no worker exceeded 10.600 seconds. Existing
1 GiB host, 120-second cooperative, 180-second watchdog and 900-second aggregate
limits were retained. These observations do not establish sustained throughput,
allocator peak or a safe full-image schedule.

The local evidence is `build/environment-loss-20260909/qualification/cohort.json`,
SHA-256 `c8af76f25931a778afb6801853bf21fab94edfab5174cdc3b411cb0442d19b1d`.
The cohort binds exact sources, installed wheel, corpus, independent reference
histories, pixel identities, work counts and actual monitored PIDs. Saved replay
recomputes independent references and adjoints, not public shading. All old image
cohorts and numerical sources remain unchanged.

The testing page is `shading/environment-loss`; the finite demonstration is
`shading/environment-loss-demo`. Its swatches show **target / before / after**
for each of the two measured pixels. They must not be read as full images.
The `learn/autodiff` chapter explains the two-pass chain rule.

Coverage remeasurement is deferred until after M1 at the user's request.
Correctness tests continue. No runtime dependency or public API was added.
