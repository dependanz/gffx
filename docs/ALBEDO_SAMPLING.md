# Bounded public sampling comparison

This is development verification, not a new runtime API or a universal fast path.
`tools.albedo_sampling` compares grouping up to eight surface points with the
existing one-point sampler. Both call the public material and light integrator,
retain at most 1,024 point/light pairs per call, and preserve each point's
accepted quadrature order, world-space directions, weights and both partitions.
Different integration orders are grouped separately.

The experiment selects the first and last pixels of each previously qualified
tile, for each saved albedo field: 18 pixel/profile observations per worker,
72 across two materials and two lighting environments. Exact parent batch hashes
and rows are checked during replay. This is reused regression evidence, not a
new physical holdout or full-image acceptance.

Each worker alternates method order over one warmup and three measured passes.
Both methods use the same outer timing boundary, including validation. Internal
sampling times are retained separately. All calls are counted once, including
all participating pairs. The batched layout makes 1,016 calls per selected pass
versus 1,008 originally; it does not reduce calls by serving more points per call.

The local result is mixed. Classic medians decreased by roughly 6–10%.
Principled uniform-light timing changed by only about 1%, potentially noise;
principled coloured-light batching was about 6% slower. Keep the original method
for that slower profile. Three repetitions on this laptop are not a general
performance guarantee. No runtime implementation, resource admission, dependency,
accuracy threshold or memory limit changed.

Evidence lives in `build/albedo-sampling-20260909/screen`; devbrain's shading
contract section 8.33 and NUMBERS.md own exact hashes and measurements. The
testing pages are `shading/albedo-sampling` and `shading/albedo-sampling-demo`.
The demo is a finite saved comparison, not a continuously running benchmark.

The assessment helper requires previous public results for strict regression
parity. New image pixels have no such results. The separately tested
`tools.albedo_image_sampling` interface now accepts reference-only inputs for
[full-image continuation](ALBEDO_IMAGES.md). It preserves the selected numerical
routes without fabricating previous results or copying reference values into a
field claiming measured public outputs. The assessment source remains frozen.
