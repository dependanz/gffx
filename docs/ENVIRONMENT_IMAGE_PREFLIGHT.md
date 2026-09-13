# Complete environment-image preflight

The full-image gate remains **open**. These checks establish accurate references
and reveal the cost of actual public material integration before an image sweep.
They do not substitute a diffuse-only image or private shader for the candidate.

The immutable head buffers from `classic-images-v1` supply eight pixels sorted by
positive viewing cosine, spanning grazing to frontal views. Each is measured under
constant unit radiance with albedo [.2,.4,.6], classic specular .1/exponent 2, and
principled roughness 1/F0 .04 with the exact accepted CPU float64 budget.

## Preserved reference failures

The original `env-sh2-v1` uniform-azimuth rule does not converge at the most
grazing head point (view cosine .014622699987302479). Splitting the half-azimuth
into Gauss-integrated intervals resolves classic but leaves principled unresolved.
Squaring the radial coordinate clusters samples near the horizon, but principled
still obtains only one qualifying refinement rather than the required two.

These are failed measuring methods at this profile, not observed shader failures.
All histories remain in `build/environment-images-20260908`.

## Successful independent reference

`constant-env-separated-v1` retains two angular partitions and the squared radial
coordinate. It integrates the reciprocal matte term exactly under constant light:

    integral of a * L(view) * L(light) / (pi * M) * cosine = a * L(view)

Here M is the cosine-weighted mean of the same loss curve. The scalar reference
owns independent Hermite interpolation and adaptive segment integration. The
glossy term is still integrated numerically from the original scalar material.
Neither plain albedo nor `1-L(view)` replaces the complete material.

All sixteen reference points converge with the unchanged `1e-11 + 1e-9*abs(value)`
threshold, two successive passing transitions, five levels, 87,296 sample cap and
30-second reference deadline. Total actual samples: 331,776. Every serial worker
stays within the separate one-thread, 120-second cooperative / 180-second watchdog
and 1 GiB sampled whole-worker limits.

For classic, an independent exact control is:

    albedo + specular * (5 * viewing_cosine / 8 + 5 / 12)

The rule is restricted to constant illumination and isotropic materials. It does
not qualify arbitrary SH lighting, transport, full images or gradients.

## Actual public candidate and remaining work

The most-grazing principled point is also evaluated with the actual installed
`material.principled` and `light.integrate`, using the full BRDF in ten-direction
chunks. That keeps expanded RGB resource queries at 30, below the current cap 32.
Both estimates match the independently converged reference; maximum absolute
error is `2.1077883882725246e-10`, within the unchanged image tolerances.

But 65,536 directions require 6,554 public material calls and **110.205 seconds**
of sampling for this single point (113.784 seconds including worker setup;
548,593,664 bytes sampled peak). This is not an image frame-time measurement or a
prediction for all other views. It demonstrates why an unbounded full-image loop
is not the next responsible experiment.

A bounded CPU integration/resource optimization and its fresh validation are the
next prerequisite under discussion. No larger query limit, runtime dependency,
new table, ordinary-environment replacement, CUDA support or full-image acceptance
is claimed. The tester page `shading/environment-image-progress` replays pinned
reports only; it starts no background renderer.
