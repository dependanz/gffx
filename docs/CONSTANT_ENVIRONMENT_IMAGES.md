# Constant-environment image qualification

This development harness checks two complete 128×128 images of the existing
synthetic head, with 6,536 visible pixels each. The material is either classic
(specular 0.1, exponent 2) or principled (roughness 1, F0 0.04), with constant
albedo `[0.2, 0.4, 0.6]` and uniform unit environment radiance.

These are shared-geometry shader tests, not an independent rasterizer, an artistic
head model, general HDR lighting or a training renderer. See the authoritative
devbrain shading contract §8.24 and numbers ledger for measured status.

## Independent reference

The fixed-profile tensor evaluator is separate from the public implementation.
Individual samples are checked against the frozen scalar evaluator. At roughness
one, correlated GGX visibility simplifies the glossy term to
`F / (2 π (N·V + N·L))`; the already-qualified separated reference integrates
the reciprocal matte term exactly. The candidate does **not** use that shortcut:
it evaluates the entire public BRDF and calls `light.integrate`.

Every pixel must pass two successive resolution transitions and agreement between
two angular partitions. The reference tolerance remains
`1e-11 + 1e-9 * abs(reference)`. Both public estimates must independently satisfy
`1e-10 + 1e-8 * abs(reference)`. A reference from a different pixel is never
interpolated or reused as that pixel's answer.

## Bounded execution and replay

The application explicitly assigns 128 visible pixels per worker, 52 tiles per
material, with one CPU thread. Existing 120-second cooperative, 180-second watchdog
and 1 GiB sampled-worker limits apply. The cohort has an 1,800-second aggregate
launch budget and stops at its first failure, without retrying or overwriting.

Each tile retains pixel IDs, reference histories, both public estimates, source
and wheel identities and actual-worker supervision. Final assembly requires every
tile in order and all visible pixels exactly once. The saved background is exactly
zero. Partial tiles and failed cohorts remain evidence of progress or failure,
never complete images. Dashboard playback uses saved evidence, not a live renderer.

No runtime dependency, material formula, table, precision or acceptance tolerance
is changed by this harness. Image gradients, different albedos, nonuniform
environments, CUDA/export and end-to-end consumer qualification remain separate.

## Approved checkpoint continuation

The original run stopped at its aggregate guard after completing all classic
tiles and principled tiles 0–31. That stopped report remains immutable.
`tools.environment_continuation` implements the separately approved 900-second
budget for tiles 32–51 only. It verifies the original report, all passing tile
identities, pixel geometry and convergence histories before starting new work.
It refuses existing output directories and changed numerical source files.

The continuation records original, additional and cumulative elapsed times,
without pretending the entire computation took only the continuation time.
Final replay combines the old and new tile references exactly once and verifies
the completed image arrays. A successful continuation does not erase the earlier
stop or extend qualification to nonuniform lighting or gradients.
