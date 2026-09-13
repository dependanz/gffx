# Textured environment-lighting image checkpoints

This is development verification, not a new public shader or complete-image claim.
It extends the independently verified96-point preflight to image tiles from the
same immutable synthetic fixture. No new geometry, texture or lighting table is
generated. The base package gains no dependency.

The selected profile has two materials (classic and principled), two environments
(uniform and fixed coloured affine world light), and three saved albedo fields
(vertex colour, gradient texture and checker texture). Each of these12 combinations
owns covered-pixel tiles0,26,51:128,128,8 pixels. This gives36 tiles and3168
pixel/profile comparisons, but only264 of6536 covered pixels per image.

`tools.albedo_tiles` uses six serial, create-only batches. Every pixel gets the
exact saved colour, fresh independent convergence history and two actual public
material/light integration estimates. Reusing worker-local fixed material moments
does not reuse a pixel answer or a success flag. Original reference sources and the
CPU wheel remain hash-bound; source or wheel changes require new evidence.

Each batch has900 seconds, reserving180 before launching another worker. Workers
retain120-second cooperative deadlines,180-second watchdogs,1GiB sampled whole-
process host memory and1024 lighting pairs per call. The whole selected procedure
allows at most5400 seconds and968933376 direction/moment samples. A failed batch
stops later work; existing output directories are never overwritten. Measurements
include worker imports; sampled working set is not an allocator peak measurement.

Reports under `build/albedo-tiles-20260909/batch-0` through `batch-5` retain each
tile's complete identity, pixel order, colours, geometry, convergence, errors,
work counts and supervisor results. `replay(root, digest, corpus)` requires the
exact batch hash. `combine` accepts the six ordered, strictly replayed batches;
it does not upgrade selected tiles into complete images. Devbrain's shading
contract8.32 and numbers ledger own measured outcomes and exact hashes.

The testing page is `shading/albedo-tiles`; the finite demonstration page is
`shading/albedo-tiles-demo`. Each demonstration panel puts independent reference
on the left and public gffx output on the right. Gray means not computed, including
the background. Display uses half exposure then sRGB; comparisons use unscaled
linear RGB. No continuous sender is started.

Remaining full images use the separately bounded, uniquely owned
[reference-only continuation](ALBEDO_IMAGES.md), which reuses these36 checkpoints.
Image-loss gradients, independently rasterized boundaries, arbitrary HDR maps,
other roughness profiles, resource device admission and M1 closure remain separate.
