# Principled direct-image checks

This development profile checks the pixels produced by the **public** complete
material and light integration against the independent scalar evaluator. It uses
the saved synthetic head, not a newly generated mesh or a real person's model.

Four albedo fields (constant, vertex colour, gradient texture and checker texture)
share the fixed 128x128 geometry buffers. Roughness is 1, RGB F0 is .04, and a
white directional light points along normalized (1,1,2). The exact accepted
`ggx-smith-loss-cpu-f64-v1` resource is used without experimental flags.

The scalar reference implements Hermite interpolation and adaptive mean integration
separately from the public tensor implementation. Its fixed-roughness curve is
explicitly prepared once, not rebuilt for each pixel. Both sides share immutable
geometry and table data; this checks the declared shading model, **not independent
rasterization or a new physical holdout**.

## Reading the comparison

Each albedo group has a reference image, the public result, and an error image.
The error is divided by `1e-10 + 1e-8 * abs(reference)` in linear RGB: black means
zero difference; white means at least the allowed difference. Display-only sRGB
conversion and clipping never change the original float64 comparison arrays.

## Why tiles and separate gradient tests?

Rendering a preview needs only the current answer. Fitting also retains information
needed to calculate derivatives, which costs more memory. This tool explicitly
uses at most eight points per forward-only call and rejects gradient-bearing input
buffers. It is not a training API and does not detach an application's graph.

Under the current complete-material limit, eight points with backward exceed the
64 MiB workspace estimate. Tests preserve that rejection and compare the gradients
of four points together against two two-point calls, including shared parameters.
No resource limit is increased to obtain a green image.

Each full image runs in its own serial CPU worker: one thread, a 120-second
cooperative deadline, a 180-second watchdog, and 1 GiB sampled whole-worker memory.
Reports and float64 payloads use new paths and are never overwritten. A failed
identity, numerical or resource gate stops the run and remains evidence.

Complete environment-lighting images require their own convergence work. These
direct images do not close that gate, GPU/export support, full shading acceptance,
or real-model gffit M1. They are numerical fixtures, not artistic showcase renders.

## Local result, 2026-09-08

All four saved images pass on the exact CPU wheel `6ab5ccb2...`, each with 6,536
covered pixels. The largest absolute linear error is `1.942890293094024e-16`.
Worker wall times were 20.17–25.95 seconds, including reference preparation and
verification, not just public shading. Sampled whole-worker peaks were below
559 MB. These are workload observations, not a throughput promise.

Evidence is in `build/principled-direct-20260908`: the initial wrong-hash stop,
four successful reports and paired float64 binaries, TDD XMLs, and final regression
(232 passed, one optional coverage-tool skip). The focused live suite has 30
passing cases, including saved-pixel replay and tamper controls. Dashboard page IDs
are `shading/principled-direct-images` and `shading/principled-direct-demo`.
