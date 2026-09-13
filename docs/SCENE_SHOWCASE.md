# Kinetic Atelier — a scene presentation, not a conformance render

The stable testing page is `scene/hierarchy-demo`. The current producer is
`tools/testing_dashboard/scene_showcase.py`; the earlier `scene_demo.py` and its
hash-bound `SCENE_CORPUS.md` entry are preserved, not silently reattributed.

## What to look for

The turquoise anchor belongs to the turntable. Coral and ivory belong to a second,
upper arm, itself a child of the turntable. Gold connectors and joints make those
relationships visible. The plinth does not inherit either moving transform.

- **Whole assembly:** the turntable carries all three objects together.
- **Upper branch:** coral and ivory orbit; turquoise and the plinth stay fixed.
- **Explore:** drag the static mesh to inspect the bevels and connections.

The two24-frame sequences are finite authored transforms, not a physics solver,
skinning implementation or acceptance of the planned native animation operations.
Playback defaults to the dashboard's8fps cap; the speed control changes playback,
not the measured production speed. The mesh sender exits after publication.

## Why the cubes look different

The old preview used eight shared vertices per cube and omitted both colours and
normals from its mesh messages. The viewer then averaged normals across hard
corners, producing misleadingly dented-looking faces. The new first-party
procedural cubes have actual bevel geometry and explicit outward face normals.
Display vertices are split at triangle corners so creases are not accidentally
smoothed. Every static and queued mesh message includes colours and normals.

## Rendering and evidence boundaries

`gffx.scene.Scene.flatten` supplies packed geometry, attributes and world transforms.
The existing native rasterizer renders three1280×720 stills. Demo-side conservative
screen bins restrict each64×64 tile to nearby triangles; a test compares the result
with untiled native visibility and interpolation at a fixed2e-6 float32 tolerance.
This helper is not a new public tiled rasterizer or a gradient/near-plane contract.

Studio images use the existing cinematic Phong display recipe, tone mapping and
display-space edge smoothing, not public Principled evaluation, ray-traced shadows,
Cycles fidelity or numerical shading acceptance. Interactive WebGPU uses its own
preview light and will look different. No new external dependencies or downloaded
assets were introduced. Read `learn/scenes` for the underlying mathematics.

## Reproduction and resource limits

`tools.scene_showcase_probe` checks the exact installed wheel and records source
hashes. Its `tests`, `render` and `publish` modes run under the existing one-thread
CPU1GiB host,120s cooperative deadline and180s watchdog. Output reports are
create-only. Select the environment and wheel explicitly; do not replace a user's
normal installation merely to run this demonstration.

The finite clips contain48 mesh messages total, two24-frame recent windows and
snapshot_every=1 so each authored frame can be replayed. This does **not** change
the sparse-checkpoint policy for a continuous producer. Logical mesh tensor bytes
are recorded separately from wire/disk bytes and whole-process peak memory.

The first untiled high-resolution attempt was deliberately stopped for excessive
cost; its report is retained under `build/scene-showcase-20260909/render1.json`.
Rendered PNGs and subsequent reports remain local build outputs, not source assets.
Automated checks and agent visual inspection do not substitute for user acceptance.
