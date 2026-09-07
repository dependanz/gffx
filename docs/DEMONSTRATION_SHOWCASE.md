# Visual demonstrations

## High-resolution presentations (2026-09-07)

`camera/fit-demo` now leads with **Recover the shot**: the target and the actual
optimized rendering, an external camera view and a target/result wipe. It fits
orientation, position and log focal scale from 96 known landmark correspondences.
Adam is followed by L-BFGS; playback contains actual saved optimizer states.
The objective is landmark reprojection, not image loss or unknown-object recovery.
Independent derivative comparisons and numeric diagnostics remain expandable.

`shading/lights-demo` gains **Light reveals form**, with a choreographed key, cool
rim light and camera movement. The original analytic falloff/SH evidence remains
underneath. The display recipe uses a reciprocal Phong highlight with a conservative
hemispherical energy bound: its exponent-72 lobe is scaled by 73/(2*pi), then mixed
with a 0.12 specular and 0.88 diffuse budget (albedo at most one). The cosine factor
is at most one, and the unweighted hemisphere lobe integral is bounded by 2*pi/73.
This is a tested development recipe, not the pending public principled material.
No shadows or reflections are claimed. One Loop-style display refinement smooths
the offline fixture; it is not a public gffx subdivision operation. Original asset
files and attribution are unchanged. A bounded display-space edge filter softens
aliasing; these images are not numeric or gradient acceptance evidence.

The presentation targets 1280x720 frames and a 1920x1080 still. Stdlib PNG encoding
uses struct/zlib/CRC; no external codec is required. Content-addressed PNG files live
under the tester root's `presentation/`, not in live queues. A small record references
them. Limits: 8 MiB per PNG, 256 MiB per publication bundle, 512 MiB checked store
budget before new writes, and at most 96 frames accepted by the browser. Existing
assets are deduplicated by SHA-256. Files are written atomically before the demo is
published; generation failure does not replace the current page. Existing previews
are not automatically deleted. Serialize publishers; the pre-write store budget is
not a cross-process transactional quota.

The browser requests only the current frame pair and fixed target, validates local
digest URLs, sizes/dimensions and hashes, releases decoded images, and cancels work
when hidden or disposed. PNG browser caching is private and immutable by hash.
Playback never starts rendering or writes history. Requested FPS remains a ceiling.

```powershell
python -m tools.testing_dashboard.cinematic --address http://127.0.0.1:8791 --root dist/testing-dashboard-preview-20260905 --run recover-UNIQUE --mode fit
python -m tools.testing_dashboard.cinematic --address http://127.0.0.1:8791 --root dist/testing-dashboard-preview-20260905 --run reveal-UNIQUE --mode lights
```

Use fresh names; generation uses the optional installed PyTorch/CUDA adapter.
`--device cpu` is supported but slower. `--preview` is explicitly 320x180 diagnostic
output, not the presentation-quality target. Existing low-resolution studio/camera
playground clips remain available as diagnostic demonstrations; this first HD pass
prioritizes the fitting and lighting headlines.

The testing application uses the same gffx.dashboard panels as ordinary users. Its
scene examples additionally require the already optional PyTorch adapter. No new
frontend, rendering, physics, image-codec or asset-download dependency is added.

## Earlier diagnostic presentations: reproduce from a checkout

Use an installed adapter-enabled gffx wheel and a running tester. Pick fresh execution
names; these commands save finite clips and exit. CUDA is optional: omit `--device cuda`
for the slower CPU reference composition. Neither option advertises live rendering FPS.

```powershell
python -m tools.testing_dashboard.showcase --address http://127.0.0.1:8791 --run studio-UNIQUE --device cuda
python -m tools.testing_dashboard.lighting_demo --address http://127.0.0.1:8791 --run lights-UNIQUE --studio --device cuda
python -m tools.testing_dashboard.camera_fit_demo --address http://127.0.0.1:8791 --run fit-UNIQUE
python -m tools.testing_dashboard.classic_demo --address http://127.0.0.1:8791 --run classics-UNIQUE --inspect-meshes
python -m tools.testing_dashboard.mesh_animation_demo --address http://127.0.0.1:8791 --run deformation-UNIQUE
python -m tools.testing_dashboard demo --address http://127.0.0.1:8791 --run panels-UNIQUE
```

Depth publication is `tools.testing_dashboard.showcase.publish_depth(address, run,
device='cpu')`; it uses the same scene and public `Camera.unproject`.

## What the earlier diagnostic publications teach

The commands above reproduce the earlier saved evidence. The HD publishers supersede
the current fitting and lighting page bindings; older executions remain inspectable.

- `shading/lights-demo`: warm/cool, neutral studio and approaching-point presets.
  The selector switches between color, world-normal RGB and positive-depth images.
  The original analytic curves and signed-SH comparison remain below the scene.
- `camera/studio-demo`: orbit, pan, dolly, zoom and dolly-zoom, with a synchronized
  external view of the camera/frustum. The reference plane, not the whole object,
  keeps its scale during dolly-zoom. Clip selection pauses playback.
- `camera/depth-demo`: the camera image, positive view depth and an orbitable cloud
  reconstructed from a four-pixel-stride grid of visible pixels. The unseen back is not reconstructed.
- `camera/fit-demo`: 32 known bunny landmarks, four fitted intrinsics, fixed pose and
  geometry, 300 optimization steps and 61 saved frames. Arrows show pixel residuals;
  the shared plot cursor uses the same optimization step, never the derivative sweep.
- Classic meshes: native normal-color display, picked-face area/normal readouts and a schematic normal arrow
  when published with `--inspect-meshes`. Fixture units are not asserted to be metres.
- Ball: slow motion, impact close-up, and synchronized energy bars. The final bar is
  initial minus current mechanical energy, not independently measured heat. The
  centroid/contact illustration still is not a gffx physical-simulation implementation.
- Deformation: grey rest-grid overlay and fixed-scale 3D displacement colors, not
  strain/stress, skinning, blendshapes or an implemented animation API.
- Panel playground: vertex selection highlights its associated RGB swatch; the
  history curve is explicitly synthetic rather than a fabricated optimization result.

## Rendering and evidence boundaries

The studio uses gffx camera projection, native vertex normals, hard rasterization and
perspective-correct attribute interpolation. Its direct Lambert lighting is development
composition, not the future scene/shading/deferred API. Demo fixture winding is normalized
by signed volume before computing normals; bundled assets remain unchanged. See their
separate licenses in `tools/testing_dashboard/demo_assets/README.md`.

The point source uses a small distance guard; no source coincides with the displayed
surfaces in the default scene. The fixed fill and final x/(1+x), gamma-1/2.2 display transform
are not physical validation. A yellow source cross and cyan camera frustum are schematic
overlays, not shadow/occlusion tests. There is no indirect illumination, shadow solver or
claim of Cornell Box reference agreement.

Studio clips contain 24 frames at 256x192, explicitly saved once with recent_window=24
and snapshot_every=1. Producers exit. Playback is serial, initially paused and bounded;
hiding/leaving a page stops it. Requested 4–60 fps are ceilings, not achieved rates.

The tester can unload finished, idle, fully durable execution trees when its existing
eight-tree limit is reached. It never deletes logs or discards active clients/pages,
legacy queue tails, ephemeral modern frames or recording-warning state. The general
dashboard keeps its original capacity policy. Persistent logs retain their quotas.

Tests: `tests/python/test_demo_showcase.py`, existing dashboard tests and Node frontend
checks. They do not replace DB-11, physical mobile acceptance, public-operation gradient
tests or sustained memory/latency measurements. The camera and lighting textbook chapters
link these demonstrations and explain their mathematical limits.
