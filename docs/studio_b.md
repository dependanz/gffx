# Raster scene reflections and indirect light

Studio B extends the forward raster renderer with actual captured geometry,
environment illumination and a static diffuse bake. These local development
features are not yet a published PyPI release. See `examples/studio_b.py` for a
compact caller; the scene author/export and comparison harness live in
`tools/studio_b`. Blender authors/renders the separate reference, never gffx's
image. No other renderer is a gffx dependency.

## Public capabilities

`g.Environment.from_latlong(linear_rgb, size=256, samples=64)` consumes a
scene-linear `[height,width,3]` Torch tensor. Longitude is
`atan2(y,x)/(2*pi)+0.5`; latitude is `acos(z)/pi`, with image row zero at the north
pole. `from_cubemap` accepts `[6,size,size,3]` faces in `+X,-X,+Y,-Y,+Z,-Z`
order, Z-up except the two poles, whose up is +Y. `constant` is an analytic
test/convenience constructor. `load(bundle)` reads the raw linear environment
array from a portable gffx scene ZIP, without an image-decoding dependency.
`device='auto'` in scene/environment loading uses CUDA when available.

`g.ReflectionProbe(position, bounds=None, exclude=(), resolution=256, samples=64)`
describes a local six-view capture. Pass it in `g.render(..., probes=(probe,))`
to update on **every render call**. `snapshot = probe.capture(scene, ...)`
instead returns an owned snapshot; passing that snapshot reuses it until the
caller explicitly recaptures. `exclude` removes named meshes from capture
visibility. In automatic frame capture they still cast the frame's shared
shadows. `bounds=(minimum, maximum)` enables box-projected parallax for points
inside that box. Multiple captures currently select the nearest capture; they
do not blend through rooms or perform visibility-aware probe selection.

`g.PlanarReflection(mesh, point, normal)` captures the reflected camera for a
named flat mesh, with clipping to its physical front half-space. It updates on
every render and samples that image on the mirror surface. Camera motion and
offscreen geometry are included. The named material supplies reflectance; use
a metallic, near-zero-roughness surface. Current PBR roughness has a minimum of
0.001, which the shared fixture uses in both engines. A planar capture can use
the already captured cubemap; it contains no recursive planar captures. A
rough planar surface does not get a separately convolved planar image: use a
local probe for rough surfaces.

`g.IrradianceVolume.bake(scene, bounds=..., resolution=(3,3,3),
environment=world, exclude=..., ...)` is an **explicit static bake**. It renders
six views at every grid point, cosine-convolves their scene radiance into nine
SH coefficients, and uses trilinear interpolation. The captured image includes
directly lit surfaces and the visible environment: this supplies a secondary
surface bounce, rather than an arbitrary multi-bounce solution. Exclude moving
objects from a static bake. Changing a camera needs no rebake; changing baked
geometry, materials, lights or world does. Snapshot coefficients/bounds are
ordinary Torch tensors that callers can save and restore. Outside the volume,
diffuse lighting falls back to the unoccluded environment.

`g.render(..., integrator='raster', environment=world, probes=(probe,),
irradiance=bounce, mirrors=(mirror,))` retains explicit shaded subpixel AA and
existing lights/shadows. Request `outputs=('colour','mask','timings')` to get
synchronized preparation, reflection-update, primary-image and total seconds.
Image writing is outside these timings. Capture filtering and uploads belong
to update; shadow construction belongs to preparation. Keep the first image
separate from warm-frame distributions. Preparation caches are confined to
one owned frame/bake; no public mutable scene is cached across frames.

## Quality and approximation contract

Primary/capture visibility is rasterization. No screen-space trace is needed
for gffx's offscreen reflections. The CPU float32 and optional CUDA path share
C11 scalar image-lighting math; float64/unsupported native routes use a
separate Torch implementation. The CUDA helper uses the Driver API with
offline PTX, and the CPU helper does not load NVIDIA. Rendering is forward-only;
use the existing deferred integrator for its supported gradients. Image
lighting is not silently accepted by that different integrator.

Cubemap lookup is bilinear across face seams. GGX importance filtering uses
finite samples and mip levels, with a separately integrated 64x64 split-sum
BRDF table. This is single-scatter isotropic environment lighting, not complete
Blender Principled parity. Box parallax assumes a box-shaped distant scene;
nearby objects, occluded positions and nearest-probe transitions can be wrong.
Capture exclusions omit self-reflections. There are no recursive glossy
bounces. Finite maps can alias, and static snapshots become stale by design.

Diffuse SH loses high-frequency visibility, can ring and leak across walls,
and has no dynamic occlusion from objects excluded from the bake. The current
normal offset and clamped nonnegative irradiance are explicit approximations,
not physical conservation guarantees. The corner fixture keeps points in
known free space. General automatic probe placement/leak prevention is future
work. EEVEE uses its own probe filtering/volume bake plus screen tracing and AO-only rough-surface fallback; its
internal reflection-update subtime is not exposed by `bpy` and remains unmeasured.

Studio A's metal-rim/coat differences remain open. Neither passing controls nor
successful video encoding establishes full visual parity or human acceptance.
The reference campaign remains paused indefinitely, and publication requires
separate approval.

The independent world-direction fixture caught Blender's opposite longitude
sign. The reference author packs its panorama with U reflected so the physical
lighting matches gffx's documented `atan2(y,x)` convention. Main EEVEE controls
use AO-only Fast GI as the rough-surface fallback; disabling Fast GI forced
rough diffuse surfaces into the screen-trace path and produced an almost-black
indirect-only control in the installed5.0.1 setup. Close B08/B09 turn screen
tracing off to isolate the actual static diffuse bake. These controls show
colour transfer on the white sphere, and also expose a boundary difference:
gffx's world-space normal-offset sampling reaches the near-boundary receiver
plate, while the EEVEE probe extent falls back to the black world there.


Irradiance influence uses the **unbiased surface point** inside the closed authored
bounds. `normal_bias` offsets only the trilinear lookup, which clamps to the
sample grid; it never grows the volume. Outside points use environment diffuse
lighting (black if no environment). Probe leakage, fade/validity and visibility
weighting remain distinct limitations. The preserved Studio B8 boundary fixture
exposed an earlier influence-extension defect; the follow-up provides separate
interior and boundary comparisons rather than hiding it with moved bounds.


`IrradianceVolume.bake(..., interpolation="normal")` is an explicit optional
normalized positive-cosine-squared blend. It suppresses probes behind a receiver
and preserves constant-field energy, with coincident/empty-front fallbacks.
Defaults remain `interpolation="trilinear"`, including historical serialized
fields. Persist the selected mode with baked coefficients.

The Studio B audit found a tradeoff: normal-facing weights improve the plate
against EEVEE but worsen the interior sphere. The option is not adopted as a
universal fidelity fix or automatic default. It does not query geometry
visibility or add light bounces; those remain separate work. No exposure,
irradiance multiplier or light-power adjustment compensates for this gap.


### Capture exclusions and shadow ownership

Capture exclusions remove the selected geometry from both capture visibility
and that capture's shadow casters. They do not remove it from the primary scene.
Inline descriptors and explicitly saved snapshots use the same filtering.
The per-frame lighting cache shares maps only for identical scene/caster inputs,
shadow enablement and map size. An earlier shared-cache defect left ghost
shadows from excluded objects in inline captures; the B12 review preserves the
before/after controls and four CPU/CUDA regressions.

The corrected Studio B v5 fixture places the two main spheres exactly on the
floor; B8's saved geometry had a 10 mm gap. Existing A metal/coat mismatches and
finite-proxy reflection/GI differences remain. EEVEE 5.0.1's native planar
capture also has a distinct diffuse evaluation path for reflective closures:
an ordinary reflected-camera render retains chrome/glossy detail absent in
that capture. The gffx reference comparison labels this limitation instead of
imitating white-looking metal. B8 full clips, B9 bounds and B11 optional-GI
controls remain separate preserved evidence; B12 stills do not claim a
rerendered full animation or accepted equal-quality performance target.


### Exact-output capture-update optimization

The local B13 optimization retains B12 capture exclusions and map ownership.
A depth-only CUDA sample first tests the same signed closed-edge coverage rule
before performing double divisions. Covered samples retain the original depth
expression; no sampling, map size, filtering or BRDF parameter changes.
The per-frame cache evicts obsolete child lighting packets while an active
parent retains its own packet. Identical capture faces still reuse the same
maps. Different caster sets never share their maps, and snapshots remain
independently equivalent to inline capture.

The Studio B fixture still builds 128 maps for each of three caster sets.
Five corrected B12 PNG/linear controls and six Studio A PNGs remain exact;
218 controls pass. Separate cold-plus-three-warm measurements give 19.095 s
for B12 and 15.903 s for B13 at the same corrected view and quality. These
are local development measurements, not an equal-quality EEVEE parity claim.
Fresh paired sourcev5 animation results are recorded separately from the
historical B8 clips. Dynamic captures update every frame; diffuse irradiance
remains an explicit static bake. No new public switches are required.
