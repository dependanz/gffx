# Studio D: D01 glass reference

D01 adds a reusable `Material.dielectric` to the ordinary scene API. A scene
that assigns this material to a mesh uses a bounded geometric transmission
path when rendered with `integrator='raster'`. Scenes without a referenced
dielectric keep the existing opaque raster path.

```python
import gffx as g
from gffx.examples.studio_d import build_scene

scene, camera = build_scene(device='cuda', size=(192, 192))
result = g.render(scene, camera, integrator='raster', shadows=True,
                  supersample=1, background=(.035, .045, .065),
                  outputs=('colour', 'mask', 'transmission'))
g.io.write_png('studio_d_d01.png', result['colour'][0])
print(result['transmission'])
```

The short `examples/studio_d.py` entry point accepts `--device`, `--size`,
`--samples` and `--output`. The packaged authoring helper contains scene and
mesh construction; the optical algorithms live in `gffx.torch`.

## Material and geometry

```python
glass = g.Material.dielectric(
    'glass', ior=1.5, absorption=(.12, .025, .008), device='cuda')
```

`ior` is a finite scalar in `[1,4]`. `absorption` is a finite, nonnegative RGB
coefficient in inverse world units. A path of length `d` inside the material
multiplies radiance by `exp(-absorption * d)`. D01 interprets one world unit
as one metre. The air cavity of a hollow object contributes no glass absorption.
The material has polished interfaces; rough glass, normal maps and textures
are outside this first implementation.

Glass meshes must have closed, consistently oriented boundaries with outward
winding. The authored tumbler includes its outer wall, rim, inner wall,
inner bottom and outer bottom. The empty interior is connected to exterior air
through the mouth. A horizontal ray crosses four interfaces; a ray through
the mouth and bottom crosses two.

The camera, ordinary opaque surfaces and emitters must be in air. One occupied
glass solid is tracked at a time. Entering another solid before leaving the
current one, encountering an unmatched exit, and ending a path at opaque
geometry inside glass raise explicit errors. Edge closure and signed volume
checks do not prove that arbitrary meshes have no self-intersections; disjoint,
non-nested solids remain the admitted domain.

Portable `gffx-scene-1` bundles recognize material `kind: "dielectric"` with
`ior` and `absorption` parameters. `write_bundle` in the D01 example exports
the fixture's constant materials, geometry, lights and camera for later
reference rendering. Geometry arrays use the format's existing float32 storage.

## Transport and sampling

Every reflected or refracted ray queries the actual triangles along its new
direction. The implementation uses exact dielectric Fresnel, Snell refraction
and total internal reflection, with a separate object-medium state for each
branch. Reflection and transmission weights sum to one before absorption.
Radiance IOR factors cancel on completed paths whose two endpoints are in air;
the implementation's endpoint restrictions are essential to that convention.
The Fresnel and radiance conventions follow the
[PBRT dielectric model](https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF).

The optical normal is the geometric triangle normal. Curved objects therefore
retain the facets of their authored meshes. Supersampling uses the existing
grid positions and averages fully shaded contributions. `mask` reports primary
geometric coverage, including glass; it is not material opacity. Optional
`depth` and `face_index` describe the centre primary ray.

Direct illumination uses a separate **straight-ray Fresnel/Beer shadow
approximation**. Shadow rays do not refract toward emitters. This supplies
colored attenuation, but can over-darken oblique configurations and cannot
serve as a refracted-direct-light or caustic reference.

## Explicit bounds and initial cost

- At most **1024 interface events** per camera path.
- Branches below **1e-4** maximum RGB path weight are discarded.
- Active branches share a block capacity of
  `max(256, primary_rays_in_block * 32)`.
- At most **32 glass interfaces** along a straight shadow segment.
- Triangle queries use bounded chunks and have **O(rays * triangles)** cost.
  There is no acceleration hierarchy in this first reference path.

The `transmission` output records these limits, query counts, TIR events,
truncation counts and per-pixel discarded RGB path weights. Discarded weight
is a transport diagnostic, not a universal image-error bound. Long trapped
paths can retain nonzero weight at the depth limit. A scale-dependent positive
query threshold prevents immediate self-hits; it is not a proof of watertight
intersection at arbitrary scale.

The initial 48-event limit cut off a real tumbler path before its exit at
interface 52. A focused one-pixel regression now retains that contribution.
The higher bound is still finite and does not establish convergence for
arbitrary scenes or camera views.

CUDA transport batches contain up to **2048 primary rays**, while camera rays
are still generated in the original groups of at most **256**. Keeping those
camera groups preserves their matrix-multiplication rounding before the rays
are combined for transport. CPU generation and transport remain at at most
**64** rays. Triangle face chunks remain **256**; these batch changes do not
alter the optical bounds above.

`fragment_block` remains an upper bound on the transport batch. Each batch
contains whole original camera groups, so a CUDA value of `300` uses `256`
transport rays, and `600` uses `512`. Values below `256` retain their original
smaller camera and transport groups. The final image group can be shorter.
Centre-depth calculations retain the same original camera groups as well.
The `transmission` output exposes `primary_ray_generation_chunk`,
`transport_ray_chunk` and `query_face_chunk` so validation can check the actual
batch policy. Larger transport groups share a larger active-path capacity
under the existing per-primary formula; unchanged truncation diagnostics are
part of the batching validation.

For glass scenes, the initial surface support is constant diffuse, glossy,
PBR and emissive materials, ordinary sampled lights and optional environment
lighting. Textures, subsurface, reflection probes, planar mirrors and
irradiance volumes are explicitly rejected. The forward-only behavior remains
explicit; implicit requests for an autograd graph raise an error.

## Experimental directional plane caustics

The default dielectric path above is unchanged. An explicit `caustics`
configuration adds a bounded photon map for one selected horizontal receiver:

```python
scene, camera = build_scene(device='cuda')
receivers = [p.name for p in scene.prims
             if isinstance(p, g.Mesh) and p.name.startswith('tile_')]
result = g.render(
    scene, camera, integrator='raster', shadows=True,
    background=(.035, .045, .065),
    caustics=dict(receivers=receivers, photons=65536, seed=1701,
                  cell_size=.02, kernel_size=3, max_interfaces=256),
    outputs=('colour', 'mask', 'timings', 'transmission', 'caustics'))
```

Receiver meshes must tile one complete axis-aligned rectangle, with constant
height and +Y geometric and shading normals. Each mesh must be a two-triangle
rectangle. Holes, overlaps, nonplanar geometry and a nonintegral grid extent
are rejected. Materials remain the existing constant PBR or classic BRDFs;
photons store incoming directions and power before any receiver material is
evaluated. Thus neighbouring checkerboard tiles do not exchange pre-shaded
colour, and the same incident photon can produce different PBR responses for
different endpoint materials or view directions.

On those receiver faces, direct-light shadow queries treat glass as opaque.
A separate term includes only photons whose first intersection is glass, so
straight transmitted light is not counted twice. Other surfaces retain the
previous straight-segment Fresnel/Beer shadow approximation. The map is sampled
at every camera-path receiver endpoint, including floor points reached after
reflection or refraction; it is not a screen-space overlay.

The first implementation admits explicit rectangular lights only, with no
environment lighting or diffuse continuation. Background miss radiance remains
available. It reuses the accepted light quadrature and samples one conservative
bound containing all glass objects for each emitter sample. A source inside
that bound, or a cap crossing the emitter horizon, uses a full emitting
hemisphere proposal. Equal packet counts per source sample and the actual
proposal density determine each packet's power; missing the glass does not
change the denominator. A glass owner is tracked across every interface so
disjoint solids can have different refractive indices and RGB absorption.
The hollow tumbler's cavity remains air. Unsupported nested/intersecting media
or an emitter actually inside glass are rejected when encountered.

The estimator divides gathered photon power by physical cell area and the
fixed kernel area, evaluates the existing BRDF with the stored incoming
light direction and current view, and does **not** multiply another receiving
cosine. The grid covers the whole receiver, independently of camera crop.
The `.02` metre D01 setting gives a 150 by 120 grid. The actual X/Z cell
widths account for the authored float32 endpoints and are recorded. Kernel
sizes 1 and 3 are supported. A kernel that extends beyond the finite floor
loses that portion of its mass; the loss is reported rather than redistributed.
Finite cells, fixed filtering and finite photon counts remain sources of bias
and sampling error. This is not a convergence or visual-acceptance claim.

Photon intersections, positions, propagation directions and optical interface
state use **float64** on CPU and CUDA, while the accepted camera path keeps its
authored precision. This separate reference-precision policy preserves real
short flights at a glass rim: a captured D01 path has a 0.135 micrometre flight
to an inner-wall TIR event, below the float32 self-hit threshold. The same
accepted triangle-query/interface code and strict medium guards are retained;
no ray is discarded or assigned a different medium to hide that case. Stored
receiver directions remain float64 in the evidence and are cast to the
receiver material's dtype only for BRDF evaluation. The wider photon state
has an unmeasured native performance cost.

Photon queries use bounded 2048-ray by 256-face chunks, independently of the
accepted camera-generation/transport batches. Per-endpoint
BRDF gathers are split into at most 65,536 photon pairs; all pairs are evaluated.
There is no acceleration hierarchy, and the map is rebuilt for each render.
There is no claim of real-time performance or a general caustic renderer.

Compact diagnostics are under `result['transmission']['caustics']`. The optional
`result['caustics']` contains a JSON-safe `metadata` dictionary and named CPU
`tensors` for launch samples, terminal accounts, optical depths, directional
receiver packets and the raw/filtered power grids. Packet flux is already
normalized by the total launch count. The metadata accounts for initial
misses, receiver power, escape, other surfaces, absorption and paths that reach
the interface limit. A small residual is a power-accounting observation, not a
universal image-error bound. Existing camera truncation diagnostics remain
separate. Omitting the configuration bypasses all photon setup; requesting
caustic evidence with no configuration raises an explicit error.

## D01 validation and development artifact

Focused tests cover independent Fresnel/Snell/TIR values, plane-parallel slab
displacement and multiple reflections, absorption through two walls with an
air gap, real authored-tumbler controls, medium/topology rejection, float32 and
float64, and portable scene round trips. The Windows runner also compares an
opaque control byte-for-byte against the original candidate16 runtime and
checks a small closed-slab image on both CPU and CUDA.

Batching controls compare the original and regrouped primary rays exactly for
perspective and orthographic cameras in float32 and float64. Public render
controls compare colour, coverage, centre depth, primary faces and transport
diagnostics across batch limits, including supersampling and a limit that is
not a multiple of the original camera group. Native full-frame comparison
against the saved D01 image remains a separate required validation step.

The first preview is a **192x192, one-sample-per-pixel still**. Its coarse mesh
facets and sampling are part of this initial fixture. Visual acceptance,
Blender/EEVEE comparison, a short orbit and performance work follow after the
native CPU/CUDA evidence has been reviewed.

The development runner creates a complete new source copy and a separate
candidate16 runtime with the reviewed Python source overlay. It does not build
a new wheel. The imported base distribution still reports `studio3.16`; the
candidate manifest and source hashes identify the D01 changes. Runtime
dependencies remain Python and PyTorch. Native binaries are reused unchanged.
