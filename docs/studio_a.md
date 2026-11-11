# Studio A: forward raster development comparison

The experimental `integrator="raster"` path targets high-quality forward images.
It uses gffx raster visibility, sampled direct lighting, shaded subpixel
antialiasing and receiver-plane filtered shadow maps. It is rasterization, with
no ray tracer or external rendering library in the gffx runtime. Blender EEVEE
is used separately to produce the comparison images.

```python
import gffx as g

scene = g.Scene.load("studio-a.gffx")
image = g.render(scene, "camera", integrator="raster", supersample=4,
                 shadows=True, shadow_map_size=2048)
g.io.write_png("studio_a.png", image["colour"])
```

The shared bundle contains evaluated triangles, split corner normals, UVs,
materials, textures, lights and a camera. Loading selects an available device at
setup; an operation never silently falls back from CUDA to CPU. The example
renders the bundle's initial light pose. The comparison runner applies the
saved frame-36 pose for stills and all 72 poses for the moving-light clip.
The development wheel is local; these additions are not a PyPI release.

`Material.pbr` exposes linear RGB base colour, perceptual roughness, metallic,
IOR, coat and coat roughness, plus optional `TextureSource` bindings and a
tangent-space OpenGL normal map. This is conventional single-scatter isotropic
GGX with correlated Smith visibility and Schlick Fresnel. It is separate from
the existing empirical glossy and certified reciprocal-matte `principled`
surface; its results do not inherit their evidence or support claims.

`Light.point` takes total isotropic radiant power in watts; `Light.spot` clips
that point intensity by its cone and blend, without redistributing the clipped
power; `Light.sun` takes irradiance in watts per square metre and an angular
diameter in degrees. Existing `Light.rectangle` takes radiance and full-edge
span vectors. Radiometric inputs are authored rendering parameters, not a
measurement of a real lamp. Blender unit parity and the differences between
its Principled model and this GGX model remain comparison questions.

The path is explicitly forward-only. Requested gradients in geometry, camera,
attributes, textures, material, emission or lights raise an error unless the
caller explicitly enters `torch.no_grad()`. Use the existing differentiable
deferred integrator for its supported gradient paths. The raster path does not
silently imply differentiable silhouettes or shadow visibility.

Shading runs in material-compacted fragment blocks. The optional private native
helpers fuse forward shading and use sorted conservative tile lists for CUDA
hard rasterization. The lists retain ascending face order and the existing
double edge / stored-float depth convention. CPU builds use a C11 helper with no
NVIDIA dependency. CUDA builds use a separate Driver-API library with offline
PTX; end users need no CUDA toolkit or runtime compiler. Float64 and missing
helper configurations use the existing tensor/native composition. The private
helper ABI is not a new supported public C API.

`supersample=4` shades sixteen independent subpixels. Colour and mask are
averaged; requested depth and face indices refer to the pixel centre. Point
shadow maps cover six directions, and crossing caster triangles are clipped.
Perspective maps preserve relative precision with scaled reciprocal depth;
orthographic maps use ordinary depth. Rectangle maps currently use a 140-degree
projection for this fixture: arbitrary full emitting-hemisphere coverage and
general primary near-plane clipping remain open. The path is one camera/frame
per call and does not yet cover GI, environment reflections, transmission,
volumes, motion blur or depth of field.

The portable `gffx-scene-1` ZIP contains `scene.json` and typed little-endian
float32/int32 arrays. UVs use a lower-left origin. Texture arrays declare
`origin: "top-left"` (default for decoded images) or `"bottom-left"`; the loader
orients rows before mip generation. Normal RGB channels are non-colour data;
the orientation correction does not invert the green channel. Bundle parsing
does not extract executable files or execute scripts.

The developer tools under `tools/studio_a` build/export the Blender fixture,
render both engines, verify frame hashes, encode previews and compare saved
linear/display images. `supervise.py` writes live progress, sampled total-device
VRAM, process RSS and terminal receipts. Render inputs and output generations
are identified separately; verified frames can be reused without being counted
as newly rendered. Preview H.264 compression is separate from the preserved
linear HDR and lossless PNG evidence. These observations establish development
comparisons, not visually accepted EEVEE parity or a performance guarantee.


### Continuous rectangle highlights and preparation reuse

The forward raster PBR path integrates rectangle shape with horizon-clipped linearly transformed cosines (LTC). This removes the grid of isolated specular point sources while retaining the authored radiance, surface roughness and geometry. Base and coat lobes use the GGX fit separately. Point/sun BRDF evaluation, the empirical classic material and the differentiable deferred path retain their existing evaluation methods. The LTC fit and shared visibility factor are approximations; this is not full Blender Principled/OpenPBR parity.

Area shadow visibility still uses every authored quadrature position and the same receiver-plane PCF taps. One cosine-weighted visibility factor modulates the continuous lobes, so partial specular occlusion is approximate. Depth maps share a slab; light metadata is packed in bulk; immutable shadow camera matrices and the coplanar rectangle caster preparation are reused within a render. No public mutable camera or cross-frame geometry cache is introduced. The final CUDA shadow path builds depth-only tile bins with device count/scan/fill operations. It omits face-order sorting and primary face/barycentric output buffers while preserving tested coverage/depth values. Primary visibility retains the original sorted binned path. Measured slower triangle-driven prototypes are preserved separately.

Spot blend is specified in cosine space: outer = cos(angle/2), inner = outer + blend*(1-outer), followed by smoothstep. This agrees with the Blender spotlight convention; it does not redistribute total power into the cone.

The 128 KiB GGX lookup is from [Heitz, Dupuy, Hill and Neubelt, Real-Time Polygonal-Light Shading with Linearly Transformed Cosines (2016)](https://eheitzresearch.wordpress.com/415-2/) and the [authors' reference data](https://github.com/selfshadow/ltc_code). Its permissive redistribution license is bundled as gffx/torch/_ltc_LICENSE.txt beside the table. The polygon clipping/evaluation and gffx integration are implemented within gffx; no additional renderer or runtime compilation is required.

Studio A remains a measured development comparison. Remaining bright-rim/coat differences, reconstruction-filter differences, partial-occlusion approximations, general primary clipping and rectangle shadow hemisphere coverage are explicit limitations. See the paired dashboard and evidence instead of interpreting a test count as visual acceptance.
