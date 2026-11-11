# Simple scenes (0.2.1 baseline and 0.2.2 preview API)

The 0.2.1 API provides scene construction, a camera, rendering and
PNG export. These additions are **not in the older 0.2.0 wheel**. `pip install gffx` installs a compatible Torch runtime too;
`python -m gffx.examples.cornell_box` runs the example included in the wheel.
The complete example is [examples/cornell_box.py](../examples/cornell_box.py):
one `import gffx`, scene declarations, a render call and a save call. Users do
not write a sphere generator, shadow mapper, shader or image encoder.

```python
import gffx as g

scene = g.Scene([
    g.Material.diffuse('white', (.75, .75, .75)),
    g.Mesh.quad('floor', u=(3, 0, 0), v=(0, 0, -3), material='white'),
    g.Mesh.sphere('ball', radius=.5, center=(0, .5, 0), material='white'),
    g.Light.rectangle('ceiling', center=(0, 2, 0),
                      u=(.7, 0, 0), v=(0, 0, .7), radiance=(15, 15, 15)),
]).to('auto')
camera = g.Camera.look_at((2, 1.5, 3), (0, .5, 0),
                          size=(512, 512), device='auto')
image = g.render(scene, camera, shadows=True, ambient=.08)
g.io.write_png('render.png', image['colour'])
```

## Construction and device selection

`Mesh.quad` takes a center and two full-width spans. Their cross product sets
the front normal. `Mesh.sphere` creates a closed smooth sphere; `segments` and
`rings` control tessellation. Material names bind meshes to scene materials.
`Material.diffuse` accepts scalar or RGB reflectance in [0,1].
`Material.emissive` accepts nonnegative scene-linear RGB radiance.

`Light.rectangle` uses the same span convention and emits toward `u × v`.
It provides its own visible surface by default (`visible=False` hides only the
surface). `samples=4` means 16 deterministic quadrature samples. A parent or
transform affects both the illumination and the visible surface.

Constructors accept Python values or floating tensors. Tensor parameters keep
their gradient connections. `.to(device, dtype=...)` explicitly creates a new
scene on the requested device/precision and keeps those connections; it does
not mutate the original. Known dataclass sources such as textures and nested
containers transfer their tensors too. Opaque callback payloads and device-qualified
material budgets retain their own contracts and must be constructed for the
target device. Default construction uses CPU float32, unless input
tensors supply a precision/device. Cameras use the same explicit selection. The opt-in `auto` device selects
CUDA only if both Torch CUDA and the gffx provider are usable, otherwise CPU.
This setup probe may load the driver; ordinary operations never fall back.
`Camera.look_at(eye, target, size=(height,width), fov=...)` creates a single view
and handles its batch dimension. FOV is the vertical angle in degrees.

## Rendering and differentiation

The existing `gffx.render(scene, camera)` remains the renderer entry point and
returns named tensors (`colour` is linear RGB; `mask` is coverage). Low-level
scene records and `gffx.torch` calls remain available. The constructors and
top-level class names select the current PyTorch adapter lazily; plain
`import gffx` remains dependency-free and does not probe a GPU.

Rectangle-light sampling, visible emitters and shadow handling are renderer
responsibilities. They compose the existing native camera/raster/geometry and
material/light interfaces. There is no Cornell-specific renderer or primitive
exception. `ambient` is explicit uniform diffuse environment radiance. This is
direct lighting, not multi-bounce global illumination.

Hard raster and shadow-map visibility are discrete. Shadow depth comparisons
are detached; smooth camera, geometry, albedo, emission and light paths retain
first derivatives where defined. This does not add silhouette gradients or
differentiate through occlusion changes. Shadows use approximate perspective
maps with receiver-plane PCF; near-plane whole-triangle clipping and map-cone
coverage remain limitations. Directional/custom-light shadow maps are not
implemented. This is a development feature, not a new stable renderer, GPU
fleet, compiled/exported, higher-order or performance claim.

The renderer keeps a declared workspace budget. Without an explicit override,
unshadowed CPU retains its 64 MiB allowance; the new opt-in shadowed CPU
compositor uses 512 MiB. Explicit caller budgets remain authoritative. CUDA uses min(1 GiB, one quarter of device
capacity), without sampling free memory or changing scheduling. All input,
image, lighting and shadow-map estimates still fail before allocation when the
requested workload exceeds the budget. Use `workspace_limit_bytes` for an
explicit budget, and `shadow_map_size` for advanced shadow-map resolution.

## Display previews (next-release development API)

The next-release source checkout adds `gffx.image.to_display` to make
notebook previews match `gffx.io.write_png`. This API is **not** in the
published Windows-only 0.2.1 artifact. Render output remains scene-linear RGB
(and may exceed 1.0), while display images are explicitly converted to sRGB:

```python
import gffx as g
import matplotlib.pyplot as plt

rendered = g.render(scene, camera, shadows=True, ambient=.08)
linear_rgb = rendered['colour']         # scene-linear, retain for physical math / gradients
display_rgb = g.image.to_display(linear_rgb, tone_map='aces', exposure=0)
plt.imshow(display_rgb[0].detach().cpu().numpy())
plt.axis('off')
plt.show()
g.io.write_png('render.png', linear_rgb, tone_map='aces', exposure=0)
```

Both functions use the same exposure-in-stops, ACES-style rational tone mapper
(or `tone_map='clip'`) and linear-to-sRGB transfer. The preview is a
float32/float64 tensor in [0,1], on the input device with a first-order gradient
where the mapping is smooth; the PNG differs only by rounding each channel to
8 bits. The PNG writer detaches deliberately, never mutating the input.

Do **not** pass `display_rgb` to `write_png`: that would tone-map and encode
already display-ready data a second time. For unmodified linear/HDR interchange,
retain the original tensor or use a suitable scene-linear/HDR format in a future
exporter, rather than the 8-bit sRGB PNG path.

`to_display` accepts an HWC or BHWC floating RGB tensor (three channels,
nonnegative finite values) and `tone_map='aces'|'clip'`, `exposure=<finite
number>`. Like the native image-transfer operation used by PNG export,
CUDA tensors require a usable GFFX CUDA provider; it does not silently move
GPU data to CPU. Merely importing `gffx.image` remains inert.

## Image export and dependencies

`gffx.io.write_png(path, image, tone_map='aces', exposure=0)` exports linear RGB
as an sRGB PNG using the standard library and the existing image transfer
operation. It accepts HWC or a one-image BHWC tensor, explicitly detaches and
transfers for file output, and preserves the source tensor/gradient graph.
Existing files are refused unless `overwrite=True`; `tone_map='clip'` omits
the display curve. No NumPy, Pillow or external renderer is needed. PyTorch (`torch>=2.10.0,<=2.14.1`) is the one
direct Python runtime dependency in 0.2.1; its CPU/CUDA variant follows the
[installation guide](INSTALLATION.md). The core and bare import remain lazy.
