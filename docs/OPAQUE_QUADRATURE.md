# Native quadrature for nonlinear opaque scene composition

The C11 primitive computes visible polygons with affine depth, applies caller
material cut numerators, triangulates each remaining polygon and emits tensor
Gauss5 Duffy nodes. Each node carries original face identity, three SCREEN
barycentrics and its pixel area weight. Shade at each node with perspective
attributes, then accumulate radiance times area; never interpolate corner BRDF
answers as a substitute for nonlinear integration.

`opaque_quadrature(vertices,faces,cuts,height,width,origin=(0,0))` returns
owners[P,800] and samples[P,800,4], P=height*width. Unused owners are -1 and
samples zero. A material cut has corner values `(attribute-threshold)/clip_w`;
its zero set is a straight line even for perspective attributes. Both sides
are retained with disjoint interiors. At most16 cuts per face,32 conservative
candidate faces and32 final triangles per pixel; overflow invalidates all
outputs, never truncates geometry. Original face order wins exact depth ties.

The VJP differentiates geometry, affine-depth intersections, cut positions,
node barycentrics and area weights through the same clipping arithmetic.
First derivatives follow the selected topology. Branch transitions and depth
ties need not have an ordinary derivative; higher derivatives are unsupported.
All scratch is caller-owned and exposed through the C workspace query; C outputs
are preallocated. CPU and CUDA float32/float64 share the scalar-dual arithmetic. CUDA validates
indices and finite values on device, uses caller scratch and one ordered worker;
only a status word returns to the host.
Python/PyTorch and the lazy facade expose the same operation. Workspace/output
admission stays within64MiB, with1GiB enclosing CPU process and120/180s deadlines.

This is a quadrature GEOMETRY primitive. It cannot certify the integration error
of arbitrary shaders it does not receive. Its Gauss5 rule is exact through the
corresponding tensor polynomial degree in Duffy coordinates, apart from native
rounding. Nonlinear shader image/gradient accuracy requires independent bounded
reference qualification. `coverage="opaque"` composes bounded CPU/CUDA float64
Principled/directional scene windows using these nodes. There is no general
shader, dense-scale, full material-transition or M1 completion claim from the primitive.

Reference evidence lives in opaque56: exact moving material cuts, normalized
bilinear clamp/repeat cells, perspective UV boundary transport, independent
90-digit Principled formulas, and separately enclosed spatial-roughness RGB and
first derivatives. Preserve the failed material-branch trials. The roughness
certificate uses centred derivative enclosures and a seventh-order Taylor
remainder, not successive sampled-grid agreement. Stored-resource integration,
physical loss-table approximation and native floating rounding are distinct.

## Scene composition qualified in opaque56

The explicit opaque colour path projects scene geometry, computes depth-ordered
visible regions, splits perspective UV texel-centre and roughness-resource-node
isolines, reconstructs perspective attributes at native nodes, and shades those
nodes with the existing installed Principled and directional-light operations.
Area-weighted accumulation gives RGB and coverage; uncovered area receives the
optional background. Geometry, depth, material cuts, barycentrics, weights,
attributes, camera and lighting remain on the first-derivative path.

The exposed profile is CPU float64, constant or explicit attribute material fields,
and bilinear level-zero albedo textures with clamp/repeat addressing. Other shader
or light kinds, texture modes and partial camera-plane/depth clipping refuse.
The planner charges all800 slots per pixel through the existing scene/material
allocation model, plus native scratch. An8x8 textured fixture correctly refuses
188662888 requested bytes against66060288 available after the1MiB reserve.
A4x4 fixture fits. This conservative whole-image path is not dense admission;
a bounded streaming implementation remains necessary for the original mesh.

Five complete scene tests compare one nonlinear pixel with the independent
outward-rounded reference, compare either square diagonal with texture/roughness,
and check a combined derivative. Eight additional independently perturbed
parameter directions check geometry,UV,texels,roughness,F0,light,normal and camera
with two finite differences each. The latter corroborate implementation
connectivity; they do not prove a global quadrature bound. Native geometry also
has exact affine/depth/cut comparisons and full selected-topology gradcheck.

Fixed Gauss5 integration is not adaptively certified for every scene accepted by
the experimental API. Qualified fixtures are explicitly limited: arbitrary
angular knots, limiter switches, horizons, sharp roughness or texture variation,
all topology transitions, batch/dtype/CUDA/tile surfaces and dense performance
remain gates. Do not label arbitrary accepted input numerically qualified merely
because it ran. Reference integration excludes native rounding and physical
loss-table approximation. Historical hard/soft rendering remains unchanged.

## Bounded streaming and device qualification (opaque57)

Opaque colour rendering now uses row-major windows in original camera coordinates.
`opaque_tile_pixels=None` selects the largest width admitted by the conservative
planner; a positive integer supplies an upper bound. The64MiB total allowance
charges output/cotangent reserves, input adjoints and active window scratch.
It refuses if even one pixel or the retained outputs do not fit. Mask-only calls
reject this colour-only option. Existing hard/soft rendering is unchanged.

Forward retains input tensors, not a graph per window. Backward replays one pixel
at a time in fixed ascending row-major order for every forward tile cap. Input
version checks detect mutation. The first VJP remains connected to geometry,
materials, textures, camera and lights; higher derivatives still refuse.
Three-term perspective normalization and its first VJP have explicit arithmetic
order, repairing the preserved CPU/CUDA gradient discrepancy without a tolerance
change. The shared native quadrature uses no floating atomics or CPU geometry copy.

Finite checks cover an8x8 textured/roughness scene formerly refused by resident
planning, batch/depth/material/subpixel windows, partition-independent VJPs,
normalization gradcheck and individual parameter directions. Device fixtures cover
float32/64 geometry, nondefault streams, invalid device data, and1x1/2x2 connected
float64 scenes. CUDA scene capture rejects device-to-host tensor copies during
render/backward and measures incremental tensor memory separately from host RSS.
These checks do not admit dense-frame performance or arbitrary shader accuracy.

Broader independent pixel/flux fixtures cover constant roughness0.15,0.45,0.7,0.95,
F0 zero/one and intermediate values, two-sided normal orientation and the unlit
branch. Exact moving partitions and outward intervals enclose both RGB and first
boundary derivatives within the unchanged1e-12 reference target; a separate90-digit
formula and installed quadrature provide independent cross-checks. Arbitrary
spatial angular/limiter/horizon crossings, broader varying-roughness intervals,
other texture filters, clipping and topology transitions still require their own
certificate or explicit refusal. Fixed Gauss5 nodes alone do not supply one.


## External storage qualification boundary

`tools/opaque_frame_store.py` is a development storage component. It exclusively
creates a new directory,appends RGB/mask and their cotangents in canonical order,
seals only a complete frame,and verifies pinned identity,manifest and each block
before replay. Payload is limited to512 pixels/32768 bytes; metadata is limited
to1MiB. It uses ordinary bounded file reads,not full-frame tensors or mmap.
Full1080p synthetic storage uses132710400 payload bytes on internal disk.
A separate3x3 installed textured/varying-roughness scene test preserves RGB,mask
and ordered vertex VJP exactly against each device's resident baseline.
These are storage/composition checks,not a public API change or dense rendering.

The resident opaque renderer still refuses its200179712-byte1080p reserve under
64MiB. Its original-mesh32/64-square startup frames also exceed32 candidate
triangles: maxima68/44 at56/18 pixels respectively. No cap is increased.
There are no two matching complete original opaque frames for the approved D297
cost forecast. Crops,other scenes,synthetic storage timing and old hard/soft
frames cannot replace those anchors. Full opaque frame accuracy,cost and the
production external-storage boundary remain separate open gates.


## Production horizon and external-window integration (opaque60)

Production scenes now partition exact linear view horizons for geometric,
uniform or constant normal domains, and directional-light horizons for linear
normal fields. A varying-normal/view dot product is quadratic in perspective
barycentrics. Outward-padded Bernstein coefficient signs can exclude a crossing;
an unresolved crossing refuses. Equal vertex samples do not prove a constant
field under differentiation. Grazing cases refuse and the shared16-cut cap stays.
The separate independent horizon fixture validates colour and first motion
rather than interpreting a CPU/CUDA match as an accuracy certificate.

Camera projection uses tensor image-dimension denominators to avoid a backend
scalar reciprocal shortcut. RGB integration uses an explicit three-channel
coverage-weight VJP. Captured3x3 projection, quadrature, shading and all retained
adjoints now have matching CPU/CUDA bytes. The moving-knot caller separately
needs explicit broadcast/channel/sample reductions; the old harness remains
historical numerical evidence. This does not promise arbitrary PyTorch caller
expressions or arbitrary camera inversion/matrix products are byte identical.

`gffx.render.opaque_window(scene,camera,(y,x,height,width),...)` and the PyTorch
surface expose the same pure global-camera window operation. The output reserve
covers this rectangle; input-adjoint and material work remain charged. A512-scalar
per-packed-face reserve additionally accounts for horizon/material cut fields,
retained operands, coefficient/error/selection temporaries and adjoints. Unfit
work refuses. Ordinary full-frame rendering still charges its entire output.
There is no renderer file I/O, geometry cache or implicit ambient-VRAM admission.

`tools/opaque_frame_execution.py` is the host integration driver, not a core
runtime dependency. It uses exclusive internal disk files,256-pixel blocks
(16384 payload bytes, below32768 even during byte-buffer conversion), bounded
manifest metadata, complete sealing, pinned input bytes, mutation checks and a
shared120-second forward/replay deadline. Backward visits each pixel in canonical
order, verifies replayed image bytes and consumes all block/hash checks before
returning gradients. Its current replay receipt binds live input objects within
the same process; cross-process resume is not claimed. Batch1/float64 and fixed
cotangents define this finite host profile. Larger complete frames still need
matching cost anchors, capacity admission and independent numerical evidence.
