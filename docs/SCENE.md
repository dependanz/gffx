# Minimal scenes

`import gffx` stays dependency-free. Accessing `gffx.scene` selects the currently
implemented PyTorch adapter (PyTorch2.10+, CPU/CUDA, float32/float64). This is an
eager scene composition, not a new native renderer. No new external dependency.
API tier: development-preview eager PyTorch composition. Local CPU/CUDA tests
qualify the recorded profiles; release/hosted support-matrix publication is pending.

```python
import torch
import gffx

s = gffx.scene
vertices = torch.tensor([[-.5, -.5, .2], [.5, -.5, .2], [0., .5, .2]])
faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
parent = torch.eye(4)
parent[0, 3] = .1
world = s.Scene([
    s.Rig('group', transform=parent),
    s.Mesh('triangle', vertices, faces, parent='group', attributes={
        'colour': s.Attribute(torch.tensor([.2, .6, 1.]), 'constant'),
    }),
])
packed = world.flatten()
fragments = gffx.torch.render.rasterize(
    packed.vertices, packed.faces, 64, 64,
    cull_mode=gffx.torch.render.CULL_NONE,
    vertex_offsets=packed.vertex_offsets, face_offsets=packed.face_offsets,
)
colours = gffx.torch.render.interpolate(
    fragments[0], fragments[1], packed.attributes['colour'].values,
)
```

The example authors coordinates directly in NDC for brevity. General world-space
scenes need an explicit camera projection before rasterization. Flatten itself
does not project, shade, skin, simulate or render anything.

## Objects and bindings

Mesh, Material, Light, Camera and Rig have unique names, optional parent names and
local transforms. `Material(name, evaluator, parameters)` and
`Light(name, evaluator, parameters)` retain existing public shading callables and
parameter mappings; `Mesh(..., material='name')` refers to the material.
`Camera(name, camera)` retains an existing `gffx.torch.camera.Camera`.
`packed.world_matrices` gives explicit world transforms for all prims; integrators
apply camera/light coordinate semantics rather than having payloads silently rewritten.
Rig is a transform group, not an implementation of skeletal deformation.

## Shapes and packing

Vertices are `[V,3]` or `[B,V,3]`; faces are shared `[F,3]` int32 local indices.
Local transforms are `[4,4]` or `[B,4,4]`, affine, finite and invertible with
representable nonzero determinants in the selected precision. Parent matrices
multiply on the left. Singleton batches broadcast; incompatible batches fail.
For a repeated static scene, use `flatten(batch_size=B)`.

Packing is frame-major, then insertion-order meshes. Faces are globally rebased;
vertex/face offsets `[B+1]` separate frames, not objects. Per-frame object offsets
`[B,M+1]`, mesh_names and face_material preserve object and material ownership.
Material index -1 means no binding. Shared input topology is duplicated and rebased
in packed outputs; do not assume zero-copy or zero memory growth with frame count.

Attribute domains: constant `[C]`, uniform `[F,C]`, vertex `[V,C]`, face_varying
`[F,3,C]`, optionally with leading B. Reserved uv/colour/normals have2/3/3 channels.
Packed attributes expose corner `values`, a per-face `present` mask and authored
`domains` by mesh. Missing data are zero with false presence: consumers must choose
their own defaults. Normals use normalized, winding-aware inverse-transpose transport;
other attributes retain authored values. UV seams survive face-corner packing.

## Ownership, errors and gradients

Scene objects snapshot container membership, but borrow tensor storage and opaque
evaluator payloads. Every flatten revalidates data; do not mutate borrowed tensors
during calls or while an autograd graph still uses them. Packed geometry/attributes
have fresh storage and connected gradients. No persistent transform/geometry cache.
No implicit dtype/device conversions, GPU-to-CPU validation fallback or tensor detach.
Use explicit data movement before construction. Empty scenes default to CPU float32
unless dtype/device are specified. CPU/CUDA data must agree across geometry and
transform tensors; evaluator payload semantics remain the evaluator's responsibility.

TypeError reports wrong types/precision; ValueError reports malformed shapes,
references, cycles, incompatible devices/batches or nonrepresentable arithmetic.
Packed index overflow rejects. Discrete topology/IDs are not differentiable.
Smooth geometry, attributes and transform chains are; hard visibility changes
retain the existing rasterizer's gradient limitations. Compilation/export and
native scene APIs are not admitted profiles. CPU mesh.validate remains CPU-only.

## Scope and evidence

One in-memory container with hierarchical transforms; no composition layers,
instancing, external scene references, variants, file formats or topology animation.
Tests are `tests/python/test_scene.py` and `test_scene_integration.py`; the finite
dashboard demonstration uses the public API. Synthetic5023-vertex/9976-face budget
fixtures use no licensed FLAME data. Exact build/results belong to devbrain NUMBERS;
local tests do not establish a published support matrix or real-time capability.

## Simple authoring API (unreleased 0.2.1.dev0)

Top-level `gffx.Scene`, `Mesh`, `Material`, `Light` and `Camera` are lazy aliases.
Generic `Mesh.quad`/`Mesh.sphere`, `Material.diffuse`/`Material.emissive`,
`Light.rectangle`, `Camera.look_at` and explicit `Scene.to` reduce user setup.
Material records additionally accept an optional `emission` tensor/source; light
records have `visible=False` by default, while the rectangle factory defaults
to a visible emitter. Existing positional low-level constructors remain valid.
See [simple scenes](SIMPLE_SCENES.md); PNG export is explicit `gffx.io.write_png`,
separate from scene flattening and the image arithmetic namespace.
