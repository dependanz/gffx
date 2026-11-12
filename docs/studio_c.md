# Studio C: skin, wax, strands and fine surface detail

`examples/studio_c.py` loads an explicitly provided portable scene and renders it
through gffx. The source Einar asset is separately attributed CC-BY4.0 Blender
Studio work by Simon Thommes, Andy Goralczyk and Julien Kaspar; it is not bundled
with the library. Blender is an authoring/reference tool, never gffx's renderer.

`examples/skin_wax.py` is a self-contained alternative without licensed assets.
Its rectangle lights are one-sided: their emitting direction is `cross(u, v)`.
The span winding points each light toward the objects.

```python
skin = g.Material.pbr('skin', (0.55, 0.27, 0.18), roughness=0.4,
                      subsurface=1.0,
                      subsurface_radius=(0.006, 0.0012, 0.0006))
hair = g.Mesh.strands('hair', points, radius=0.0002,
                      tip_radius=0.00001, material='hair-material')
```

Subsurface is an optional dielectric raster approximation. RGB radius is a
world-space mean free path in the scene's units; Studio C uses metres.
The diffuse-surface fit converts it to profile length `d = radius / s(A)`,
where `s(A) = 1.9 - A + 3.5*(A - 0.8)**2` and A is relative RGB albedo.
Blender's legacy artist radius has a different scale: the paired exporter maps
it to mean free path by dividing by `4*pi`. Equal raw numeric values would
therefore describe different materials. This corrects the unpublished initial
Studio C candidate's excessive diffusion, without tuning light power or colour.
`subsurface=0` retains the prior PBR path exactly. Diffuse illumination is
separated from specular, normalized spatially across visible samples belonging
to the same object, then remultiplied by sharp surface colour. Depth rejection
reduces leakage between separated surfaces. The radial sampling distribution is
the normalized two-exponential diffusion approximation described by
[Christensen and Burley](https://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf),
independently implemented here; it is not copied from Blender.

`subsurface_transmission=1` enables a separate thin-backlighting approximation
when shadows are enabled. Existing shadow maps estimate distance from the first
light-visible surface to the receiver and apply exponential RGB attenuation.
Set it to zero to isolate front diffusion. Offscreen entry surfaces can enter
this term, but screen-space diffusion cannot recover hidden lighting/geometry.
This Beer attenuation term differs from EEVEE's transmission lookup profile.
Multiple blockers, finite map precision, discrete diffusion sampling and missing
multiple-scattering paths remain approximations. This is not random walk, full
BSSRDF transport, or an energy certificate. Integrated scattering currently
requires the packaged native float32 CPU/CUDA provider. The explicit raster
integrator remains forward-only; differentiable deferred semantics are unchanged.

`Mesh.strands` accepts `[strand_count, sample_count, 3]` centreline points, root
and tip radius, and 3–32 sides. Parallel-transport frames reduce twisting;
adjacent coincident samples and reversing tangents are rejected. The result is
an ordinary mesh with smooth normals/UVs, preserving tensor links through mesh
construction. It provides explicit silhouette geometry, not a dedicated hair
scattering model. Einar's source groom is converted once to shared explicit
tubes for the paired fixture. Original hair shader/eye transmission behaviour
is not implied by that conversion; glass/wet interfaces remain Studio D work.

The independent controls cover constant diffuse response, mask/depth boundaries,
RGB spreading, C11/CUDA versus float64 tensor diffusion, analytic slab
attenuation, tube winding/topology/derivatives, zero-weight preservation and
actual integrated images. A/B regressions track existing metal/coat,
reflection-proxy/recursion, diffuse-GI and efficiency differences separately.

`normal_space='world'` explicitly interprets a normal texture as world-vector
data; tangent space remains the default. World maps are appropriate for a
frozen evaluated mesh and must be regenerated or transformed for deformation.
Opaque strand tubes retain their geometry in all comparisons; higher raster
supersampling improves subpixel coverage rather than removing fine strands.
The portrait study uses 64 shaded subpixels and the isolated strand study 256;
EEVEE uses its own 256-sample temporal accumulation and reconstruction filter.
Those are different sampling/filtering schemes, not a claim of equal work.

The dense-groom CUDA path uses conservative coarse bins for large triangles
and exact fine-pixel bins for microtriangles, preserving ascending face order
and depth ties. No geometry decimation or lower-resolution shadow substitute
is used. Coalesced face attributes reduce transient geometry memory.

Paired renderer image calls include scene/shadow/scattering updates. Source
conversion and texture baking are preparation costs; file transfer/EXR/PNG/raw
linear output and video encoding are measured separately. EEVEE is visually
audited as an approximate reference and is not assumed to be ground truth.

The paired exporter preserves every polygon material index while replacing slots.
Clearing slots resets indices in Blender: the real four-object control restores
pupil/iris/sclera and gums/teeth assignments, checks the original evaluated
topology and preserves all original triangles and corner attributes. This is
a source conversion correction, not a material-lighting workaround.

EEVEE animation references use Blender’s native animation operation. Three
independent frames match the per-frame operator PNG/linear output exactly;
per-frame image/update and output boundaries are explicit in each result.


### Geometric receiver uncertainty and shadow sampling

The raster integrator uses deterministic multi-jittered rectangle strata: each
sample occupies a different two-dimensional cell and a different projected
sub-stratum. Sampling positions remain fixed across frames. Shadow maps and
native light integration use the same packed positions. The deferred integrator
keeps its existing sampling and differentiability behavior.

Bilinear comparison filtering reconstructs each map's visibility. The area
light's extent produces soft shadows through source integration rather than a
wide spatial filter. Receiver-plane correction handles planar slopes. A Frobenius bound of affine geometric
vertex-normal variation per world unit estimates additional quadratic depth
uncertainty on curved receivers, before normal textures are applied. A cap based
on a fraction of a texel bounds the correction. Flat surfaces keep their existing
precision-only bias, preserving close planar contacts.

This is finite-resolution shadow mapping. Curvature estimates and the cap can
still miss small occluders or retain false self-shadowing, especially at grazing
angles and on geometry smaller than a shadow texel. It is not exact ray visibility.
The correction does not change geometry, lights, material values, exposure,
supersampling, map count or map resolution. Authoring stays the same:
`g.render(scene, camera, integrator='raster', shadows=True, ...)`.
