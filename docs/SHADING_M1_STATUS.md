# Shading: the locally tested M1 slice

Current milestone declaration,15October2026: M1, Group4 and finite Group5 are complete for adopted development readiness. See [exact scope and accepted presentation](M1_DEVELOPMENT_READINESS_CLOSEOUT.md). Historical technical evidence below retains its original artifact/device/workload limits; independent filtered qualification and original corpus profiles remain unpassed/unadmitted. No universal current-wheel, consumer-completion, SIGGRAPH-quality or likeness claim.

The local M1 shading gates are complete as of 2026-09-09. This is an
implementation/evidence map, not a release announcement. Development
work remains on `m1-camera`; hosted qualification and publication are separate.
M1 means readiness to START gffit/Compositron, per D282's 2026-09-09 correction. Earlier real-consumer exit wording is superseded. B1 demonstrates public CPU fitting integration and the scoped M1 milestone is complete; independent profile-wide qualification remains separate. See M1_CLOSEOUT_RECONCILIATION.md.

## What the public interface provides

| Surface | Current implementation | Important boundary |
|---|---|---|
| Lobes | Lambert, Blinn-Phong, GGX distribution, correlated Smith visibility, Schlick Fresnel | Eager tensor compositions; sampling/PDF interfaces and native shading are later work |
| Lighting | Directional, point, real band-2 spherical harmonics; tagged custom packets and checked integration | SH is irradiance, not a general visibility-aware environment renderer; delta weights and solid angles are not interchangeable |
| Materials | Classic and energy-budget-coupled principled; checked custom BRDF callables | Custom materials gain no automatic conservation guarantee; principled is not Blender's complete Principled BSDF |
| Albedo | Constant/broadcast colours, interpolated vertex/corner colours, existing native UV/mip sampler outputs | Callers supply tensor fields; no opaque texture/source engine, implicit loader or new dependency |
| Resources | Exact named CPU and separately named CUDA float64 bundles | No acceptance by cast, payload labels, runtime download or unrequested fallback |
| Differentiation | Public lobe/light/material checks, complete native mip-to-material chains and small per-parameter descent fixtures | Piecewise boundaries retain their declared conventions; small all-input checks are not full-image all-input qualification |

The base `import gffx` remains dependency-free and lazy. Using its PyTorch adapter
requires the separately installed supported PyTorch dependency. There is no new
mandatory dependency from these shading changes.

## Exact evidence, not interchangeable versions

| Evidence | Artifact / result | Scope |
|---|---|---|
| Textured environment forward matrix | CPU artifact `1c97f8df`, progress `1752ce8e` | Twelve complete 128x128 profiles, each with 6,536 covered pixels; declared independent model and shared immutable geometry |
| Direct-image colour gradients | Same CPU artifact, cohort `aa3950e9` | Both materials, complete source-texture gradient, independent finite differences and fresh descent |
| Environment-image colour gradients | Same CPU artifact, contract 8.38 | Four scheduled profiles; see completion state below |
| Named CUDA resource | Artifact `d46d826c`, admission `b3e723b0` | Five physical/chain/repetition stages plus six named-constructor/use checks; Windows RTX 5090 Laptop / PyTorch 2.10.0+cu128, float64 |
| Latest CPU boundary/material regression | Artifact `d46d826c`, report `102b3472` | 368 passed, no skips, all nine optional evidence bindings supplied; not repository-wide coverage |
| Explicit device ownership | Artifact `d46d826c`, three additional constructor tests | Explicit CPU allocation survives an ambient meta-device context; coefficient tensors remain bit-identical |
| Compile/export boundary | Report `5018ddf5` | Six explicit rejection tests; **not** compiled or exported execution support |

Short identities above are navigation labels. Devbrain's `NUMBERS.md` and the
reports contain full SHA256 values, exact paths, stage provenance and limits.
Do not merge different artifact identities into an invented single-wheel run.

Environment-image checkpoint, 2026-09-09: classic/uniform (`b7bf063b`) and
classic/affine (`f6e5d1ab`) pass all 104 workers each, both sampling offsets and
fresh before/after measurements. Principled/uniform (`0c2ab0e4`) also passes
all104 workers and both image measurements. Principled/affine (`68ab3179`) is
complete as well. All four pass strict saved replay: 416 workers, 948,128 material
calls and 970,883,072 sampled directions. Total recorded profile time is
8,255.2956223 seconds; maximum sampled worker memory is 598,310,912 bytes.
Every before/after image, source-texture gradient and offset loss meets the
unchanged independent-reference checks. Both losses decrease in every profile.
Final replay/display/guide tests pass 25 cases, with no skips.

The final wheel fixes two ambient-device allocations found during close-out;
three failing-first tests precede the repair. It preserves coefficient equations
and table bytes. The complete image evidence remains on its original artifact,
not retroactively attributed to this new wheel. Its predecessor `fc6746c5`
qualification and original failed device controls remain retained.

## What remains deliberately outside this slice

Native C11/CUDA shading kernels, native shading `_out`, general HDR decoding,
whole-image CUDA gradients, compiled/exported shading, arbitrary high-resolution
throughput, sustained real-time guarantees and full Cycles fidelity are not
claimed. Finite energy tests do not prove every continuous parameter value or
every GPU. Camera/scene/animation/deferred/image and real-subject gffit K0–K4
acceptance remain separate milestones. Coverage is deferred until after M1.

## Follow the tests and mathematics

The testing dashboard has individual test explanations as well as these feature
pages: `shading/cuda-bundle`, `shading/fitting-closeout`,
`shading/environment-image-adjoints`, `shading/environment-image-evidence`, and
`shading/environment-image-adjoints-demo`. The finite image gallery is display
evidence, never an extra test pass. Unknown pixels in partial previews stay grey.

The `learn/autodiff` article explains why complete radiance precedes the image
loss, why the full colour Jacobian matters, and why tile losses are weighted by
pixel count. The energy-conservation chapter distinguishes the rejected model,
the replacement budget and finite physical qualification.

For usage and exact contracts, see [materials](PRINCIPLED_MATERIAL.md),
[classic shading](CLASSIC_MATERIAL.md), [resource identity](PRINCIPLED_BUDGET_V1.md),
[CUDA admission](PRINCIPLED_CUDA_QUALIFICATION.md), and
[environment adjoints](ENVIRONMENT_IMAGE_LOSS.md).
