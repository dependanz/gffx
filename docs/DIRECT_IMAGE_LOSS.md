# Direct-light image-loss gradients

This development fixture checks whether image error reaches a source colour
texture through the installed gffx sampler and complete material. It is not a
new public fitting API, a converged reconstruction, or M1 completion.

## Fixed experiment

- Shared immutable 128x128 normal/view/UV/coverage buffers; 6,536 covered pixels.
- A 32x32 RGB gradient texture, native one-level pyramid, bilinear repeat sampling.
- Target texture is 0.85 times the initial texture; the independent target image
  uses the declared classic or principled material and white direct light (1,1,2).
- Classic specular 0.1/exponent 2; principled roughness 1/F0 0.04 with the exact
  accepted `ggx-smith-loss-cpu-f64-v1` resource.
- Mean squared linear RGB error divides by **3 times all covered pixels**.
  Background never enters the denominator.

`tools/image_loss_reference.py` uses ordinary Python arithmetic, not torch or
automatic differentiation. Independently evaluated intercepts/slopes describe
the colour-affine material response. A separate four-texel stencil and its
transpose provide the source-texture derivative. Three directional central
differences at two scales cross-check this reference. Public RGB, loss and every
gradient entry must pass the prospectively fixed tolerances.

The public path takes derivatives in four-pixel chunks, releases each graph,
and accumulates detached source gradients. It then updates the texture by
minus five times the gradient, without clamping, and builds fresh graphs for
a second measurement. Both loss reduction and independent gradient agreement
are required. Only source colour varies: geometry, UV, visibility, lighting,
roughness and specular remain fixed. Environment-image and other parameter
derivatives are separate obligations.

## Evidence, 2026-09-09

Two 64-pixel preflights and two complete-image workers passed in 55.7254 seconds
total, with 6,600 public backward calls and 26,400 scalar coefficient evaluations.
The largest sampled whole-worker working set was 568,446,976 bytes. Each worker
retained the 1 GiB host cap, 120-second cooperative deadline and 180-second
watchdog; the four-worker aggregate cap was 900 seconds. Sampled working set is
not a proof of allocator peak or sustained-workload behaviour.

The complete-image losses changed as follows after one step:

| Material | Initial loss | Second measurement |
|---|---:|---:|
| Classic | 0.00015462617249806894 | 0.00015429106511035892 |
| Principled | 0.00015048778135993144 | 0.00015016653892317087 |

The local cohort is `build/image-loss-20260909/qualification/cohort.json`, SHA-256
`aa3950e93addb1c313ae9e482e48867348c52f978294611f25ccdeb51cd214dc`.
It binds source hashes, four reports, exact installed wheel, resource, corpus,
pixel ownership, measured arrays, actual worker PIDs and supervision results.
These local build outputs are not repository assets. Reports are create-only;
an existing directory or failed worker is not an automatic retry request.

Saved replay checks hashes, recomputes scalar coefficients/adjoints, verifies
the fitting update and rejects forged numeric/identity/resource evidence. It
does not rerun public shading. Against the exact isolated installed CPU wheel:

```powershell
$env:PYTHONPATH = 'C:/dev/gffx/build/light-batches-20260908/final-installed'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:GFFX_IMAGE_LOSS_ROOT = 'C:/dev/gffx/build/image-loss-20260909/qualification'
$env:GFFX_IMAGE_LOSS_SHA256 = 'aa3950e93addb1c313ae9e482e48867348c52f978294611f25ccdeb51cd214dc'
& C:/dev/venvs/gffx-torch210-cu128/Scripts/python.exe -m pytest -q tests/python/test_image_loss_gradients.py
```

The testing dashboard pages are `shading/image-loss-gradients` and
`shading/image-loss-gradients-demo`. The latter shows target / before / after,
plus absolute source-gradient maps normalized **only for display**. Tiny visual
change is intentional; this is one correctness step, not a finished fit.
`learn/autodiff` explains the objective, sampler transpose and graph streaming.
No runtime dependency was added; torch remains the explicitly selected adapter.
