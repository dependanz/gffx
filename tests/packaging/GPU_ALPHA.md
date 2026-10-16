# GPU alpha fixture methods and limits

`test_gpu_alpha.py` checks the installed public operation route. Tiny smooth
triangles, points, matrices and textures use deterministic matched input bytes.
For 12 primitives it compares CPU/GPU forward values and matched first VJPs
byte-for-byte in float32/float64. Float64 gradcheck independently perturbs inputs
on GPU using PyTorch's standard fast directional method/tolerances. Five
animation operations receive additional independent gradchecks, alongside the
existing animation parity, boundary, storage and caller-stream controls.

Primitives: face geometry, vertex normals, face gather, point transforms,
perspective divide, KNN distance, closest-mesh distance, surface sampling,
raster barycentrics/depth, attribute interpolation, texture pyramid and sampling.
Surface sampling uses one nondegenerate face so its selected-face branch stays
fixed under finite perturbations. Its key/counter are explicit uint32 tensors
on the geometry's device; public/raw-dispatch mismatch tests ensure host RNG
operands are refused before driver copies. No valid sampling arithmetic changed.
Raster finite differences cover stable selected fragments away from branch
boundaries; hard visibility discontinuities are not claimed differentiable.

The selected release cohort also exercises existing small scene hard/soft,
classic/Principled material, SH-light, image warp/blur/compositing, stream and
repeated-backward controls. Larger physical-resource, performance and framework
matrices remain separate. Historical PyTorch arithmetic attribution tests are
not reclassified as passing native tests. The filtered opaque/reference campaign
and photograph fitting are not run.

Install the exact audited wheel into an isolated selected-framework environment
before running tests. `tests/conftest.py` reports the imported package path;
verify version and installed native-file identities against the wheel. Test
helper directories are fixtures, never a source-package fallback. The release
runner rejects skipped GPU cases and records its exact selection, paths,
framework/device provenance, JUnit and resource peaks outside source Git.

For an initial direct check from the repository with the wheel installed:

```powershell
python -m pytest --import-mode=prepend tests/packaging/test_gpu_alpha.py tests/pytorch/test_animation_cuda.py tests/pytorch/test_animation_completion_cuda.py -q -o junit_family=xunit1
```

The full frozen release selection and exact outcomes are in
[the a2 profile](../../docs/ALPHA_0_2_0A2.md). These are finite artifact/host tests,
not all-input proofs, GPU fleet qualification or sustained timing guarantees.
