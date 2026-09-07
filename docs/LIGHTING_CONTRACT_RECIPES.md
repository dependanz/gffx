# Lighting contract recipes and finite illustrations

This is **Phase-0 mathematical development**, not an installed public lighting API.
`tools/light_contract_reference.py` expresses the proposed point, directional and
spherical-harmonic calculations in the already-required development PyTorch.
There is no new base-package or dashboard dependency. The authoritative contract
is the registered devbrain's `SHADING_CONTRACT_V0_1.md`, section 8.5.

## What the recipes specify

- Explicit point, light and batch axes; float32/float64, same device and dtype.
- Nonzero normal/direction normalization as part of the new M1 recipes. The older
  renderer's separate arithmetic is unchanged.
- Point light weight = RGB radiant intensity / squared distance, before surface
  cosine. Coincident pairs return a false mask and zero payload/gradients, not an
  invented bright value. Unrepresentable arithmetic raises an error.
- Distant light weights do not decay with distance. Delta lights have no ordinary
  solid-angle density; these mappings do not invent a `pdf`.
- Nine incident-radiance SH coefficients per RGB channel, not preconvolved values.
  The result is Lambert irradiance, not complete principled-material shading.
  Signed coefficients/results are retained, including truncation ringing.

Run the independent fixtures using the configured installed-wheel environment:

```powershell
python -m pytest tests/python/test_light_contract.py -q
```

Finite-difference comparisons cover interior derivatives. A passing recipe is not
runtime API, native parity, CUDA, compilation/export, corpus or performance acceptance.
The material-budget evaluator and full missing-public-API baseline remain separate work.

## View or reproduce the demonstration

The tester page is `/testing/?page=shading/lights-demo`; its tests are at
`/testing/?page=shading/light-contract`. The existing energy textbook chapter now
contains point/distant lights, SH, worked examples and an exercise:
`/testing/?page=learn/shading/energy-conservation`.

Against an already-authorized local tester, use a fresh execution ID:

```powershell
python -m tools.testing_dashboard.lighting_demo --address http://127.0.0.1:8791 --run lighting-example-UNIQUE
```

The producer computes 16 distance samples, 181 angular samples, one 256x128 RGB
image and a small record, sends five artifact references, then exits. It does not
start a continuous producer, live animation or viewport renderer. The image uses
analytic sphere normals: exact directional cosine on the left, band-2 SH on the
right. Both use the same display scale. Magenta explicitly identifies negative SH;
the graph preserves its signed numerical value. These are mathematical illustrations,
not mesh-rasterizer or gffit-scene acceptance.

The temporary-server publication fixture checks integration without reusing live
test counts or treating a demonstration as a passing test. Every test definition
has an authored purpose, method and limitation in the tester's explanation catalog.

## Learning content

Build the original MathML lesson without KaTeX, MathJax or network assets:

```powershell
python -m tools.testing_dashboard.build_lessons
python -m tools.testing_dashboard.build_lessons --check
```

The SH addition-theorem polynomial and inverse-square numeric examples are
independently checked in the recipe tests. Point-source units follow
[PBRT's point-light treatment](https://pbr-book.org/4ed/Light_Sources/Point_Lights);
the SH basis/convolution follows
[Ramamoorthi and Hanrahan](https://graphics.stanford.edu/papers/envmap/envmap.pdf).
