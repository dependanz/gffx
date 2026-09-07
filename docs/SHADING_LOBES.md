# Elementary shading: Phase-0 recipes

These are executable contract recipes in `tools/shade_contract_reference.py`, not
exported `gffx.shade` or `gffx.torch.shade` APIs. They prepare the mathematical and
tensor contract before public material implementation. They require only the
already optional PyTorch dependency. No wheel rebuild is needed for this tooling.

The five functions evaluate Lambert, an approximate classic Blinn-Phong highlight,
GGX distribution, height-correlated Smith visibility and Schlick Fresnel. They do
not render geometry, perform shadow tests or assemble the `principled` material.
That material's approved energy-budget coupling is separate and remains required.

## Exact distinctions

- Lambert returns `albedo/pi`, not lit RGB.
- Blinn uses the normal-half-vector cosine with `(n+8)/(8*pi)` scaling. This is
  distinct from reflected-vector Phong and from the normalized Blinn NDF.
  Its normal-view, exponent-two energy integral is **25/24**: a negative control,
  not a passing conservation result. No principled energy threshold is changed.
- GGX D uses perceptual roughness to the fourth power. Normalizing its projected
  microfacet area does not establish conservation of an assembled material.
- Smith visibility already includes division by `4*nV*nL`. Multiplying by another
  such denominator would be a bug. At either horizon the explicit extension is
  zero with zero branch derivatives; interior derivatives are tested separately.
- Schlick is an RGB polynomial with unclamped endpoint derivatives.

Cosine queries are `[P]` or `[B,P]`, with same-device float32/64 inputs. The first
input owns the batch. Scalar parameters may be scalar tensors, `[P]`, `[1,P]` or
`[B,P]`; RGB Fresnel parameters may be `[3]`, `[P,3]`, `[1,P,3]` or `[B,P,3]`.
Lambert albedo is `[P,3]` or `[B,P,3]`. There is no implicit point-axis or additional
scene broadcasting. Cosines must be in [0,1], albedo/F0 in [0,1], exponent >=0 and
roughness in (0,1]. Invalid inputs fail rather than being silently clipped.
Empty/strided tensors are accepted. Roughness zero is not repaired with epsilon.

Eager checks may synchronize a GPU. These CPU recipe fixtures do not establish
CUDA, compilation/export, native ABI or edge performance support.

## Reproduce

From the repository, using the configured environment and installed gffx wheel:

```powershell
python -m pytest -q tests/python/test_shade_contract.py
python -m tools.testing_dashboard.build_lessons --check
python -m tools.testing_dashboard.shade_demo --address http://127.0.0.1:8791 --run shade-demo-UNIQUE
```

The finite producer exits after publishing one 1280x360 four-sphere image, an
explanation, five angle curves and a record of closed-form examples. It adds no
continuous mesh logging. `shading/lobes-demo` is an illustration; acceptance tests
live on `shading/lobe-contract`. Each test has its own explanation and evidence.

Left to right: teal Lambert, matte plus classic highlight, gold-colored
single-scattering GGX at roughness 0.25, then at 0.65. The analytic sphere normals
are not a mesh renderer or a metal workflow. The common fill and display transform
are explicitly illustrative. Numeric plot curves are not brightness-normalized.

The energy textbook chapter explains the formulas, half-vector, normalizations,
singularities, gradients and worked exercises using the in-tree MathML renderer.
Formula references: [Filament's material equations](https://google.github.io/filament/main/filament.html)
and [Lagarde's normalization discussion](https://seblagarde.wordpress.com/2011/08/17/hello-world/).
