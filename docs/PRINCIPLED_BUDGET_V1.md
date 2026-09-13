# Bundled GGX/Smith loss table v1

`ggx-smith-loss-cpu-f64-v1` is the exact self-generated table qualified by the
finite `public-budget-v0.1` profile. It is package data, not a download, a new
runtime dependency, a lighting environment image, or a table generated on startup.
Load it once during setup:

```python
import torch
from gffx.torch.material import PrincipledBudget, principled

budget = PrincipledBudget.bundled(
    "ggx-smith-loss-cpu-f64-v1", device="cpu", dtype=torch.float64,
)
n = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
brdf = principled(
    n, n, n,
    albedo=torch.tensor([.2, .4, .6], dtype=torch.float64),
    roughness=torch.tensor(.4, dtype=torch.float64),
    f0=torch.tensor(.04, dtype=torch.float64), budget=budget,
)
```

This returns a linear RGB BRDF, not an illuminated image. The accepted bundle
needs neither experimental flag. Other valid tables still require explicit
`allow_unverified=True` at loading and `allow_experimental_budget=True` at use.

CUDA float64 has a separately named finite-profile bundle:
`ggx-smith-loss-cuda-f64-v1`. It reuses these exact table bytes but has its own
acceptance receipt and explicit named-construction route. The CPU ID and generic
experimental transfers do not acquire CUDA acceptance automatically. See
[CUDA qualification and usage](PRINCIPLED_CUDA_QUALIFICATION.md) for measured
hardware, supported construction, evidence and unqualified modes.

## Identity and reproducibility

- Model: `ggx-correlated-smith-schlick-reciprocal-matte-v1`.
- Representation: `bounded-cubic-loss-v1`; envelope format 1.
- Generator policy: `conditional-allocation-v1`.
- Generator source SHA256: `62df9a500bca26e9eeb79fef05275e42567de45eef301a369ba7d6fa838839fd`.
- Original fine-table file SHA256: `c85381a9e78538d2367a13134c895dbc0c7c4180516e504ae8a5b0b4cfce9046`.
- Canonical numerical payload SHA256: `eb9393c7afd6e1a4964019643ee827dff72696896bd05786e8a2ec908ca538e5`.
- Packaged envelope SHA256: `5e095d1aa417dcba460f7cb2a5f72b1cf8c857d793cba1da1b6e77b9fc769cb7`.
- File: `gffx/data/principled/ggx-smith-loss-cpu-f64-v1.json`, 2,877,218 bytes.
- Shipped acceptance record: `gffx/data/principled/public-budget-v0.1.json`, SHA256
  `df424fcb92630ccd2bc70f8cf26997bcc3b0929ad1d56d3ca22e6ad1f4b98131`.

The 129-by-513 grid contains two endpoint losses per knot. Packaging wraps the
saved numbers without changing or recomputing them. The create-only development
tool `tools/package_principled_budget.py` verifies the frozen reports, their
predecessor chain, generator/reference sources and table before writing an envelope.
It is not imported by the installed runtime. Data is self-generated under this
repository's license; no third-party mesh, texture or numerical library is bundled.
Git attributes preserve both JSON files byte-for-byte across newline conventions.

Admission matches the full validated numerical identity, generator, domain,
model, representation and execution lane. A changed `evidence_id` cannot grant
acceptance to other data or revoke acceptance from otherwise identical data.
Unknown IDs—including `latest`—have no fallback. Bundled file bytes must match
the pinned file digest before parsing. Ordinary file/envelope loading accepts
semantically identical validated data regardless of JSON whitespace/key order.
Missing files retain filesystem errors; corrupt or unsupported data raises
`ValueError`. Hashes establish identity, not a cryptographic trust boundary against
someone who can replace the installed Python code itself.

## Finite physical acceptance profile

The following cohort was fixed before observing its results on 2026-09-08:
roughness .00179/.0193/.213/.827; ordinary cosines .019/.463/.947/.9997;
additional directional cosine .0027*r*r; F0 .07/.47/.93/1.

1. 80 generation endpoint comparisons: 280/32 versus 512/64 conditional
   quadrature, including horizon/normal and both sides of the allocation boundary.
   Relative tolerance .0001; exact-zero absolute tolerance 1e-12.
2. 80 public directional triples and all three first partials: independent
   512-order integration. Absolute tolerance 5e-4, relative/refinement .001,
   derivative tolerance .01+.02*absolute(reference partial).
3. 16 public average losses and both partials: streaming 128-light/64-view
   integration with relative .001 and the same derivative tolerance. Separate
   scalar-curve quadrature agrees with the public analytic mean and derivatives
   within 1e-11 absolute.
4. 192 furnace cases with albedo 0/.4/1: independent 512-order specular integration
   plus the public residual budget ratio and scalar-curve integration. Energy
   range [0,1.001], white and refinement error below .001.

All pass. Each stage used one CPU thread in a separate supervised worker, with
120-second cooperative/180-second external limits and 1 GiB sampled whole-worker
memory cap. The public numerical implementation was measured in wheel
`e22255c1582fd961cf9674c30b372b984d3ed0e679350bdfcb95d743d5727927`;
packaging does not change its curve/query mathematics.

Frozen report SHA256 values:

| Stage | SHA256 |
|---|---|
| Guard | `76f78076db5024c7aee3e41215876a4b7700d3d745114b0f727f90c8dcad1714` |
| Directional | `178c501f2e9053215acd35a33cdd069e8d8739a13dd321668da0f2e3a8c0f957` |
| Mean | `8621ce530af1e067156e56020ffeb16ee67454be7145060cc225ab566e0cb6b4` |
| Furnace | `2310c62ac3dc89dfe3aacf509125c6eed56637feeb587781aa36d0e277221b62` |

## Limits that remain

This CPU-named profile is CPU float64 only. Roughness [.001,1], cosine/F0 [0,1], maximum 32 scalar/channel
entries per query; the material's expanded work also has a 64 MiB planning cap.
This is finite empirical qualification, not proof of worst-case accuracy at every
point in the continuous domain. The observed cohort is now regression data, not
a fresh holdout for future tuning. The separately named CUDA float64 profile has
its own [finite qualification](PRINCIPLED_CUDA_QUALIFICATION.md); it is not inherited
from this CPU ID. Float32, compilation/export, broader renderer-sized workloads
and sustained performance remain separate gates. Neither this bundle nor a demonstration closes full M1 or proves Cycles
fidelity. See [the material API](PRINCIPLED_MATERIAL.md).
