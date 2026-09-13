# Synthetic-head image references

`classic-images-v1` connects the registered first-party test head to installed CPU
camera, rasterization, interpolation and texture sampling, followed by independent
scalar lighting. Development reference tooling, not the public material API or a
Cycles comparison. Contract 8.17 in devbrain owns the profile.

## What the fixture checks

- Fixed 128×128 float64 buffers: perspective-correct world positions, smooth
  normals, view directions, corner UVs and vertex colours.
- Independent pinhole projection and pixel round-trip, finite fields, explicit
  coverage, zero background, and repeated identical buffers.
- Constant RGB, vertex RGB, gradient and checker textures. Constant-source
  equivalence and every visible texture sample are independently checked.
- Eight direct-light references: Lambert and classic highlights for all four
  sources. A separate vector equation checks scalar lighting on shared geometry.
- Four exact unit-environment Lambert images, plus independently converged
  hemisphere integrals at eight deterministic head pixels. Not image-wide
  complete-classic or principled quadrature.

Compare linear RGB at `1e-10 + 1e-8*abs(reference)` on the fixed mask. Reports
retain pixel/channel counts, signed mean, MAE, maximum error, MSE and PSNR with
peak 1. Zero error uses null PSNR and an exact-match flag; nonzero error whose
square underflows is rejected. Background mismatch fails. SSIM/FLIP are unavailable.

## Measured local result — 8 September 2026

`build/image-reference-20260908/measured.json`: 12 reference images, 6,536 covered
pixels / 19,608 channels per direct comparison, maximum direct-control error
`1.1102230246251565e-16`. Eight environment controls used 43,008 evaluations total;
maximum error `2.220446049250313e-16`.

The actual supervised worker took 8.553 seconds including imports, sampled peak
573,378,560 bytes, below unchanged 180-second / 1-GiB limits. One setup run, not a
renderer speed, real-time or incremental-memory guarantee.

The immutable sidecar lives under the mapped `reference_corpora` root:

```text
profiles/synthetic_face_m1/classic-images-v1/a6d19536f1708f720e85bf514a6e9bcafa33b50fa1f8a7a3000fe42ce4cfa45e/manifest.json
```

Expected SHA256: `a6d19536f1708f720e85bf514a6e9bcafa33b50fa1f8a7a3000fe42ce4cfa45e`.
It pins the parent scene, actual installed wheel/package hashes, settings, eleven
tool-source snapshots and MIT notice. Two little-endian binary files hold exact
arrays; all payloads total 8,856,474 bytes plus a 24,367-byte manifest. Bounded
loading validates fixed fields/shapes/dtypes, nonoverlapping complete containers,
hashes, finite values and parent provenance. It never executes snapshots or scans
for a newest bundle. The parent remains development-only with both qualification
profiles `not_measured`. Integrity is not mathematical or physical admission.

## Reading the finite demonstration

`build/image-reference-20260908/measured.png` is a visually inspected 16-panel,
1024×1080 contact sheet. Across every row: constant, vertex, gradient, checker.

| Panels | Meaning |
|---|---|
| 1–4 | Unlit albedo, isolating colour sources and the deliberate UV seam |
| 5–8 | Independent directional Lambert reflection |
| 9–12 | Independent directional classic highlights plus diffuse |
| 13–16 | Exact unit-environment Lambert result |

The 128×128 images are enlarged 2× without additional samples. Clipping and sRGB
conversion are display-only; comparisons retain raw float64. No fabricated
candidate or residual panel. This modest stylized head is a diagnostic, not an
anatomical model or a substitute for FLAME/gffit.

Source pages: `shading/image-reference`, `corpora/image-bundles`,
`shading/image-generation`, `shading/public-images`, `shading/image-reference-demo`.
Every test family has a plain-language guide. The rasterization textbook chapter
explains the pipeline and independence boundary. These pages are now deployed to
the private testing dashboard following the separate 8 September deployment request.

## Reproduce without replacing the ordinary environment

The old normal environment lacks camera modules. Use the complete isolated CPU
wheel, not Python source copied beside older native binaries. The approved wheel
at `build/image-reference-20260908/wheels/` has SHA256
`5c8417aae72eb874ac177b843ab30588fa3e4872d10fae90afe3237faab1d899`.
Built offline with PyTorch ON / CUDA OFF, installed only under
`build/image-reference-20260908/installed`.

From `C:/dev/gffx`, temporarily select that directory through `PYTHONPATH` when
using `C:/dev/venvs/gffx-torch210-cu128/Scripts/python.exe`; leave
`GFFX_TEST_SOURCE_TREE` unset and disable `GFFX_DASHBOARD` for offline tests.
Restore prior environment values afterward. The three
`test_shading_image_{reference,bundle,probe}.py` files now contain 67 passing cases,
including the subsequent publisher-provenance regression.
The eight public image cases separately fail at absent `gffx.torch.material`,
before candidate comparison. Do not turn them into skips or private substitutes.

For a fresh approved experiment, `python -m tools.shading_image_probe` takes
`--fixtures ROOT --wheel EXACT_WHEEL --output NEW_REPORT.json`. It checks installed
bytes against the wheel, supervises a bounded worker, saves a create-only sidecar
and sibling PNG, then exits. Existing output names and identical bundle publication
are refused. Read existing evidence with `load_bundle(root, path=EXPLICIT_PATH,
sha256=EXPECTED_HASH)` instead of repeating generation. Optional publication uses
`tools.testing_dashboard.image_reference_demo` with explicit `--root`, `--manifest`,
`--sha256`, `--address` and `--run`; it starts no server. The repaired publisher
was exercised as `image-reference-demo-20260908`; all sixteen images were read
back through verified private HTTPS/WebSocket and matched the saved display bytes.

## Evidence and remaining limits

Retain all XMLs in `build/image-reference-20260908`: initial missing-module red,
old-wheel setup errors, bundle/probe TDD red, and corrected CPU regression.
`final.xml` accidentally selected CUDA camera cases against the CPU-only wheel:
229 passed / six unsupported-provider failures. `final-cpu.xml` corrects the lane:
228 passed / one hardware-dependent skip / six CUDA cases deselected in 36.26 s.
`public-red.xml` retains eight missing-material failures in 3.87 s.

Public shader green tests, principled resource admission, complete-material
environment images, Cycles fidelity, GPU/export qualification and remaining M1
consumer gates are separate work. No new runtime dependency, normal-environment
install or Git publication occurred.

## Authorized live deployment

The tester retains `dist/testing-dashboard-preview-20260905` on port 8791 behind
the existing private route `https://danzielaptop.tail4c8306.ts.net:8444/testing/`.
Metadata was backed up to `build/image-dashboard-deploy-20260908/testing-before.sqlite`;
no history was pruned. The current source server uses the same isolated CPU wheel.

The initial publisher was rejected because its start-event provenance contained
unsupported keys. `publisher-red.xml` reproduces that error before repair: the
start now names `artifact_sha256`, while the detailed producer remains in the
evidence panel. Generation snapshots and the immutable reference bundle are unchanged.

The first live regression then exposed a global JSON-parser monkeypatch in the
oversized-registry test. It disrupted the reporter before teardown; the incomplete
`image-reference-cpu-live-20260908` is explicitly retained as failed, not green.
The test now intercepts only the corpus decoder and checks ordinary JSON still works.

`image-reference-cpu-repaired-20260908`: 229 passed, one hardware skip, six CUDA
cases deselected in 57.88 s; all 230 selected tests completed, no reporter diagnostic.
`image-public-red-live-20260908`: eight genuine missing-material failures in
3.47 s, all completed. XML evidence is in the deployment directory. Coverage was
not remeasured. The demo finished with eighteen artifacts; no continuous producer
was started. Current test explanations and textbook pages are live, not phone-UX
acceptance. The server remains running.
