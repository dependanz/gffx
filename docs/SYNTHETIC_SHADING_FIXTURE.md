# Synthetic shading fixture and independent reference

These are development tools, not new `gffx` runtime APIs or a FLAME replacement.
The authoritative recipe and reference profiles live in devbrain's
`CORPORA_V0_1.md` and `SHADING_CONTRACT_V0_1.md` section 8.15.

From the repository root:

```python
from tools.synthetic_face_fixture import generate, encode, sha256
from tools.shading_independent_evaluator import classic, direct_buffer, LossCurve

fixture = generate()  # In memory; no files, downloads or asset registration.
assert len(fixture['vertices']) == 1986
assert len(fixture['triangles']) == 3968
content_bytes = encode(fixture)
content_hash = sha256(fixture)

n = [0., 0., 1.]
reference = direct_buffer(
    [n], [n], [True], [1., 1., 2.], [1., 1., 1.],
    model='classic', albedo=[.2, .4, .6], specular=.1, exponent=2.)
```

The fixture includes welded triangles, face-corner UVs, vertex RGB, three small
texture maps, a fixed camera/light and material settings. A null principled
budget ID deliberately means no physical resource has been admitted. Geometry
uses +Y up/+Z front; the light direction is normalized at evaluation. Texture
sampling uses half-texel bilinear repeat with v=0 at the first stored row.

The reference has no dependency on PyTorch, gffx or private candidate functions.
`LossCurve(payload)` copies bounded-cubic endpoint data, evaluates scalar Hermite
curves and computes the weighted mean through separate adaptive segment
integration. The latter uses a total absolute target of 1e-12 per endpoint mean
and at most 18 refinements per segment; exhaustion raises an error. These are
reference-integration controls, not relaxed physical acceptance tolerances.
It is deliberately slow and does not provide autodiff or a rasterizer.
`direct_buffer` accepts shared normal/view/coverage buffers, a single light and
shared material fields, with a 16,384-entry ceiling. It does not create those
buffers or establish independent camera/visibility correctness. Background is
black; an entirely empty covered mask errors.

The in-memory calls above write nothing. A separate development registry can save
the fixture with actual source/artifact hashes and the MIT notice. Same-environment generation is repeatable;
cross-platform transcendental bit identity is not promised. Generator code uses
the repository's MIT license; retain its notice when qualifying redistribution.

## Local development registration

The standard-library tool is repository tooling, not an installed `gffx` API.
From the repository root, choose an explicit corpus directory outside Git and run:

```powershell
python -m tools.corpus_registry C:/dev/devspaces/gffx/corpora --source-root C:/dev/gffx --base-commit <full-checkout-commit>
```

Replace the placeholder with `git rev-parse HEAD`. This command creates a new
entry only; rerunning for an existing ID refuses to overwrite it. The local
2026-09-08 entry already exists, so load it instead:

```python
import json
from tools.corpus_registry import load_entry

entry = load_entry('C:/dev/devspaces/gffx/corpora', 'synthetic_face_m1',
                   allow_development=True)
fixture = json.loads(entry['files']['scene.json'])
```

The loader checks indexed manifest and artifact hashes, bounded sizes, license
notice, source metadata and safe paths. It never executes the copied generator.
The generator is currently uncommitted: `source.commit` is null, `base_commit`
identifies the checkout ancestor, and the copied bytes/hash identify actual source.
Do not claim the ancestor commit contains this generator. Both rendering profiles
remain `not_measured`; `require_profile='gffx_model_m1'` or `'cycles_fidelity'`
therefore raises even with development opt-in. Integrity is not rendering accuracy.

An exclusive writer lock and atomic index replacement prevent partial entries
being advertised. A crash may leave a stale lock or unregistered revision;
inspect it manually, never delete it or retry automatically. This is a small,
development-only registry, not a generic third-party asset admission system.
Its hashes do not protect against an owner maliciously rewriting the whole root.

## Evidence and limits

The initial baseline in `build/fixture-evaluator-20260908/red.xml` contains
24 missing-module failures: 13 public edge cases plus five generator and six
reference tests. After implementation the first 11 utility tests pass; expanded
tests add nonconstant roughness, malformed input, copied payloads, convergence
failure and agreement with the installed CPU texture sampler. A separate failing
scene-setting regression precedes adding camera/light/material metadata.
The public edge cases still require the absent public shading APIs; utility
success is not evidence those assertions passed.

Final `verified.xml`: 24 passed (20 utility cases and four teaching/catalog checks),
4.48 seconds pytest / 4.469 seconds JUnit. Preserve `final.xml` with 23 passes and
one failure: the new link check initially omitted the valid textbook index and
was corrected to include it. Fifteen textbook pages regenerate reproducibly.

The dashboard source has explained pages `corpora/synthetic-face`,
`shading/independent-evaluator` and `shading/public-edges`. The topology and energy
chapters explain their scope. No live service or rendered demo is started.
Future finite demonstrations should show the mesh, UV seam and diagnostic maps,
then candidate/reference/residual panels once actual public shading exists.
The independent single-point environment quadrature now has known-answer and
convergence/failure tests; see [ENVIRONMENT_REFERENCE.md](ENVIRONMENT_REFERENCE.md).
Image-wide qualification, physically admitted budgets, full renderer images,
Cycles comparison and real-model fitting remain separate work.

Registration evidence: `build/corpus-registration-20260908/green.xml` has 40
passes (30 registry cases, six generator cases, four teaching/catalog checks),
JUnit 6.945 seconds. Preserve the initial 28 missing-module failures, the first
27-pass/one-test-helper-failure run, the incomplete reparse mock failure and the
JSON overflow regression (`1e999` previously decoded as infinity). An authored
source-only `corpora/registration` page explains all 13 registry test families.
No server or continuous producer was started.
