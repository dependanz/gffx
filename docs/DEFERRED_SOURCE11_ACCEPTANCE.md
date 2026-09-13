# Deferred source11 acceptance sequence

Prepared2026-09-10; **not executed**. Source11 is the next integrated candidate,
not a known wheel identity yet. Root owns building/installing it and assigning
each numerical lease. This is gffx-only readiness evidence, not implementation
or acceptance of gffit/Compositron. Do not run the sequence unattended as a sweep.

## Before the first worker

1. Obtain the verified source11 build/install report and its exact wheel SHA256.
   Use `installed11`/`wheels11` below only if root's report actually names them;
   otherwise substitute the report's exact paths. Never manufacture the SHA.
2. Confirm source11 includes `_deferred.py` SHA256
   `076d473b30dc97712858e2f368746139ae16e8be262bd40f9116b361d0ab53df`,
   the15-case test SHA256
   `85be660979894ec4ba3bed7400884915f68bdec721c5204c7373ad13c69bd785`,
   and `deferred_scale.py` SHA256
   `117bf6fa424a7125cbd3cb6526fd2517b6304878141522559bd8af0625b79683`.
   If integration changes any of them, review/re-pin instead of asserting these
   old hashes still describe the artifact. Preserve source11's immutable inputs.
3. Confirm current test/tool files equal the selected snapshot. Each worker also
   verifies every installed wheel member and records its test/driver identity.
   Set installed-root-first `PYTHONPATH`; disable source-tree override. Never pass
   `--deferred-source` for installed-wheel acceptance.
4. Obtain the sole heavy-worker lease for each stage separately. CPU1GiB host;
   CUDA2GiB host and64MiB allocated GPU; one thread,120s cooperative and180s
   watchdog. A failed/refused/expired stage is preserved and reported before any
   subsequent numerical stage. No cap increase, device fallback or tolerance
   relaxation. Root may separately authorize an independent lane after failure.

Preparation commands (replace the deliberately invalid SHA first):

```powershell
Set-Location C:/dev/gffx
$python='C:/dev/venvs/gffx-torch210-cu128/Scripts/python.exe'
$wheel='C:/dev/gffx/build/m1-team-20260909/wheels11/gffx-0.2.0.dev0-cp310-abi3-win_amd64.whl'
$installed='C:/dev/gffx/build/m1-team-20260909/installed11'
$artifactSha='TBD_FROM_VERIFIED_SOURCE11_BUILD_REPORT'
if ($artifactSha -notmatch '^[a-f0-9]{64}$') { throw 'Verified source11 SHA required' }
if ((Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $artifactSha) { throw 'Wheel identity changed' }
$reports='C:/dev/gffx/build/deferred-20260909'
$env:PYTHONPATH="$installed;C:/dev/gffx"
$env:GFFX_TEST_SOURCE_TREE='0'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTEST_ADDOPTS=''
$env:GFFX_TEST_DASHBOARD='http://127.0.0.1:8791'
function Assert-NewReport([string]$path) {
    if ((Test-Path -LiteralPath $path) -or (Test-Path -LiteralPath ([IO.Path]::ChangeExtension($path,'.xml')))) {
        throw 'Preserve existing report; choose a new reviewed name'
    }
}
```

## Stage1: installed CPU partition controls

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:GFFX_TEST_RUN='deferred-source11-forward-blocks-cpu1'
$report="$reports/source11-forward-blocks-cpu1.json"
Assert-NewReport $report
& $python -m tools.m1_worker_probe --fixtures $reports --output $report --wheel $wheel --sha256 $artifactSha --test tests/python/test_deferred_forward_blocks.py
```

Require15passes, no skipped/error cases, matching wheel/test hashes and a clean
supervisor within CPU limits. These include4/2/1 exact values, runtime call sizes,
inactive work, retained backward partition and matched cotangents. The existing
installed9-plus-source-overlay15green is not a substitute for this stage.

## Stage2: tiny exact CPU/CUDA matrix

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:GFFX_TEST_RUN='deferred-source11-cuda6-1'
$report="$reports/source11-cuda6-1.json"
Assert-NewReport $report
& $python -m tools.m1_worker_probe --fixtures $reports --output $report --wheel $wheel --sha256 $artifactSha --cuda --test tests/python/test_deferred_cuda.py
```

Require6passes and zero skips, CUDA actually available, exact colour/geometric
values and matched VJPs, same mathematical CPU/CUDA budget payload, allocated GPU
peak<=67,108,864bytes and clean2GiB/180s supervision. The six cases are float32/64
hard geometry, float32/64 classic+albedo VJP, float64 soft silhouette+vertex VJP,
and float64 Principled+albedo/roughness/F0 VJPs. A one-ULP mismatch is a failure.
This does not admit any large GPU render. Release the lease after this worker.

## Stage3: matched one-point16-square forward calibration

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:GFFX_TEST_RUN='deferred-source11-onepoint16-forward1'
$report="$reports/source11-onepoint16-forward1.json"
Assert-NewReport $report
& $python -m tools.deferred_scale --fixtures $reports --output $report --wheel $wheel --sha256 $artifactSha --size 16 --workspace-limit-bytes 12582912 --expect-point-block 1
```

Prospective estimate9,911,368bytes for the unchanged2-face,1-light CPUfloat64
fixture. The explicit12MiB allowance selects1point; this is not a raised limit.
Require actual plan selection1, all256pixels covered, finite positive RGB,
forward-only profile, exact wheel/current driver and clean CPU supervision.
This standalone benchmark writes JSON, not pytest events. Its timing is measured
only after the report is complete; old4-point timings cannot calibrate1point.

## Stage4: matched one-point32-square forward calibration

Only after stage3passes and a fresh lease:

```powershell
$env:GFFX_TEST_RUN='deferred-source11-onepoint32-forward1'
$report="$reports/source11-onepoint32-forward1.json"
Assert-NewReport $report
& $python -m tools.deferred_scale --fixtures $reports --output $report --wheel $wheel --sha256 $artifactSha --size 32 --workspace-limit-bytes 12582912 --expect-point-block 1
```

Prospective estimate12,296,008bytes. Require selection1, all1,024pixels covered,
finite positive RGB and the same identity/supervision conditions. No backward
timing is implied. Compute the source-reviewed forecast from these exact reports:
twice the slower seconds-per-pixel rate times6,536covered reference pixels,
plus30seconds overhead. Forecast must fit120seconds; it is not a measurement or
hard upper bound. The actual head worker checks remaining time again.

## Stage5:128-square Principled preflight only

```powershell
$env:GFFX_TEST_RUN='deferred-source11-head128-principled-preflight1'
$report="$reports/source11-head128-principled-preflight1.json"
Assert-NewReport $report
& $python -m tools.deferred_head_probe --fixtures C:/dev/devspaces/gffx/corpora --output $report --wheel $wheel --sha256 $artifactSha --material principled
```

Require verified first-party synthetic corpus revision/hash,1986vertices,
3968triangles, pinned128-square camera and one directional light, and the complete
plan before render allocation. Expected1-point plan64,320,112bytes is prospective
until measured here; it must remain<=67,108,864bytes. Require `mode=preflight`,
selection1, backward false and no image/reference-comparison claim. Preserve the
old source9four-point82,050,928byte refusal. A changed estimate needs review, not
permission to overwrite that failure or widen the limit.

## Stage6: one admitted128-square reference comparison

Only after exact preflight, matching1-point calibration/forecast, root review and
a fresh lease:

```powershell
$env:GFFX_TEST_RUN='deferred-source11-head128-principled-render1'
$report="$reports/source11-head128-principled-render1.json"
Assert-NewReport $report
& $python -m tools.deferred_head_probe --fixtures C:/dev/devspaces/gffx/corpora --output $report --wheel $wheel --sha256 $artifactSha --material principled --render --scale16 "$reports/source11-onepoint16-forward1.json" --scale32 "$reports/source11-onepoint32-forward1.json"
```

The worker independently verifies corpus/independent-scalar/reference identities,
complete preflight, matching wheel+driver+partition calibration, remaining time,
exact coverage equality and declared-model image metrics before saving its two
create-only393,216byte arrays. Require6,536covered pixels, passing image metrics
and clean CPU supervision. No tolerance adjustment for a coverage mismatch.
Shared-native reference rasterization is not an independent raster oracle;
analytic edge controls remain separate. This is one constant-albedo CPU forward
profile, not all DF04materials/sources, full-image backward, Cycles quality,
realtime or a finished wow-factor demonstration. No dashboard publication.

## Prerequisite edits and remaining boundaries

No additional shader source edit is presently needed for this sequence. The
candidate,15controls and calibration guard4green must be included in source11;
only root's verified build/install identity is missing. The generic worker is
root-owned and already records installed member/test identity and resource peaks.
It remains root's responsibility to check exact expected test counts in XML.

Classic128reference work remains a separate required profile. Its prepared
driver currently expects the recorded4-point calibration because its plan keeps
that default; do not reuse1-point timings without an explicit reviewed policy.
An independent fresh4-point16/32calibration or a separately reviewed conservative
classic forecast is needed before that candidate. Per-vertex/textured DF04 and
larger render targets remain separate acceptance work. No consumer repositories,
private model assets, new native raster API or memory-cap changes are involved.
