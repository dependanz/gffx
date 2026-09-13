# Deferred workload planning checkpoint

Status: built-in planner written after two refusal reds, 2026-09-09.
`resource-flatten-red1.json` and `resource-packet-red1.json` under
`build/deferred-20260909` preserve the installed4 failures. Four planner controls
passed on installed5, followed by separately supervised 8x8 forward and backward
profiles (`resources-green1`, `profile8-forward1`, `profile8-backward1` JSON/XML).
Forward and backward profile test times were 0.79 and 1.54 seconds; their sampled
whole-worker peaks were 572,551,168 and 610,177,024 bytes respectively. These
include imports and test overhead, not isolated shader timing. Do not infer
full-image admission from the tiny recomputation controls. Red tests use allocation
spies so an intentionally refused workload is never actually allocated.

The planner inspects shapes and scalar configuration before `Scene.flatten`.
It must count borrowed input/resource storage, prospective packed geometry and
attributes, high-resolution and centre-sampled buffers, output storage, light
sample packets and retained source/lighting graphs. A conservative estimate is
allowed to refuse a workload that might happen to fit; it may not silently omit
light multiplicity because the image itself is small.

For each built-in Principled block, retain the admitted material's own scratch
formula: three resource workspace estimates for three colour entries per point,
plus eight bytes times point/light pairs times 192 for backward or 64 for forward.
The block has at most four points and at most 1,024 point/light pairs. Add this
transient to the renderer's live storage, rather than pretending that separate
64 MiB allowances imply a 64 MiB total. Blocks execute sequentially; retained
original inputs and integration graphs still span the image and must be counted.

Built-in directional and point light sample counts are statically visible in
their binding tensors. SH diffuse lighting has no sampled-light packet. A trusted
custom material or environment callback can allocate an unknown amount of memory
before returning. Checking its returned shape does not sandbox that callback or
prove its transient-memory bound. Its admission contract is an open review item;
this checkpoint must not silently narrow the adopted callback API, and numerical
results from built-in workloads must not be reported as universal callback-memory
qualification.

The internal estimate reports separate components for input storage,
packed geometry, buffers, sources, lighting and sequential material scratch.
Storage aliases count once; gradient-bearing inputs receive an additional
conservative derivative-storage allowance. Installed7-qualified refinements remove
two proven overcounts: an aliased centre buffer is not a second allocation, and
one retained fragment selects only one material per pixel. Per-frame/material
boolean selection masks remain separately charged; source interpolation retains
the repeated-frame allowance. The two expected reds are preserved in
`resource-reuse-red1` and `resource-material-red1`. All seven controls in
`resources-green2.json` passed on wheel
`f8d1c5f19f8576c3c2fdf4e42e45844f016d9160598c8e647428ad388d56e849`,
including both refinements and light-only fitted-parameter accounting. The
sampled whole-worker peak was 574,427,136 bytes; pytest took 1.34 seconds.
`builtins_only=False` identifies the still-open callback envelope;
it is not an assertion that the estimate controls arbitrary trusted callback code.

The 8 by 8 built-in Principled forward/backward checkpoint is green. The next
approved ladder is 16 by 16, then 32 by 32, with separate source-bound reports
from `tools.deferred_scale`, isolated forward/backward timings and the prospective
estimate components. No automatic sweep or 64+ execution is allowed by that
driver. Preserve the 1 GiB CPU whole-process ceiling, 120-second cooperative
deadline and 180-second watchdog. Larger resolutions need a separately reviewed
estimate and measurement; the six-pixel gradient pass is not their acceptance.

## 16-square continuation and immutable-table accounting

On installed8 (`b9b318ac8c95e6eb55ff4a3715bd3f7d5aebc4e6e999ced1cdd0babe92309613`),
the16-square forward profile passed with256 covered pixels, a27,642,184-byte
estimate,0.318533700003-second isolated forward and557,723,648-byte sampled worker
peak (`scale16-forward1.json`). The separate backward request stopped before
scene allocation:67,930,712 estimated bytes exceeded67,108,864. Its559,112,192-byte
worker peak describes a safe refusal, not a rendered backward or timing result.

The refused estimate tripled3,201,048 bytes of immutable budget tables whenever
roughness needed a gradient. Table adjoints do not exist under the public resource
contract; query and backward intermediates already have the full separately
charged material scratch. A two-point saved-storage metadata audit preserved table
pointers/versions and absent table gradients:302 save events,20,456 borrowed table
bytes and452,362 nonresource unique saved bytes against a28,369,920-byte scratch
allowance. It retained no extra tensor handles. This is not a total transient
allocation measurement. See `storage-audit1.json` and its XML property.

The candidate correction counts budget-only non-gradient table storage once,
retaining3x for every ordinary binding and any alias exposed through one. Material
scratch is unchanged. `resource-constant-red1` preserves the accounting failure;
`resource-constant-overlay1` passed8 controls as an explicitly hashed source-overlay
diagnostic. Installed9 wheel
`55a59de4ac3f7066cf899eba7630a071424d2b09b7eb6fbe10db6bde77c58904`
then passed all9 resource controls, including the ordinary-alias regression
(`resources-green3`:2.64s,576,241,664-byte sampled whole-worker peak).

The repaired16/32 ladder subsequently passed as four separate supervised workers:

| Report under `build/deferred-20260909` | Planned bytes | Forward seconds | Backward seconds | Whole-worker peak bytes |
| --- | ---: | ---: | ---: | ---: |
| scale16-forward2.json | 27,642,184 | 0.3952908 | Not requested | 558,604,288 |
| scale16-backward2.json | 61,528,616 | 0.4201190 | 1.4296900 | 597,372,928 |
| scale32-forward1.json | 30,026,824 | 1.6833120 | Not requested | 557,936,640 |
| scale32-backward1.json | 66,274,088 | 1.7164163 | 3.9802589 | 599,556,096 |

These are fixed two-triangle, one-light, float64 CPU profiles; backward fits shared
roughness only. They are not high-resolution, CUDA, arbitrary-material or sustained
frame-rate qualification. Full scalar-domain/gradient controls remain separate.
The original16-square refusal remains preserved. No cap was increased.

## Studio diagnostic and fixed-head preflights

The original258-face studio geometry passed its dedicated topology/normal check.
`studio32-render1.json` then rendered three32-square views in1.6442196 seconds,
with a31,938,864-byte estimate per view and559,271,936-byte sampled process peak.
Six cached images include the three colour views and normal/depth/coverage
diagnostics. Visual review classified these as useful small diagnostics, not a
finished wow-factor showcase. They have not been published as one.

On the same installed9 artifact, `head128-classic-preflight1.json` admitted the
fixed first-party head at55,208,792 planned bytes. No image was rendered or compared.
`head128-principled-preflight1.json` refused82,050,928 planned bytes before scene
flattening or rendering. Its564,453,376-byte process peak is a preflight/refusal
measurement only. The existing16/32 forward measurements yield a51.48852975-second
conservative candidate forecast, not a measured head-render time.

The minimal proposed next Principled change is a smaller block for genuinely
forward-only workloads when four-point scratch does not fit, preserving fixed
four-point backward partitioning. One-point scratch would yield64,320,112 planned
bytes for this particular head. This is a proposal, not implemented or admitted
by the current renderer; it requires explicit chosen-block metadata, exact
one/two/four-point forward controls and runtime call-size checks. It does not by
itself admit the two-light128-square studio or larger full-frame targets.

## Paused checkpoint after M1 scope reconciliation

M1 means gffx readiness to start gffit/Compositron, not implementation or real-model
acceptance of those consumer projects. This lane has used only gffx source and
the mapped first-party synthetic corpus. No consumer repository or private asset
was accessed. No Git publication or dashboard showcase publication was performed.

The latest renderer qualification artifact remains installed9 with SHA256
`55a59de4ac3f7066cf899eba7630a071424d2b09b7eb6fbe10db6bde77c58904`.
All reports cited above are preserved under `build/deferred-20260909`. Standalone
scale reports are benchmarks, not pytest events. The studio cache is diagnostic
evidence only. `tools/deferred_head_probe.py` defaults to preflight-only; rendering
requires matching artifact/driver calibration, an admissible complete workspace
plan and a forecast within the remaining cooperative time.

Prepared but unexecuted: six exact CPU/CUDA cases in
`tests/python/test_deferred_cuda.py` with matched cotangents and the separately
named admitted CPU/CUDA resources. They require a serialized CUDA worker with the
2 GiB host/64 MiB GPU caps; a skipped unavailable-CUDA test is not qualification.
Forward-only smaller-block adaptation remains a reviewed proposal awaiting a
separate implementation decision and red/green tests. No block partition, raster
window API, memory cap or backward ordering has been changed by that proposal.

### Resumed forward-only partition checkpoint

The coordinating agent subsequently approved a test-first largest-fitting
four/two/one-point forward partition, preserving the fixed four-point backward
partition and every existing cap. Fifteen tiny cases are authored in
`tests/python/test_deferred_forward_blocks.py`. The installed9 planner red
(`forward-block-plan-red1.json`) preserved the refusal26,896,984>9,166,168bytes;
the separate runtime keyword red is `forward-block-runtime-red1.json`. Both
workers stayed below1GiB. The candidate source now threads the chosen
`principled_point_block_limit` from its complete plan into real shading calls;
only sequential material scratch is recomputed for the4/2/1 admission ladder.
The explicit installed9-plus-source-overlay diagnostic subsequently passed all
15 controls in11.04s (`forward-block-overlay1.json` and XML): whole-worker
time15.6412895s, sampled peak594,743,296bytes, no supervisor failure. The source
SHA256 was `076d473b30dc97712858e2f368746139ae16e8be262bd40f9116b361d0ab53df`
and test SHA256 `85be660979894ec4ba3bed7400884915f68bdec721c5204c7373ad13c69bd785`.
Both input files were copied into the new `forward-overlay1-inputs` evidence
directory before execution, after hash checks. This is source-overlay evidence,
not installed-wheel acceptance; a new exact wheel remains required.

Timing calibration now requires the explicitly recorded selected partition;
previous reports lacking it are not silently inferred as compatible. Four
standard-library guard tests passed in0.23s after the preserved missing-keyword
red (`head-partition-cost-red1.xml`, `head-partition-cost-green1.xml`). The scale
driver accepts a smaller explicit workspace allowance and expected point block
so future one-point16/32 calibration can be measured rather than borrowing the
old four-point timings. This checkpoint does not admit the128-square Principled
image or replace its preserved refusal report.

Source-only review retains the following boundaries: block output pieces and
their final concatenation remain charged in the original full-image lighting
term; smaller partitions do not remove any borrowed source field, active mask,
packet or image storage. Zero-sample packets have zero material-query scratch,
matching the public material's `count=0` rule. Empty/offscreen geometry skips
shader calls but keeps the conservative layout estimate. Back-facing material
pairs do not reduce the prospective scratch estimate. Tests now cover those
cases, singleton-batched/noncontiguous per-point fields and exact matched
cotangents through the unchanged fitted partition. The15-case diagnostic above
subsequently exercised these additions; it does not qualify a larger image.

## Installed13 partition and tiny CUDA checkpoint

On2026-09-10, exact wheel
`4eee522cc433ac92346874a115fee36b73641071657469f1800376ed62b63053`
in `build/m1-team-20260909/installed13` passed the15forward-block CPU controls
in11.32s (`source13-forward-blocks-cpu1.json`/XML): worker16.0900735s,
sampled peak595,853,312bytes, no supervisor failure. This replaces the
source-overlay-only status for these15controls, not for larger rendering profiles.

Before the six device comparisons, an explicit signed-zero comparator regression
failed as intended (`source13-byte-red1.json`,1failure0.42s). The test helper was
then changed from floating tensor equality to shape/dtype-checked contiguous
logical-byte equality, with scalar and empty handling. Its four CPU controls
passed in0.73s (`source13-byte-green1.json`), worker5.3027706s and peak528,789,504B.
The strengthened CUDA test source SHA256 is
`11582dbc2d573287dc89758e0ecacf3f85ce5d6a42c3f4d25a3de694d1ac0bb7`;
old/new test sources and checker inputs are preserved separately in
`source13-byte-red-inputs`/`source13-byte-green-inputs`. No production shader
changed for this test-harness repair.

`source13-cuda6-1.json`/XML preserved2passes followed by1failure in2.86s:
float32/64hard geometry passed byte identity, but classic float32blue colour
differed by1ULP (CPU bytes69,118,116,62; CUDA68,118,116,62). Its albedo-VJP
comparison and the three subsequent device cases were not reached. Worker
time7.6730817s, sampled host peak1,754,165,248B, allocated GPU peak8,563,200B;
limits were respected, but numerical qualification failed. No tolerance was
relaxed and no later scaling worker ran. Lambert division is a source-level
attribution candidate, not an established diagnosis; matched intermediate-byte
evidence is required before any numerical production repair.

The following matched-operand diagnostic isolated that boundary on the same
installed13wheel (`source13-classic-attribution1.json`/XML):2diagnostic controls
passed in2.61s, worker7.7733729s, host peak1,711,652,864B and GPU peak8,571,392B.
Both XML properties are present despite the two xunit2 compatibility warnings.
Float32values and albedo VJPs first diverged at Lambert; classic, integration and
rendering added no downstream same-device difference. Direct Python-scalar
division reproduced the mismatch, while a same-dtype device-tensor denominator
matched both lanes and preserved CPU bytes. Float64matched in all sampled cases.
These are diagnostic findings for the fixed operands, not broad CUDA acceptance.

Based on that evidence, the source14candidate changes only Lambert's denominator
from `math.pi` to `albedo.new_tensor(math.pi)`. Four direct regression cases are
authored for both precisions' device values/VJPs and legacy CPU bytes on a strided
grid including signed zero/endpoints. They are not yet executed. The original
attribution source is preserved in its create-only inputs directory; its live
diagnostic now records which division variant matches the public function so
it can explain either pre- or post-repair builds without requiring the old bug.
No other shader formula, resource allowance or parity tolerance changed.

Exact installed14wheel
`7f0b04abd9c5b3346c55758142789bf27631115b5cb37b1803bf70dca9850379`
then passed all4Lambert controls in1.74s (`source14-lambert4-1.json`/XML),
worker6.8662797s, host peak828,411,904B and GPU peak3,072B. The repair is therefore qualified for
those direct precision/VJP/legacy controls. The next preserved renderer cohort
(`source14-cuda6-1.json`/XML) progressed to5passes before the final Principled
case failed on its scalar roughness VJP by1ULP: CPUbytes23,16,14,5,25,21,216,191;
CUDA22,16,14,5,25,21,216,191. Principled colour and albedo VJP matched before that
failure; F0 comparison was not reached. The cohort took4.56s, worker9.6932627s,
host peak1,912,668,160B within2GiB and GPU peak21,374,464B within64MiB.
No later worker or scaling profile ran. This
is an unresolved exactness failure, not a tolerance-qualified shader result.

`source14-principled-attribution1.json`/XML then passed its1diagnostic control
in6.16s (one xunit2 warning; all three mode properties present), worker11.2073525s,
host peak1,672,585,216B and GPU peak12,104,192B. All budget tensors and fixed
geometry/material/cotangent bytes matched. Direct public Principled already
showed the joint roughness and F0 VJP differences; recomputation and rendering
preserved the direct material's bytes on each device. Colour and albedo VJP
matched. Separately weighted R/G/B roughness VJPs differed by0/1/5ULPs, while
the individual F0 channel VJPs matched. The diagnostic ordered RGB sums do not
repair or redefine the contract: roughness is not explained solely by the final
RGB reduction, and F0's shared-channel reduction remains an attribution lead.
No production repair followed this diagnostic. The next proposed evidence point
is matched incoming cotangents at material-loss and specular subexpressions.

The first captured-stage attempt (`source14-principled-stages1.json`/XML)
stopped on a test-harness error before component replay: a shape[1],stride[3]
roughness snapshot is considered contiguous by PyTorch, but cannot be re-viewed
directly as bytes. The formatter's `contiguous().reshape(-1).view(uint8)` therefore
failed. The public objective property was captured, but no stage-attribution
claim follows. Result1failure3.32s, worker8.3285247s, host peak1,505,513,472B.
The proposed test-only correction is explicit flat same-dtype storage before
byte viewing, with a singleton/non-unit-stride regression. It awaits review;
no production change or automatic retry followed the failure.

## Full-frame scaling boundary (planning, not a measurement)

`python -m tools.deferred_preflight` performs only standard-library integer
arithmetic. It does not import torch, allocate tensors or run the renderer. Its
float64 conservative charged hard-buffer terms are:

| Target | Conservative charged buffer bytes | Interpretation under 64 MiB |
| --- | ---: | --- |
| Fixed synthetic face, 128x128 | 34,078,720 | Full exact preflight still required |
| FLAME-scale target, 256x256 | 136,314,880 | Refused before adding any other component |
| FLAME-scale target, 512x512 | 545,259,520 | Refused before adding any other component |
| FLAME-scale target, 1920x1080 | 4,313,088,000 | Refused before adding any other component |

These are the candidate planner's conservative buffer terms, not actual process
peaks, performance measurements or proof that these bytes are all unavoidable.
Soft coverage plus requested geometric outputs needs a separate hard centre
buffer and doubles this term at supersample one. Colour/mask-only does not.
Neither a partial-estimate pass nor a recorded safe refusal completes a required
large-image rendering profile. The implementation needs a reviewed image-tiled
execution/storage design or other justified planner/storage refinement before
those profiles can run inside the unchanged limit. FLAME topology, material and
light counts must be specified before a complete estimate; the resolution label
does not invent or load a private FLAME asset.
