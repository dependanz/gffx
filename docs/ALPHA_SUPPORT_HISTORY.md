# Support status

The GPU-capable `0.2.0a2` experimental Windows candidate adds the existing lazy
CUDA provider to the CPU core/optional adapter. See [its exact profile](ALPHA_0_2_0A2.md)
and [GPU build recipe](CUDA_PLUGIN_BUILD.md). Published CPU-only `0.2.0a1` and all
older artifact-specific evidence retain their own scope. Publication availability
is separate from preparation; the reference campaign stays paused indefinitely.


GFFX `0.2.0a1` is an experimental Windows x64 CPU prerelease candidate with
an optional separately selected PyTorch adapter. Its exact candidate/runtime
results are in [ALPHA_0_2_0A1.md](ALPHA_0_2_0A1.md). Package availability and
validation are separate; no full cross-platform support tier is claimed.

The following inventory preserves earlier, artifact-specific development and
foundation evidence. Those historical results do not certify the alpha wheel.
M1 remains complete for the adopted development-readiness scope.

| Surface | Current state | Classification |
|---|---|---|
| `import gffx` and static capabilities | Verified on bounded Windows/Linux/macOS lanes | Internal foundation evidence |
| CPU `cp310-abi3` packaging on Windows x64, Linux x64/ARM64, macOS ARM64 | Four internal wheels built and smoke-tested | Internal foundation evidence |
| Public C operation inventory | Current source inspection finds87 public declarations and88 classified symbols including the private self-test; Gaussian weights and Principled curve/query/material first-VJP entries are included | Verified source inventory, not release support by counting |
| First-order native derivatives | Foundation derivatives plus all five animation operations; `mesh.build_edge_topology` publishes none by design | Measured fixture cohorts, not all-input/platform proof |
| CUDA provider on Windows x64 | Foundation evidence retained; all five animation operations pass the source7 CPU/CUDA forward/backward byte-comparison cohort | Internal evidence on **one** host only |
| CPU/CUDA bit-identity across hosts, drivers, or architectures | Laptop RTX5090 and PC RTX3080 evidence are separate artifact-specific cohorts; cross-host identity has not been measured | Target, not support |
| CUDA hardware lane | Current source7/native7 local run passes 51 CTests with no skips; previous CI results retain their own revision | New M1 artifacts are not requalified by historical CI |
| CUDA release artifacts | Not built as release candidates | Target, not support |
| Linux GPU lane | Not provisioned | Target, not support |
| PyTorch Stable-ABI adapter | Current PC execution uses Windows x64, CPython3.14.6 and PyTorch2.10.0+cu126; older laptop combinations retain their own evidence | Internal development evidence, not the full support matrix |
| Camera, scene and shading compositions | Local M1 cohorts recorded separately from the native primitive inventory | Experimental; see authoritative acceptance records |
| Deferred renderer | Complete native Principled first-VJP and small device/gradient controls pass; valid5002V/10000F scale profiles expose larger resource limits; soft-coverage arithmetic is separately qualified | Experimental; large-frame and final visual gates remain open |
| Image pipeline | Bounded CPU/CUDA image and recovery cohorts plus explicit1080p caller-chunk evidence are recorded; that chunk result is not general renderer tiling | Experimental; complete matrix owned by project records |
| Readiness for starting gffit and Compositron | M1 development readiness complete: finite generic supply, exact recorded profiles, CI-01-CI-07 and B1 public CPU fitting integration | Scoped milestone; full consumer completion, original-stack parity, universal current-wheel support and photograph likeness are not claimed |
| CPython 3.10-3.14 × PyTorch 2.10-2.13 full matrix | Adapter *load* covered by hosted lanes; adapter *operation* measured on one combination | Target, not support |
| PyTorch 2.9 | Not supported; the adapter refuses to load below 2.10 | Unsupported |
| JAX CPU/CUDA adapters | Not implemented | Target, not support |
| Spatial acceleration and real-time profiles | Conservative CPU raster block rejection and CUDA outside-distance rejection qualified on the PC; original candidate equations and ordering retained | Sustained real-time and edge profiles remain unqualified |
| New native PyPI distribution | `0.2.0a1` preparation; see exact alpha profile and package index for availability | Experimental; publication separate |

## Historical PC raster and grouped-autograd checkpoint, 2026-09-14

The raster41 wheel passes39 exact-output/window/stream checks and85 affected
regressions. The latest edges43 wheel retains that native implementation and
passes57 grouped-gradient/stream controls. It skips copies for unrequested
autograd outputs while still evaluating the complete native Jacobian and errors.
Both wheels are local development artifacts, not published releases.

Saved CPU512 hard/soft forward jobs improve from42.8518872/42.9297443 seconds to
11.3084861/11.0143612 seconds with unchanged image bytes. CUDA256 hard/soft forward
plus albedo backward on edges43 takes4.1932615/4.2035898 seconds and matches CPU
image, mask and gradient bytes. These are single evaluations of the synthetic
5002-vertex/10000-face fixture, not sustained throughput or broad speed guarantees.
Ten of24 distinct large profiles now have complete results across explicitly
recorded artifacts; remaining512 backward/CUDA and1080p profiles still lack
admission or qualification. Original worker, workspace and forecast limits remain.
Final visual acceptance and expanded M1 closeout are unfinished. The dashboard's
512 comparison is engineering evidence; it does not claim SIGGRAPH-quality output.

## What the evidence does and does not cover

The configured hosted foundation matrix is CPU-only. Its historical load/packaging results
belong to the revisions and artifacts that actually ran, not today's dirty M1 checkout. The
separate hardware lane must verify required GPU cases by name and reject skipped or missing
cases. Its current local measured floor is 51, replacing the old 40-test inventory; CPU-only
build counts must be read from their own reports, not derived from this number.

The source7 checkpoint has 36 passing installed animation CUDA tests and a matched native7
51-test pass. Native7 names the RTX 5090 Laptop, driver 595.79, Toolkit 12.9.41 and capability
12.0. These are local development-toolkit results, not CUDA 12.8 release-artifact qualification.
Reference-mode comparisons use bytes rather than a tolerance. A single measured cohort does
not prove identity for every input, architecture or transcendental implementation. Earlier
sanitizer reports remain earlier evidence; no fresh sanitizer claim is made for M1 additions.
See [the CUDA hardware lane](CUDA_HARDWARE_LANE.md).

For reproducibility, local build reports are retained under `build/m1-team-20260909` and the
lane-specific `build/animation-20260909`, `build/image-20260909` and
`build/deferred-20260909` directories. Those generated artifacts are not distributed as source.
Devbrain's gffx `PLAN.md` and `NUMBERS.md` record complete hashes and superseded failures.
Source8 includes the targeted blur and scalar-colour binding repairs; implementation is not
automatically a green result on that new artifact.

Some execution boundaries and selected costs have been tested. An expected rejection of
compilation/export is not working export, and a tiny timing is not sustained throughput,
jitter or callback safety. No complete real-time, edge-device or cross-platform operation
matrix is claimed. Higher-order derivatives and compiled paths remain unqualified unless
their exact operation/profile has dedicated evidence.

Public PyPI `0.1.x` artifacts are an inherited prototype and are unsupported by the new contract.
The repository now prepares the `0.2.0a1` alpha; historical rows retain their original artifacts. This does not change
the identity or support status of the public `0.1.x` artifacts.

See the devbrain support matrices for the normative planned gates.


## Historical local PC native-curve checkpoint — 2026-09-14

The unpublished `nativecurve23` development wheel integrates C11/CUDA loss-curve
construction and its first-coordinate VJP into `PrincipledBudget` and the material
graph. The rest of the material is still partly eager. The three new shading.h
entries use caller-owned scratch and an append-only provider tail; no query,
material-point,point/light-pair or numerical limits change. See
[the native design](PRINCIPLED_NATIVE_DESIGN.md) for exact ownership and gradient
scope. Public-surface inspection covers77 declarations and78 classified symbols.

The integrated predecessor passes154 affected checks and576 physical comparisons.
The CUDA scheduling candidate passes11 device/ABI/material controls and288 physical
comparisons,exactly matching the retained CPU rows,and20 finite fresh graphs without
postwarm live-allocation growth. These are PC development results,not a release,
new hardware architecture or sustained-performance claim. Current64-square CUDA
Principled calibration takes11.76822 seconds;larger-frame time admission remains
unsatisfied. M1 and its final polished renderer demonstrations are unfinished.
Devbrain PLAN.md and NUMBERS.md retain exact artifacts,skips and failed reports.


## Current local PC material and soft-coverage checkpoint

The unpublished nativepoint29 family completes existing Principled material values
and first VJP through C11/CUDA and the Stable-ABI adapter. Public8-forward/4-gradient
points,1024 point/light pairs and32 standalone query entries remain unchanged.
Four-point renderer recomputation remains in place. Independent composition and
exact CPU/CUDA shared-input tests pass; a captured eager CPU sqrt difference is
retained with a90-digit certificate rather than treated as mathematical ground truth.

Soft31 uses fixed-rounding logistic arithmetic and an explicit tensor divisor for
soft coverage. Its32-square scale-mask identity and26 affected compositor tests
pass on installed wheeldc0e107ed4ec39d5bf5e4fafd42e7f1a5a9530e4db8fe97a3e04cd0613f77b24.
Standalone sigmoid checks cover523 samples per dtype with independent high-precision
values and exact device VJPs. The blur model,rounded-output derivative convention,
precision and numerical acceptance criteria are unchanged.

The separately registered synthetic_flame_m1 input has5002 vertices and10000
outward nondegenerate triangles. It is first-party procedural geometry,not licensed
FLAME data or a fidelity reference. Soft31's24 larger refusals remain historical
evidence. Groups35 subsequently passed two CUDA256 forward profiles.

On2026-09-14 installed validate39 wheel
7acf671ee7a8e42aa8db2901b831a690a1573478a1af97b0d843bc334c560022
passes53 grouped/storage/validation controls and107 public/integration checks
(two saved-report environment checks remain explicitly skipped). Native temporary
accounting and fitting cohort selection keep the original limits;scalar curves
and complete-value validation are reused only within each evaluation. All16
CPU/CUDA hard/soft32/64 forward/backward scale profiles pass. Saved complete
colour/mask and albedo-gradient bytes match across devices and historical forward
pixels are unchanged. This does not qualify every fitted scene parameter.

Three of24 larger profiles pass current preflight and execute:CUDA256 hard forward
1.3736885 s,hard forward/albedo-backward5.2466243 s,and soft forward1.4012021 s.
The other21 profiles remain refused by unchanged storage/time guards. Historical
256 forward images match the new saved images exactly. Larger-frame execution,
independent full-image/all-input gradients and polished visual acceptance remain
open. The phone dashboard retains the engineering plates and all failures.
M1 remains unfinished;readiness is to START gffit and Compositron,not implement them.
