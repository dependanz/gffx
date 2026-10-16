# Bounded opaque sample composition

The C ABI and PyTorch `shading_opaque_samples` entry compose the existing binary64 perspective interpolation, normal normalization, Principled value, single directional illumination and adjacent-pair integral. This is an execution change, not a new coverage or shading model.

The input has 1 through 16 pixels and exactly 800 slots per pixel. Geometry and prepared scalar-roughness curve inputs are borrowed dense arrays. Owners and triangle indices are int32; every other input is float64. Invalid layouts, devices, domains, aliases and insufficient scratch are refused. CUDA providers without the append-only operation are refused by the workspace query before dispatch.

Each pixel processes 128 sample slots per cohort. CPU evaluates them serially; CUDA uses 128 lanes per pixel and a barrier between cohorts. Sample values are independent. Both reduce the same 800 ordered RGBA slots with the original adjacent-pair tree; there are no floating-point atomics or reordered sums. Signed zero and subnormal weights remain part of that arithmetic.

Caller scratch is `8 + pixels * 800 * 4 * 8` bytes, aligned to eight bytes. The first eight bytes carry status; remaining space is the per-pixel reduction buffer. The weighted output occupies `pixels * 800 * 4 * 8` bytes and RGBA occupies `pixels * 4 * 8`. At 16 pixels these sum to 819,720 bytes. The renderer charges a separate 1 MiB reserve for this inventory and small packet/camera packing, within the existing 64 MiB allowance. Existing geometry, prepared material curves and retained inputs remain charged by their prior reserves; input layouts requiring full geometry copies use the eager path.

The production selector requires one batch, one constant RGB Principled material with scalar roughness and an accepted budget, one directional sample, and the qualified four-point/128-point workload profile. Other material/lighting/layout profiles keep the eager path. Preparation is owned only by the current guarded forward window and checked for tensor-version changes. Native domains are validated on every call. General first-gradient replay retains its established composition and canonical pixel accumulation. The explicit fixed-albedo path is described in the later qualification sections below.

Qualification is incremental: opaque209 preserves 220 saved-region calls per device but exposed avoidable execution cost; opaque210 preserves all those bytes with 40 focused CPU and 40 CUDA controls plus old-provider refusal. These region and C ABI checks do not certify full frames, performance admission, geometry gradients or M1 completion. Opaque211 adds renderer-selection and storage/error/replay checks on a fresh installed wheel before new frame trials. Historical failed reports remain preserved.

## Forward window capacity

Opaque213 proposes using the existing legacy one-pixel sample-buffer reserve for a native window of at most 16 pixels. The legacy reserve is 800 times (256 float64 values plus 32 bytes), or 1,664,000 bytes. The native producer retains at most two complete quadrature arrays and three owner arrays: 16 times 800 times (two times four float64 values plus three int32 values), or 972,800 bytes. A further 4,096-byte allowance covers small returned arrays and packing. Weighted outputs and reduction scratch remain charged to the separate 1 MiB native reserve. Geometry, prepared curves, immutable inputs and caller output/cotangent arrays retain their existing independent reserves.

This bound is valid only when multi-pixel eager sample composition cannot occur. Unsupported native profiles and native errors therefore raise the existing canonical-pixel retry before eager allocation or eager diagnostic evaluation. Each one-pixel retry retains the full prior legacy buffer/lighting/material reservation. Exact positive-zero empty reductions are the only multi-pixel alternative. Batch layouts and gradient-bearing plans retain the old planner. Admission never increases the 16-pixel native capacity, per-pixel work ceilings or 64 MiB total. Qualification must check this fallback behavior and actual CUDA allocations as well as exact numerical results.


## Optional CUDA pixel lanes

Ordinary forward quadrature can process up to four independent pixels concurrently,
with one private one-MiB polygon slab per lane. The C workspace query still returns
the one-MiB minimum; larger caller-owned storage selects up to four lanes, while
minimum storage remains serial. Backward, boundary and continuation stay serial.
The framework adapter requests four MiB only when its budget fits scratch and outputs.
Public quadrature and scene planners charge the additional three MiB inside64MiB.
Domain validation and output clearing complete before lane execution. Each lane
uses the same scalar count, emission, work limits and arithmetic at the exact global
pixel origin. Outputs and scratch never overlap; only integer error status is atomic.


## Bounded explicit albedo replay

`render.opaque_albedo_replay` is a first-VJP composition for at most 256 pixels,
one fixed batch1 float64 scene, one accepted constant Principled material and one
directional sample. It returns colour, mask and an RGB-albedo gradient carry.
The optional incoming carry is copied; each pixel gradient is then added in the
same ascending order used by the host's individual-pixel replay. Chunk subtotals
are never added. Cotangents, the carry, scene tensors and camera remain borrowed
and guarded against mutation. Geometry is prepared only inside the current call;
no preparation survives return. Unsupported derivative targets are refused and
ordinary `opaque_window` remains the general autograd path.

The output, input adjoints, borrowed cotangents and temporary carry are charged
to the explicit64MiB budget. Full material Jacobian/VJP checks, per-pixel work
limits and canonical continuation remain in force. Qualification of this entry
alone is not original-frame or M1 acceptance.


## Prospective bounded native albedo VJP (opaque219)

The additive raw operation accepts1..16independent800-slot pixels and one RGB
cotangent per pixel. It computes the existing full16-input material Jacobian,
checks every derivative, applies the qualified delta-light reverse factors, then
compacts covered points in original order. Four-point adjacent-pair sums and
reverse group accumulation match the established eager material VJP. Pixel or
continuation-batch carries remain the caller's unchanged responsibility.

Workspace is8+pixels*5600*8bytes:3200RGBA reduction values and2400sample gradient
values per pixel. Explicit outputs are pixels*(800*4+4+3)*8bytes. Thus raw scratch
plus outputs total8+pixels*8807*8bytes (70464bytes for one pixel;1127304for sixteen),
excluding borrowed inputs and bounded thread-local native arithmetic. Production
selection is not yet connected; it must initially use one pixel inside the
existing native and material reserves, guard fixed inputs and avoid prepared
curves on autograd contexts. General/continuation differentiation remains eager.


Opaque220 connects the raw VJP only through an explicit private cotangent passed
from `opaque_albedo_replay` to the one-pixel ordinary sampler. The returned direct
albedo VJP is carried in ascending pixel order. General autograd and overflowing
continuations receive no such selection. Native errors remain errors after the
original diagnostic checks; unsupported layouts retain eager behavior.

Each call borrows and guards fixed inputs. The same single prepared curve is
shared with existing material groups, and a constant directional packet is kept
only in the guarded call. Neither is saved on an autograd context. Native outputs
are detached; one-pixel scratch/output total70464bytes fits the existing1MiB
native reserve. The existing one-pixel eager material reserve remains available
for unsupported/continuation fallback, with no broad tensor or memory-limit change.


## Eight-pixel explicit replay groups (opaque221)

The proposed production grouping uses at most eight independent ordinary pixels.
Native scratch and explicit outputs require563656bytes, below the existing1MiB
native reserve. Quadrature and owner arrays require at most490496bytes, below
the retained1664000byte sample-buffer reserve. No eager multi-pixel composition
is permitted. A refused group retries all canonical pixels; native errors still
propagate. The existing eager material/light storage remains reserved for those
fallbacks. Per-pixel integration/work caps and total64MiB are unchanged.

Each returned pixel gradient is added in the original ascending order. Curve-only
fixed-geometry unions use the qualified per-pixel quadrature selection; linear
horizons retain canonical fallback. Input guards, one call-local curve, old
continuation semantics and graph-free outputs remain. Tests and fresh frame
trials must qualify this integration before claiming improved frame readiness.


## Prospective direct fixed-albedo continuation (opaque222)

Only explicit fixed-albedo replay may compose one continuation batch's weighted
RGBA and full checked native albedo VJP together. It carries individual RGBA
samples with the same ordered primitive, then adds batch gradients in ascending
order as the retained general backward does. Geometry and curve preparation are
call-local and guarded. Receipt prefixes,16batches/512blocks/12800slots and the
2million aggregate clipping-work check remain. Unsupported native profiles retry
the canonical complete pixel with the same work counter, including the attempted
batch. Native errors are never converted into successful fallback.

The native70464byte scratch/output and existing one-batch contribution reserve
remain within their prior charges. No contribution arrays or geometry survive on
autograd contexts in the direct path. General differentiation is unchanged.
This proposal requires installed-wheel exact saved-pixel, error, receipt and
fallback controls before full-frame qualification.
