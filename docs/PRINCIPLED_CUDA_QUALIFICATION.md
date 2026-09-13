# Principled CUDA float64 qualification

## Named finite-profile bundle

The separately named `ggx-smith-loss-cuda-f64-v1` bundle is locally qualified by
the finite `public-budget-cuda-v0.1` profile. Request it explicitly:

```python
import torch
from gffx.torch.material import PrincipledBudget

gpu = PrincipledBudget.bundled(
    "ggx-smith-loss-cuda-f64-v1", device="cuda:0", dtype=torch.float64)
assert gpu.metadata["admission"] == "accepted"
```

This object needs no experimental flag when passed to `material.principled`.
All material and query tensors must already be on that CUDA device. It shares
the CPU bundle's exact table bytes, not its acceptance identity: the additional
receipt is `gffx/data/principled/public-budget-cuda-v0.1.json`, SHA256
`e34977b2cc9641664210f0307a881fcad996e4d8e27b9c28824b9f761fd8d96e`.
No duplicated large table, download or additional runtime dependency is needed.

The measured environment is Windows x86-64, RTX 5090 Laptop GPU (SM 12.0),
PyTorch 2.10.0+cu128. This is finite empirical qualification, not proof across
every GPU or the continuous material domain. Float32, compiled/exported shading,
whole-image CUDA gradients and sustained real-time behavior remain unqualified.

Generic CUDA envelope/file loading and `to(..., allow_unverified=True)` retain
their experimental semantics, including when transporting a previously accepted
object. To request the named acceptance, construct the named bundle explicitly.
The CPU-named bundle remains CPU-only. Metadata labels supplied in an envelope
cannot turn a generic transfer into this named acceptance.

## Explicit experimental transport

The accepted `ggx-smith-loss-cpu-f64-v1` bundle is still a CPU profile. Moving
its tensors does not grant GPU acceptance. This separate path is experimental:

```python
import torch
from gffx.torch.material import PrincipledBudget

cpu = PrincipledBudget.bundled(
    "ggx-smith-loss-cpu-f64-v1", device="cpu", dtype=torch.float64)
gpu = cpu.to(device="cuda:0", dtype=torch.float64, allow_unverified=True)
assert gpu.metadata["admission"] == "experimental"
```

Use still requires `allow_experimental_budget=True` on `material.principled`.
Every material/query tensor must already be on the resource's device. There is
no query-time CPU fallback or input migration. Validated fixed coefficients move
only during explicit construction. Resource payload serialization is unchanged;
loading a payload cannot claim an accepted GPU profile through its evidence label.
Default unqualified transfers and the CPU-named bundle's CUDA loading still fail.
Float32 resource execution remains unsupported, including with experimental opt-in.

Qualification includes ordinary/grazing value and derivative parity, complete
material derivatives, and separate supervised directional/mean/furnace comparisons
against independent CPU mathematics. The physical test bridge explicitly moves
test inputs and outputs while the public query computes on CUDA; it is not runtime
fallback. Existing query/material entry caps and scratch planning limits remain.
The physical worker separately measures host working set and CUDA allocated versus
reserved memory. Passing a small comparison does not grant broad GPU admission,
compiled/export support or a real-time claim.

Devbrain's shading contract section 8.30 and numbers ledger own the exact artifacts,
measurements and remaining admission requirements. These changes are tested in a
new isolated wheel; ongoing CPU image verification retains its original artifact.

## Recorded local checkpoints

Final explicit-device repair, 2026-09-09: wheel
`d46d826c5da0b83ba7ac7bdff355c80fafa236e9c7a2b271f09b13235855f81d`
passes the same six CUDA stages, admission
`b3e723b077af8f2b15ee80b11116522faed2a5dc3dab70cd2300466c53d31910`
under `build/device-owned-budget-20260909/qualification`. Total72.7276532s;
maximum1,803,005,952B sampled host and21,388,800B allocated GPU. The matching
CPU regression passes368 cases/no skips, with three additional constructor tests
covering an ambient meta-device context. Explicit CPU construction and subsequent
CUDA transfer now override ambient defaults; table bytes and equations are unchanged.
These remain finite local qualifications, not hosted or all-hardware acceptance.

Named admission, 2026-09-09: isolated wheel
`fc6746c5551517c4d5a5678a70f8dc0e57267424fdb9fb8d46d6f26796d5a8f0`
passes the same five physical/chain/repeat stages and six additional named-bundle
checks. The latter verify installed receipt bytes, local CUDA constants, unchanged
experimental transport, exact lane rejection, query partials and unflagged
complete-material use. Admission report SHA256
`6d155d49a94276162eb05a6781a0c9116c918ca488767426f9ff27f5092c80d6`
under `build/cuda-bundle-admission-20260909/qualification` binds all six stages.
Total 73.6329305 seconds; maximum sampled host 1,799,684,096 bytes and allocated
GPU 21,388,800 bytes. CPU image work ran concurrently: these are qualification
timings, not isolated performance measurements. This local evidence does not
claim a published wheel, hosted CI acceptance or completion of M1.

Pre-admission exact-artifact reconciliation, 2026-09-09: wheel
`c1702a82604698fd791fee9e49f5ab6f3c038546091b9a6b4c543215b03d5590`
passes all15 transport/mip-chain cases,80 directional values/partials,16 means,
192 furnace cases and20 repeated backward calls. Five supervised processes use
the same artifact, with every installed member verified. Maximum sampled host
memory is1,801,551,872 bytes; maximum allocated GPU memory21,388,800 bytes, both
within the unchanged limits. Post-warm live growth remains zero in the finite
repeat screen, not indefinitely. Evidence lives in
`build/shading-closeout-20260909/{cuda,cuda-corrected}`.
The first physical stage rejected an incorrectly selected older table directory
before executing tests; that failed attempt and the already-passing chain stage
are preserved. The corrected stages use the original404ef671/c85381a9 tables;
no physical threshold or source binding was rewritten. At that checkpoint CUDA
was experimental pending named admission; the separate qualified route above
does not retroactively relabel those experimental objects or reports.

The first 1 GiB host-memory workers stopped and remain saved. Following explicit
user approval, the CUDA-only allowance is 2 GiB; CPU workers remain at 1 GiB.
Both profiles include framework import/initialization in the sampled working set.
The watchdog stays 180 seconds and allocated CUDA memory stays capped at 64 MiB.

On wheel `4c2b5c5d`, the finite physical screen passes 80 directional, 16 mean and
192 furnace cases. Twenty maximum-pair material backward calls also pass exact
same-device repeatability, resource immutability and live-allocation checks.
Observed post-warm live growth is zero; this is not an indefinite leak guarantee.
The full hashes and original failed attempts remain in devbrain's numbers ledger.

A subsequent texture-chain test exposed the adapter calling CPU-only workspace
queries before reaching CUDA pyramid/sampling dispatch. The device-scoped repair
is in wheel `399245f2`. Full native mip-map-to-material CPU/CUDA outputs and all
input gradients pass there, alongside adjacent CPU regressions. Those newer
results do not silently relabel the older physical/repeat artifact identities.
No new dependency, float32 resource, automatic GPU admission or export claim follows.
