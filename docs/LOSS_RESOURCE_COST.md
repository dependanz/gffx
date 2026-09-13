# What does the light-budget research prototype cost?

Scalar CPU float64 baseline, not a public material or a vectorized kernel.
MSI / Windows11, Python3.13.5, PyTorch2.10.0+cu128, one CPU thread; no GPU work.
The real interpreter uses the existing virtual environment's module search paths.

## Loading is not generating

Two hash-verified saved grids were reused. Loading measures warm-filesystem read,
hashing, JSON parsing and shape checks. Construction measures tensor conversion,
physical validation and cubic constants. Neither measures quadrature generation.

| Quantity | Coarse97x385 | Fine129x513 |
|---|---:|---:|
| JSON bytes | 1,700,938 | 3,009,752 |
| Raw grid float64-equivalent bytes | 601,376 | 1,063,968 |
| All constants float64-equivalent bytes | 1,807,120 | 3,195,920 |
| Python object bytes, not RSS | 12,630,235 | 22,344,891 |
| Load median, ms | 33.518 | 54.244 |
| Construction median, ms | 77.319 | 168.919 |

The prototype stores tuples, not persistent tensors. Float64-equivalent bytes
count numbers packed as doubles, not allocated tensor or application memory.
Python accounting includes deduplicated object overhead, not query temporaries.

## Scalar-loop demonstration

All point graphs remain alive until backward through the summed directional and
mean losses. Inputs are prepared outside timing; validation and result/gradient
checksums are included. One warm-up and three measured samples, medians below.

| Points | Coarse forward, ms | Coarse forward+backward, ms | Fine forward, ms | Fine forward+backward, ms |
|---|---:|---:|---:|---:|
| 1 | 1.222 | 2.656 | 1.627 | 2.874 |
| 8 | 9.278 | 19.964 | 10.310 | 21.561 |
| 32 | 37.310 | 82.101 | 45.668 | 93.157 |

An accurate scalar research tool is not automatically an interactive per-pixel
shader. It rebuilds tensor constants from tuples and repeats curve work per point.
Tensor-resident constants and true batching are the next candidate, not a measured
speedup. The exact mean must still follow that curve and retain parameter gradients.

## Safety and provenance

The corrected worker completed in5.1499 s. Its50 ms sampled working-set peak was
631,889,920 bytes (about603 MiB), including Python/PyTorch, parsing, resources and
graphs. This is neither incremental resource memory nor an exact peak. Worker and
monitored PID46204 agree. Limits:120 s cooperative,180 s watchdog,1 GiB sampled
working set,8 MiB/file and unchanged axis/sample/accuracy caps. No regeneration.

First attempt `build/loss-cost-20260907/measured.json` watched the Windows venv
redirector, not the actual PyTorch worker: its4,210,688-byte safety measurement
is invalid and preserved. A new red regression preceded launching the real
interpreter with the same module paths and verifying worker/monitor identity.

Corrected report `build/loss-cost-20260907/measured-verified.json` SHA-256:
`00d2058238f93563caa3886139e73eea212be9f5239c6d53704c4da6571d2962`.
Tool hash at execution:
`3e1a4da7f43c1060b9512bf625cb3ad965cae693d2b9214c471e12edbf961b33`.
The report records fixture hashes, frozen math hash and environment. Reproduce:
`python -m tools.material_loss_cost --fixtures DIR --output NEW_FILE`, with the
recorded fixture directory and a new output path. Existing reports are refused.

No new dependency: stdlib plus the already optional PyTorch environment. No
float32/GPU, optimized batching, compilation/export, public material, wider
physical accuracy, fresh generation time or sustained frame budget was measured.
The proposed resource contract lives in devbrain's shading contract8.11. It
requires new red fixtures and unchanged physical screens before implementation
can be accepted. These tables are a finite demonstration, not a renderer claim.
