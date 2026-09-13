# Tensor-resident light budgets: historical research qualification

This page preserves the internal candidate experiments and their failures. The
subsequent public CPU implementation and exact packaged table are documented in
[Principled materials](PRINCIPLED_MATERIAL.md) and
[bundle v1](PRINCIPLED_BUDGET_V1.md). Historical open states below belong to their
dated experiments, not a claim that the current public bundle is absent.

The research resource now keeps constants as CPU float64 tensors and evaluates
whole batches with tensor arithmetic. It lives in `tools/material_loss_tensor.py`,
not `gffx.torch.material`: no public material was added. The frozen scalar model
and its generation code are unchanged.

## What the candidate does

It accepts scalar or explicit field-shaped cosines, with scalar or exactly
matching roughness/Fresnel fields. Shared parameters receive summed gradients.
Separate directional-loss and mean-loss queries avoid repeating a mean per light.
The mean is still the analytic integral of the same angular curve; it is never
detached or recovered by subtracting a rounded energy from one.

Construction is explicit. A strict, non-executable JSON envelope records format,
model, interpolation, dtype, domain, payload hash, generator information and an
evidence label. Unknown keys/versions, duplicate keys, malformed shapes, nonfinite
or invalid loss data and digest mismatches fail before tensor conversion. Files
are limited to8 MiB; node caps remain129/513. Hashes are integrity checks, not
proof of physical accuracy. Caller-provided synthetic data may have an explicitly
unknown generator hash; it does not receive an invented provenance claim.

Inspection returns copies. Queries perform no I/O, generation, constant rebuilding,
device transfer or implicit dtype promotion. Empty and noncontiguous fields are
accepted, but shared inputs are validated even for empty calls. The initial
calibration allows at most32 entries and never silently chunks a larger request.
This is an experimental work boundary, not a permanent public batch-size policy.

## Finite performance demonstration

One CPU thread, float64, Python3.13.5 / PyTorch2.10.0+cu128 on MSI. No GPU work.
One warm-up and three measured samples. Times include validation and result/gradient
checksums but exclude input creation. These are medians, not a frame-time guarantee.

| Points | Coarse forward, ms | Coarse forward+backward, ms | Fine forward, ms | Fine forward+backward, ms |
|---|---:|---:|---:|---:|
| 1 | 1.260 | 3.485 | 1.223 | 2.992 |
| 8 | 2.434 | 5.231 | 1.987 | 4.968 |
| 32 | 3.846 | 9.938 | 3.826 | 10.296 |

The earlier fine scalar-loop baseline took45.668 /93.157 ms at32 points. These
separate short measurements suggest the batching is useful, but are not a
controlled sustained-speed comparison. Construction from a validated envelope
took208.475 /409.306 ms coarse/fine, including hashing and constant preparation;
it is offline work, not hidden query cost or quadrature generation.

Persistent tensor buffers occupy1,810,968 /3,201,048 bytes. At32 points the saved
nonconstant autograd storage was5,121,064 /6,822,952 bytes, under the conservative
planning estimates37,847,040 /50,429,952 bytes and64 MiB target. Saved storage is
not all temporary allocation or allocator peak; the planning estimate is not a
proven OS memory reservation. The monitored timing worker's sampled whole-process
peak was645,410,816 bytes, with elapsed6.6584 s and matching worker/monitor PID.

Report: `build/loss-tensor-20260907/timing.json`, SHA-256
`2f7f6282d08213d33f8479265c6dfd3818dfd5e90f60e1988252da730da109aa`.

## Tests and the stop that prevents admission

The32-case missing-implementation baseline came first. Six first-iteration failures
were fixture mistakes: five attempted NumPy conversion of gradient-tracking tensors;
one allegedly invalid evidence label was valid text. Correcting those tests yielded
32 passes without changing math or tolerances. Three further missing-probe cases
preceded the independent integrator and bounded probe. The resulting35 candidate
cases plus22 existing measurement/watchdog cases passed together.

All168 existing directional physical comparisons passed unchanged loss, gradient
and coarse/fine limits, across the48/60/60 cohorts. These are regressions, not
fresh holdouts. Their JSON rows were copied with matching hashes from pytest temp
storage to `build/loss-tensor-20260907/directional-evidence/`; the original files
were retained. The highest relative loss error was0.000470052395261211;
the largest derivative error/allowed ratio was0.0866248757864789.

**The independent mean-validation worker crossed the1 GiB process limit.** The
watchdog observed1,088,331,776 bytes and stopped it after3.3514 s. No completed
mean XML or mean comparison rows exist. The exact allocation responsible has not
been profiled; the process includes the candidate, PyTorch and independent oracle.
Do not call this a measured candidate-only memory cost or a numerical failure.
`build/loss-tensor-20260907/mean.json` retains the stop. Furnace validation was
not started after it. No limit was raised and no failing workload was rerun.

### Validation-memory repair, 2026-09-08

The frozen oracle's `average` retained all64 directional autograd graphs until
backward. The validation-only `material_mean_validation.average_value_and_gradients`
now computes each weighted view and both first partials before releasing that graph.
It preserves the original vector node arithmetic, sample order,128-light/64-view
rule and derivative contributions from moving view coordinates and weights.
Only finished measurements are detached; this helper is not a public differentiable
query and makes no higher-derivative claim. Candidate, scalar, generator and original
reference source hashes remain unchanged.

Eleven missing-module failures preceded implementation. The first result had10
passes and an exact-node failure from scalar versus vector transcendental arithmetic;
preserving the original vector arithmetic resolved it without changing tolerances.
Small16-light controls pass both coordinate schemes and parameter derivatives.
Logical saved-reference pressure is3,513,904 bytes for the old16-view graph,
versus220,384/221,152 for streaming16/32 views. This counts repeated references,
not unique storage or RSS.

The original36 mean comparisons now pass: worker32252,27.8476 s total and
613,658,624 sampled working-set bytes. The396 furnace cases also pass:
worker19888,13.6256 s total and620,130,304 sampled bytes. Both keep the1 GiB/180 s
ceilings, original cohorts and tolerances; neither is new unseen validation.
Reports and project-local case rows: `build/mean-memory-20260908/mean.json`
and `furnace.json`, with matching XML and `mean-cases`/`furnace-cases` directories.
The previous stopped report remains preserved rather than overwritten.

Next: review the internal strategy's remaining public resource/material contract
and acceptance gaps. Passing these regressions does not close shading Phase0 step3.
Native kernels, float32, CUDA, compilation/export, larger batches, exact scratch
accounting and public material admission remain unverified. No new dependency,
server restart, branch, commit or push belongs to this experiment.
