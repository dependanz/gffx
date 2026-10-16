# Requested grouped autograd edges

Qualified edges43 implementation,2026-09-14. Preserve the full native16-input
Jacobian and first VJP,including failures in unrequested derivative columns.
Only final per-group tensor copies change: return None for an autograd input
whose needs_input_grad flag is false. Requested fields use the same slices and
contiguous copies. Group sizes,shared reductions,reverse order and native calls
remain unchanged. This is not a selective mathematical derivative evaluator.

The125-point,32-group red measures155 actual discarded/retained storage copies;
comparison helpers are outside the measurement. Albedo-only evaluation needs at
most one copy per group. Exact original four-point partials and all-input grouped
regressions must pass on CPU/CUDA against a freshly installed wheel. Existing
storage accounting remains conservative;no quota or workspace estimate decreases.

Installed-wheel qualification passes57 CPU/CUDA controls, including the32-copy
bound, exact original albedo partials and all-input gradients. Sixteen matching
small calibrations pass. Complete CUDA256 hard/soft forward plus albedo backward
takes4.1932615/4.2035898 seconds; saved images, masks and gradient bytes match
the CPU streamed references. These are single-job measurements, not throughput
guarantees. Larger time-refused and memory-refused profiles remain unfinished.
