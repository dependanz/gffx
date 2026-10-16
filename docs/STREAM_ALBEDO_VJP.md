# Explicit constant-albedo streamed first VJP

`tools/deferred_stream.py` is a measurement-only profile, not the public renderer.
Only constant float64 RGB albedo is differentiable; `_profile` rejects other
gradient-bearing inputs. Inputs/bindings are checked between windows.

The prior eager helper is retained verbatim as `previous_partials` in
`tests/python/test_deferred_stream_explicit_vjp.py`. The implementation retains
original <=4-point material layouts, verifies the derivative-bearing workspace
requirement even under `no_grad`, and calls the same installed native grouped
full Jacobian and full VJP with the same error translation. It retains at most
32 original groups, 128 points and 1024 point-light pairs.

Lighting forward executes unchanged for validation and representability checks.
For the fixed albedo objective, its reverse has no geometry/light terms: start
with `ones_like(rgb) * cotangent`, multiply by the mask, broadcast across light
samples, then multiply by measure weight (when present), cosine and incoming
light in that order, and apply the original active mask. No factors are fused.
Native albedo partials are reduced separately with the installed adjacent-pair
tree for each original group; the first group keeps the empty zero-connection
edge. `stream_into` retains the original global reverse accumulation unchanged.

Scalar roughness curve state lives in one complete backward invocation. There
is no persistent cache. All existing retained-input, native derivative and
workspace accounting remains charged. Removing eager graphs does not authorize
a larger window, group or allowance. Complete-job forecasts include cold costs.

Required controls: exact CPU/CUDA partials for tails1/2/3/4 and both cotangent
signs; lit/unlit/two-sided inputs; unchanged invalid-value and backward-admission
errors; injected full-VJP failure; parameter changes; exact full-frame outputs
and global gradient; independent finite differences; mutation/cancellation;
and repeated GPU allocation checks. Final visual quality is a separate gate.
