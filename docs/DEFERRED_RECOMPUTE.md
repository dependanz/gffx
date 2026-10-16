# Proposed bounded first-order Principled recomputation

Status: tests written first and the expected six-pixel refusal captured before
the implementation, 2026-09-09. Four tiny CPU float64 controls passed on installed
wheel `3cd26303af427041fe4e58535c03f9a8cef30d970442abb10a3d9ac676f6f8bc`:
exact values/all first partials, saved input storage, two finite differences, and
a six-pixel forward/backward. Evidence: `build/deferred-20260909/recompute-green1.json`
and its XML; four tests took 2.91 seconds and the whole worker sampled peak was
619,819,008 bytes, including Python/PyTorch imports. This does not qualify larger
images, complete renderer workspace accounting or CUDA. The initial installed renderer explicitly refuses more
than four Principled points when autograd is enabled; preserve that red evidence.

The fitted renderer evaluates fixed point blocks of at most four, in frame/material/pixel
order. Per-point AttributeSource and TextureSource values are sliced by the same
selection as normals, view vectors and light packets. Constant scalar/RGB fields
remain shared. A source map and its pyramid remain connected to their current
forward graph; no fitted input or parameter-dependent cache is detached or reused
across renders. Raw field layouts must be checked explicitly, never guessed from
a dimension accidentally equal to the point count.

A small custom autograd Function wraps only the existing Principled material:

1. Forward receives normals, view, directions and every top-level floating material
   parameter as explicit tensor arguments. A deterministic sorted-key mapping
   reconstructs their keyword names. Other arguments, including the immutable
   admitted budget and static flags, are retained as non-tensor configuration.
2. Save exactly those original input tensors with save_for_backward. This retains
   version checks and their graph connections. Run the same public material under
   forward's no-grad context; do not save its large transient material graph.
3. Backward creates temporary leaves from the saved values, preserving the original
   requires-grad mask, enables gradients and recomputes that identical small
   material call. Compute its vector-Jacobian product against the incoming
   cotangent, and return each result in the original argument order. Returning
   these partials reconnects upstream camera/geometry/map/light graphs. Temporary
   backward leaves are not a persistent detached approximation or a lost gradient.
4. Release the temporary graph after that block. All fitted albedo, roughness,
   F0, normals, view and directions must match the existing material's derivatives.
   Budget knots are immutable constants; derivatives through budget queries with
   respect to their roughness/F0/directional inputs remain required.
5. Mark this boundary first-order only. Higher-order differentiation is not an
   admitted profile and must not silently return false Hessians. Borrowed input
   mutation remains prohibited while a graph is live. No random callback,
   resource loading, compiler import, disk access or implicit backward occurs in
   forward.

Acceptance starts with two points against a direct unchunked public-material
implementation-equivalence oracle, not an independent physical reference:
bit-identical values and every first partial, two finite-
difference steps, exact saved-input storage identities, then a six-pixel render
that formerly failed the global point guard. Forward/no-grad blocks may use the
same four-point partition. The later forward-only candidate may select the
largest fitting four/two/one-point partition while preserving bit-identical
per-point values; it does not change fitted-gradient partitioning. See
`DEFERRED_WORKLOAD.md` for its separate qualification status. Memory and
latency are measured under the existing1GiB CPU/2GiB CUDA host,64MiB GPU and
120s/180s worker limits. Only after that checkpoint may64/256 pixel profiles be
predeclared. Chunking does not remove the full renderer's geometry/output/graph
storage limits and is not a blanket1080p or real-time claim.


## Ordered native groups on danziepc, 2026-09-14

The approved renderer-internal bridge keeps the four-point gradient partition and
batches up to32 such groups (128 points,1024 point/light pairs) into one native
call. Existing public material8/4-point and raw32-owner query limits remain.
Capacity is selected inside the already charged material workspace reserve;
this is dispatch grouping, not an increased memory allowance.

One renderer autograd node saves original inputs. Backward recomputes bounded
cohorts, then explicitly adds four-point contributions in reverse group order.
Shared roughness uses the already qualified adjacent-pair broadcast reduction,
matching shared RGB fields. Point slices retain their original owners. No input
is updated between groups, and higher derivatives remain unsupported.

Candidate groups35 passes15 installed-wheel controls: exact ordered-reference
and CPU/CUDA values and all first gradients, independent two-step differences,
saved-input ownership and20 fresh CUDA graphs with stable3212288-byte live
allocation and25089536-byte peak. Complete hard/soft quad shared material and
light VJPs also pass exact device checks in the73-pass integration/regression
batch. Two saved-large-batch report checks skip because explicit measured reports were
not supplied to that worker; these are not passed gates.

Groups33 and34 failed exact shared-gradient comparisons; those artifacts and
reports remain historical evidence. The repaired accumulation does not relax
tolerances, replace the independent physical oracle or qualify arbitrary
caller-created PyTorch broadcast graphs. Larger-frame speed, complete parameter
profiles and final demonstrations require their own measured acceptance.
