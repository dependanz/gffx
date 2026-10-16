# Deformation and skeleton operations

These experimental M1 operations are implemented in the C11 core, with thin
PyTorch validation and first-order autograd bindings. Qualification status is
tracked separately: implementation alone does not establish CUDA, export,
performance or release support. They add no new mandatory runtime dependency.

```python
import gffx

shaped = gffx.mesh.blend_shapes(template, deltas, coefficients)
rotation = gffx.transforms.rodrigues(axis_angle)
world = gffx.skeleton.forward_kinematics(local_matrices, parents)
deformed = gffx.mesh.skin_lbs(shaped, skinning_matrices, skin_weights)
landmarks = gffx.mesh.evaluate_barycentric(deformed, faces, face_index, barycentric)
```

The example names the operations, not a complete model loader. The caller builds
local matrices from rotations and joint offsets and composes `world @ inverse_bind`
where appropriate. No licensed model, joint regressor, rest-pose correction or
asset is supplied or silently inserted by gffx.

| Operation | Input shapes | Output |
|---|---|---|
| `blend_shapes` | template[V,3], deltas[K,V,3], coefficients[B,K] | [B,V,3] |
| `rodrigues` | axis_angle[...,3] radians | [...,3,3] |
| `forward_kinematics` | local[B,J,4,4], parents[J] | [B,J,4,4] |
| `skin_lbs` | vertices[B,V,3], transforms[B,J,4,4], weights[V,J] | [B,V,3] |
| `evaluate_barycentric` | vertices[B,V,3], faces[F,3], face_index[N], barycentric[N,3] | [B,N,3] |

Inputs are dense, C-contiguous float32 or float64 tensors; topology uses int32.
All operands share dtype and device except integer topology. Unsupported devices
do not trigger a hidden transfer. Explicit B dimensions must agree exactly.
Corrective shapes, parents, skin weights and landmark embeddings are shared
across frames as indicated above; no other broadcasting occurs.

## What the equations mean

Blend shapes add a weighted displacement to a template. A coefficient of one
adds its delta; it does not discard the template. Coefficients may be negative.
The template, deltas and coefficients can all receive gradients.

Rodrigues maps a right-handed axis-angle vector to an active rotation acting on
column vectors. Its magnitude is an angle in radians. At zero rotation the
output is exactly identity but the derivative is not zero: a tiny angle still
causes a tiny rotation. The small-angle branch prevents division by zero without
detaching or discarding this derivative.

Forward kinematics multiplies `world[parent] @ local[joint]`. Root parents are
`-1`; several roots and out-of-order joint storage are supported. Cycles and
invalid parent indices fail. General 4x4 matrices are accepted and differentiated;
consumers requiring affine matrices enforce that additional constraint.

Linear blend skinning transforms a vertex by every joint's upper3x4 matrix and
takes the weighted sum. There is no perspective divide, automatic normalization
or bind-pose correction. Conventional weights are nonnegative and sum to one,
but the primitive accepts signed unnormalized coefficients so learned weights
have an unconstrained derivative. A zero-weight vertex becomes zero, not its
original position. The unused fourth transform row has zero derivative.

Barycentric evaluation selects one triangle per landmark and combines its three
vertices using the supplied coefficients. It requires no camera or image.
Coefficients are not normalized or clipped, so extrapolation is explicit. Both
vertices and coefficients have gradients; face identities are discrete.

## Limits

First-order reverse-mode only. Higher derivatives, automatic mixed precision,
compilation/export and real-time callback safety are not implied. A numerical
failure is not sanitized into plausible geometry. Callers must not mutate tensors
while a saved backward graph uses them.

Skinning is a deformation rule, not a physical simulation. Authored motion and
demonstration clips do not implement dynamics, constraints or collision response.
Dual-quaternion skinning, inverse kinematics, sparse influences and inverse SO(3)
remain later work. Independent references, exact examples and gradients are
tested in `tests/pytorch/test_animation.py`; direct native validation and workspace
controls are in `tests/geometry/test_animation.c`.

## Local execution evidence, not release support

The 2026-09-09 Windows development qualification used the installed wheel with
SHA256 `6925bb3c0106b41f7d1d13c211059a371fa91af85eddcbee0f6f597ae3d30814`.
The 117-case CPU cohort (operations, edges, validation, recovery and presentation
controls), six additional independent shared-gradient oracles and the 71-case
CUDA cohort passed. The additional oracles compare branching-parent, shared skin
weight and shared landmark-coefficient derivatives with independently expressed
matrix products, einsum and indexing across two batches in both precisions.
Six focused gradchecks subsequently passed with the explicitly adopted
eps=1e-6, atol=2e-5, rtol=2e-4, replacing reliance on PyTorch's default tolerances.
The separate completion cohorts passed 26 CPU and 26 CUDA controls: repeated
backward bytes, nonidentity inverse-bind composition and all eight continuous
parameter groups, finite-input arithmetic overflow, and every operation's
preallocated alias/scratch/inference-only rejection. Exact test sources are
preserved beside `strict-gradcheck1.json` and `completion-{cpu,cuda}1.json` under
`build/animation-20260909`.
The CUDA composed-chain test emitted a PyTorch cuBLAS primary-context activation
warning during its matrix-product backward. Numerical checks passed; this is
not evidence of context-free or callback-safe execution. No production warmup
was added to hide it.
The CUDA cohort compared identical input and cotangent bytes, including every
operation's floating-input backward and special-angle Rodrigues cases. This is
one installed artifact/device combination, not proof for every compiler or GPU.
Authoritative reports and later qualification results remain in devbrain's numbers
ledger; final visual publication remains a separate gate.

| Execution surface | Implemented behavior | Evidence boundary |
|---|---|---|
| CPU float32/float64 public functions | All five operations, first-order autograd | Small independent references and gradient fixtures |
| CUDA float32/float64 default reference | All five operations and backwards | 71 core cases plus 26 completion controls; identity only for measured fixtures |
| Caller-owned inference `_out` | Five outputs plus explicit workspace queries | All five reuse outputs on CPU and caller-selected CUDA stream |
| Current CUDA stream | Tensor-owner context is scoped/restored; work uses caller stream | Local lifecycle/native checks, not callback-safety certification |
| Native relaxed opt-in | May reuse deterministic arithmetic | No distinct speed guarantee; costs measured separately |
| Public relaxed selector | No animation-specific selector | Do not infer one from the private/native context flag |
| PyTorch compile/export, higher derivatives, AMP | Not qualified here | Do not advertise from eager or C ABI results |
| JAX, other hosts, hosted CUDA | Not qualified by this work | Separate adapters and platform gates remain |

### Requirement-to-evidence audit (2026-09-10)

This maps AN-01 through AN-15 to observed fixtures, not to a coverage percentage.
All Python reports below identify the same source10 wheel above and verify its
63 installed members. Paths are relative to the gffx repository; local build
artifacts are not distributed as source documentation assets.

| Evidence key | Exact report/source |
|---|---|
| CPU | `build/animation-20260909/cpu4.json` and `.xml`: 117 tests, with per-file source hashes |
| GPU | `build/animation-20260909/cuda3.json` and `.xml`: 71 tests in `tests/pytorch/test_animation_cuda.py` |
| Shared | `build/animation-20260909/shared-vjp1.json` and `.xml`: 6 tests in `tests/pytorch/test_animation_shared_gradients.py` |
| Strict | `build/animation-20260909/strict-gradcheck1.json` and `.xml`: 6 adopted-tolerance gradchecks; preserved source in `strict-gradcheck1-source/tests/pytorch/test_animation.py` beside the report |
| Completion | `build/animation-20260909/completion-cpu1.json` and `completion-cuda1.json`, with XML: 26 tests each; preserved files/helpers under `completion1-source/tests/pytorch/` beside the reports |
| Native | `build/m1-team-20260909/native10-ctest.xml` and `native10-hardware-lane.json`: 51 native tests, no skips, including the five `cuda.animation.*` hardware entries |
| Import | `build/m1-team-20260909/base-import10.json` and `.xml`: 12 tests in `tests/python/test_base_import.py`, including fresh-subprocess import checks |
| Scale | `build/animation-20260909/scale-schedule1.json` and its 12 named create-only reports |

In the following table, ordinary unqualified CPU/Strict animation test names
belong to `tests/pytorch/test_animation.py`; GPU and Import names resolve to
their files in the evidence key. Other files are named explicitly.

| Requirement | Specific checks and evidence | Local acceptance interpretation |
|---|---|---|
| AN-01: lazy/public symbols | `test_animation_public_symbols` (CPU); `test_base_import_pulls_in_no_forbidden_module`, `test_base_import_loads_no_native_library`, `test_base_import_does_not_import_the_adapter_namespace` (Import) | Public reachability and fresh dependency-free base import verified; historical absence baseline remains preserved. |
| AN-02: tensor/error/storage contract | `test_animation_validation.py::test_animation_each_public_entry_rejects_invalid_tensor_contract` (CPU); `test_animation_completion.py::test_animation_finite_arithmetic_overflow_is_explicit` and `test_animation_each_out_rejects_alias_scratch_and_gradients`, plus CUDA counterparts (Completion); `tests/geometry/test_animation.c` alias, count-overflow, header and sentinel checks (Native) | Every public operation has representative rejection controls; shared native validator call paths provide reuse. Not an exhaustive malformed-descriptor Cartesian product. |
| AN-03: blend shapes | `test_blend_shapes_closed_form`, `test_blend_shapes_all_gradients`, `test_blend_shapes_empty_basis_and_empty_batch` (CPU/Strict); `test_animation_empty_reductions.py` and CUDA counterpart (CPU/GPU) | Signed/zero/unit coefficients, learned basis, shared frames and empty reduction behavior verified. |
| AN-04: rotation domains | `test_rodrigues_zero_and_quarter_turn`, `test_rodrigues_arbitrary_leading_dimensions` (CPU/Strict); `test_animation_edges.py::test_rodrigues_rotation_invariants_and_signed_axis_turns` (CPU); special-angle GPU checks | Zero, signed axis turns, arbitrary/near-half/full turns and empty/leading dimensions covered. Finite overflow is separately covered by Completion. |
| AN-05: rotation branch/Jacobian | `test_rodrigues_threshold_matrix_exponential`; `test_animation_edges.py::test_rodrigues_threshold_gradients_match_independent_matrix_exponential` and `test_rodrigues_zero_jacobian_is_exact_skew_generators` (CPU) | Both dtype threshold neighbors, independent matrix-exponential reference, exact zero generators and rotation invariants verified. |
| AN-06: hierarchy | `test_forward_kinematics_scrambled_forest`, `test_forward_kinematics_invalid_hierarchy`, `test_forward_kinematics_deep_nonrecursive_hierarchy` (CPU/Strict); native hand-computed chain and GPU invalid-hierarchy sentinels | Roots, forests, arbitrary parent order, branching and a 2048-joint nonrecursive chain covered. |
| AN-07: shared ancestor derivatives | `test_animation_shared_gradients.py::test_fk_branching_shared_ancestor_vjp_matches_independent_matmul` (Shared); forest full-entry gradcheck (Strict) | Independent two-batch branching adjoints and all matrix entries verified. |
| AN-08: skinning | `test_skin_lbs_one_joint_and_all_gradients`, `test_skin_lbs_bottom_row_is_not_perspective_division`, `test_animation_random_independent_references` (CPU/Strict); `test_animation_edges.py::test_skin_lbs_signed_and_zero_influences_have_literal_meaning`; Shared multi-batch weight VJP; recovery rotation | One-hot/identity/translation/rotation, dense signed/zero influences, shared weights and zero ignored-row derivatives covered across these fixtures. |
| AN-09: landmarks | `test_barycentric_centroid_and_gradients` (CPU/Strict); `test_animation_edges.py::test_barycentric_repeated_vertices_and_edge_extrapolation`; Shared indexing VJP; recovery vertex embeddings; GPU invalid selected-face/vertex sentinels | Corners, midpoint, centroid, extrapolation, repeated indices/requests and multiple frames covered. |
| AN-10: all floating backwards | Five individual gradchecks (Strict), `test_animation_gradient_selection.py` (CPU), shared VJPs (Shared), empty reductions (CPU/GPU), all-input GPU parity | Every floating input has first-order derivative evidence; integer topology remains discrete. Higher derivatives are not claimed. |
| AN-11: device/repeat identity | `test_animation_cuda_forward_and_gradients_are_byte_identical` and special-angle/caller-stream tests (GPU); five native hardware fixtures (Native); `test_animation_completion.py::test_animation_repeated_backward_is_byte_stable` and CUDA counterpart (Completion) | Identical transferred operands/cotangents compare raw CPU/CUDA bytes; three fresh backwards per operation/dtype repeat exactly. Relaxed flags are exercised by native Scale profiles and may reuse the same ordered path. |
| AN-12: workspace/inference surface | `test_animation_stream.py::test_animation_out_reuses_output_and_matches_functional` (CPU); all-five caller-stream reuse (GPU); all-five alias/scratch/tracked-input/tracked-output rejection (Completion); native FK underallocation/misalignment (Native) | Every inference variant is reachable and reuses output storage; nonzero workspace requirements are enforced. No callback-safety claim. |
| AN-13: complete rig | `test_animation_complete_chain_gradients` (Strict); `test_animation_completion.py::test_animation_nonidentity_inverse_bind_chain_all_inputs` and CUDA counterpart (Completion) | Explicit nonidentity inverse bind and one backward reach eight continuous parameter groups, compared against independent tensor equations; no consumer asset/engine involved. |
| AN-14: target recovery | `test_animation_recovery.py::test_animation_recovers_independent_synthetic_target` (CPU; earlier separate `build/animation-20260909/recovery1.json`) | Different initial state, independently generated target, eight bounded iterations and recorded loss/parameter error verified. |
| AN-15: scale costs | Scale: eight native forward device/dtype/mode profiles plus four public coefficient/pose-gradient chains | Admitted profiles measured under unchanged limits; all-input CUDA omissions remain explicit below. |

The source call-path audit confirms shared tensor checks are used by all five
families (`an_view` and gradient/output helpers in `native/core/animation.c`,
CUDA descriptor checks in `animation_cuda_dispatch.inc`, and the common public
`_animation_stream._out` guard). This supports representative validator fixtures
without pretending each malformed field was independently tested at every entry.

**Closure recommendation:** accept the local animation Phase 1 reference,
Phase 2 device and Phase 3 first-order gradient baselines. No concrete outstanding
adopted fixture was identified after the Import, Strict and Completion cohorts.
PLAN.md remains the status authority. The original numerical recommendation
alone did not close AN-16; the separate presentation closure below now does. It also does not qualify new hosts/compilers, export, higher
derivatives, all-input scale gradients or real-time use.

Backward requests allocate only needed gradient outputs. This matters when the
corrective basis is large: a `[400,5023,3]` float64 basis occupies 48,220,800 bytes;
keeping its equally large gradient exceeds the current 64 MiB GPU worker limit
before other buffers. All-input float64 scale measurements are therefore refused,
not silently reduced or run under a larger cap. Public Python finite-value checks
also need temporary storage, unlike a raw native timing. The performance tool
separates those paths and records omitted profiles explicitly.
Animation finite-value validation scans dense inputs in chunks of at most 65,536
elements on their existing device. This bounds temporary masks independently of
basis size; it does not remove validation or imply asynchronous/callback-safe
execution. Whole-worker and allocation peaks are measured separately below.
An empty output still has a real backward: LBS/barycentric shared gradients are
initialized to zero, even when an empty cotangent's data pointer is null. Native
sentinel fixtures and both-device empty-axis tests cover this boundary.

`tools/animation_cost.py` uses one warmup and five measured samples. Native mode
times blend shapes and skinning through the C ABI; `--public-fitting` times all
five public operations plus coefficient/pose backward. Fixed template, basis and
skin weights receive no gradients in that fitting profile. Tiny harness checks
are not scale or real-time claims.

### Bounded synthetic scale measurements

All twelve declared profiles passed on the exact artifact above, using synthetic
B=1,V=5023,K=400,J=5 inputs, one warmup and five measured samples per operation or
chain. Eight native profiles cover CPU/CUDA, both precisions and reference/relaxed
flags. Four public profiles cover CPU/CUDA and both precisions in default reference
mode, learning coefficients and pose only. This is local cost evidence, not a
real-time guarantee, cross-device benchmark or actual consumer integration.

| Full public chain | Median milliseconds, five samples | Peak allocated GPU bytes |
|---|---:|---:|
| CPU float32 | 58.6372 | Not applicable |
| CPU float64 | 62.2164 | Not applicable |
| CUDA float32 | 25.9764 | 24,737,280 |
| CUDA float64 | 27.3522 | 49,285,632 |

The public timings include validation, allocations, all five operations,
coefficient/pose backward and completed results. Native forward timings exclude
the PyTorch adapter and must not be substituted for these numbers. Across all
twelve workers, the maximum sampled host working set was 865,308,672 bytes and
maximum allocated GPU memory was 49,348,096 bytes, within the unchanged CPU 1 GiB,
CUDA-host 2 GiB and GPU 64 MiB limits. Total supervised worker time was 49.9940934 s;
this includes import/setup rather than measuring only the operation.

Reproducible local reports are listed in
`build/animation-20260909/scale-schedule1.json`, which records the prospective
schedule, fixture/driver identity and limits before execution. Individual
create-only `cost-*.json` reports retain every sample, exact installed-wheel
identity and observed peaks. All-input float64 CUDA remains refused by its
storage lower bound; all-input float32 CUDA is not qualified by the selected
gradient measurements.

## Synthetic recovery and presentation

`tests/pytorch/test_animation_recovery.py` starts with different parameters from
its target and recovers two shape coefficients plus one joint angle. The target
is constructed with independent scalar rotation and explicit pivot arithmetic;
the fitted path uses all five public operations and nonidentity inverse bind.
This is a controlled identifiable inverse problem, not a licensed model loader,
FLAME validation or a guarantee that arbitrary fitting problems converge.

The finite Kinetic Bloom showcase uses the same public deformation operations
on first-party geometry. Authored motion, normal generation and display lighting
remain distinct from numerical acceptance. See [the source fixture entry](ANIMATION_CORPUS.md).

## Finite presentation closure, 14 September 2026

The internal PC dashboard now serves three visually inspected1280x720 Kinetic
Bloom poses from the preserved finite24-frame cache. Its explanation includes
1988 vertices,2684 triangles and maximum native/reference position error
7.152557373046875e-7 from the pinned historical producer report. These remain
laptop demonstration measurements; the separately rebuilt PC animation/stream
qualification is device evidence. Authored motion and display lighting are labelled.

The private HTTPS readback verifies every PNG hash/length, gallery/explanation
binding and successful execution (pc-saved-animation-gallery2). AN-16 is complete
for this finite presentation with the existing authored test pages and animation
lesson. The earlier404/catalog and saved-log-cap failures remain in history.
No rerender, continuous producer, history deletion or quota increase was needed.
No claim of live24-frame playback or broader visual/physical fidelity is made.
