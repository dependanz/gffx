# gffx 0.2.0 Windows release policy

Adopted 16 October 2026 for the first stable Windows x64 release. Admission is
pending the exact candidate's validation, release-candidate soak and publication.
The version number alone never grants a support claim.

## Scope

This deliberately replaces the initial broad platform/framework admission target
for 0.2.0 only. The older four-platform/PyTorch 2.10-2.13 and CUDA 12.0 build-floor
targets remain future work. No previously admitted support tier is withdrawn.
Native ABI, operation semantics, numerical assertions, error behavior, history,
security obligations and compatibility/deprecation principles remain in force.
M1 development readiness remains complete; release readiness is a separate result.

The first distribution is one Windows x64 `cp310-abi3` wheel, ordinary-GIL CPython
3.10-3.14, with no mandatory Python dependencies. The base C11 core and CPython
bridge do not load PyTorch or probe a GPU. The optional PyTorch adapter is built
against CPU-only Torch 2.10.0 and the existing Stable ABI; PyTorch is selected
separately. No framework, CUDA Runtime or NVIDIA driver library is bundled.

| Surface | 0.2.0 admission target |
|---|---|
| Base import, version, static CPU capabilities | Windows x64, CPython 3.10-3.14; no Torch or GPU required |
| Functional eager CPU operations below | Torch 2.10.0+cpu on each Python minor 3.10-3.14 |
| CPU use with CUDA-enabled Torch | Exact Python 3.14.6/Torch 2.10.0+cu126 profile |
| Optional CUDA operations | Experimental exact profile below; no broad GPU/driver matrix promise |
| Other Python/framework/platform combinations | Unqualified or experimental; wheel ABI compatibility is not validation |
| Compilation, export, higher derivatives, streaming `_out`, performance/real-time | Outside this stable admission |

The stable eager CPU operation list is:

- Geometry: `mesh.face_geometry`, `mesh.vertex_normals`, `mesh.gather_faces`,
  `mesh.build_edge_topology`, `mesh.sample_surface`, `transforms.transform_points`,
  `transforms.perspective_divide`, `points.knn`, `points.closest_point_on_mesh`,
  `render.rasterize`, `render.interpolate`, `render.texture_pyramid`, `render.texture`.
- Animation: `mesh.blend_shapes`, `mesh.skin_lbs`, `mesh.evaluate_barycentric`,
  `transforms.rodrigues`, `skeleton.forward_kinematics`.
- Camera: eager pinhole and orthographic projection/unprojection, batching and
  documented front filtering through `gffx.camera`.
- Image: `convert_alpha`, `srgb_to_linear`, `linear_to_srgb`, `apply_mask`,
  `composite`, `reconstruct`, `affine_matrix`, `warp_affine`, `adjust`,
  `gaussian_blur` through `gffx.image`/`gffx.torch.image`.

Only documented first derivatives where defined are included. Integer topology,
selected hard visibility, nearest-neighbor choices and branch singularities keep
their documented nondifferentiable or local-gradient behavior. No all-input proof
follows from finite tests. Other public APIs remain available under their existing
experimental boundaries, including scene/material/deferred/filtered rendering,
native physical-reference work, PLY/file surfaces and unlisted streaming paths.
No API is removed to make the table pass.

## CUDA profile

The existing optional provider is built with CUDA/nvcc 12.6.20, PTX ISA 8.5,
target sm_75, VS2022/MSVC 14.44. This is an explicit initial Windows release
exception to the future CUDA 12.0 compiler-floor target. The declared driver
baseline is Windows NVIDIA 560.76+, also supporting the user's GPU; actual
validation is RTX 3080, compute 8.6, driver 610.88, Python 3.14.6 and Torch
2.10.0+cu126. An old-driver or cross-GPU claim is not inferred. Other CUDA-enabled
Torch distributions cannot acquire a tested claim merely by loading the adapter.

GPU checks use actual CUDA tensors, independent numerical gradients, byte-exact
matched CPU/GPU forward/VJP controls, caller streams, storage and refusals. A
missing GPU or skipped GPU case fails the selected GPU cohort. Base and CPU use
must remain independent of the driver/provider. Users need no toolkit/compiler.

The filtered reference campaign and heartbeat remain paused indefinitely. No
producer/auditor, photograph fit, model migration, large physical qualification or
performance campaign is part of this release. Retained failures stay visible.

## Evidence and CI

The exact final wheel is built once, repaired and audited for native imports,
provider ABI/PTX, metadata/tags and every RECORD entry. Each advertised CPU minor
installs that wheel in a fresh environment, runs the declared operation/gradient,
boundary/error and camera/image checks, and verifies payload identity. Test
selection is explicit; no project-wide wildcard can launch reference work.
Upgrade from the old public line and alphas, uninstall, missing-framework behavior
and lazy base loading are checked independently. Consumer tests remain evidence
of integration rather than automatic support for every renderer they exercise.

Pull-request CPU functional CI builds one CPU adapter wheel and tests all five
minors. This catches functional regressions but is distinct from qualification of
the release wheel that includes the optional provider. The final publishing
workflow must test the attached exact release artifact on the same CPU matrix
before upload. GPU validation is local or trusted manual hardware only; untrusted
PRs never execute on a persistent GPU host. Unexecuted hosted definitions are
reported as prepared, never as passing runs.

## Release-candidate soak

Retain 14 calendar days of release-candidate evaluation and at least seven clean
days after the last release-blocking code, dependency, workflow or artifact change.
Start only when the candidate and release-critical content are frozen and all
initial required local validation passes. Run the same exact candidate again at
the midpoint (at least day seven) and before final publication (at least day14).
Every required profile must pass each checkpoint. Record date, artifact/source
hashes, test selection, actual outcomes and any blocking issue. Elapsed time by
itself is not a passing soak. A blocking change resets the appropriate clock;
record-only additions of new evidence do not alter the evaluated implementation.

The unpublished candidate may already carry final `0.2.0` metadata so the exact
bytes can be evaluated throughout the soak. No public upload or automatic
scheduler follows preparation. A verifier must refuse early or incomplete
publication; no date or test result is fabricated to bypass this rule.

## Security and publication

Publish SECURITY.md and enable private vulnerability reporting before stable
admission. Existing general response/deprecation/maintenance policies apply.
The settings operation is prepared for the final approved transaction, not
silently performed during local preparation. Triage verified release-affecting
security defects; old default-branch alerts do not automatically describe this
artifact.

After all gates, obtain separate approval of the exact source/identity/message,
existing branch/remote, v0.2.0 tag and normal GitHub release, retained publishing
policies plus the one new tag policy, and exact-wheel PyPI upload. Reuse the proven
Trusted Publisher with pinned actions and OIDC. Never rebuild in the publisher,
replace a published artifact, move a tag or treat workflow dispatch as success.
Verify the actual PyPI download. Human-readable and machine-readable support
records must agree with the exact results. The wider roadmap is not a release gate.
