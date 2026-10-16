# gffx 0.2.0 Windows release policy

Adopted 16 October 2026 for the first stable Windows x64 release. Admission is
pending the exact candidate's validation, explicit owner release sign-off and publication.
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

## Explicit release sign-off for 0.2.0

On 16 October 2026 the maintainer explicitly replaced the initial 0.2.0
fourteen-day/seven-clean-day calendar hold with release sign-off after concrete
checks. This is a release-specific exception to the older general policy. The
old policy, clock and receipts remain historical evidence; no elapsed or passed
soak is claimed. No fixed calendar wait or midpoint checkpoint is required for
0.2.0. Numerical assertions, runtime scope and security obligations are unchanged.

Technical readiness requires the same reviewed final wheel, complete initial
evidence and a fresh seven-profile exact-wheel validation within 24 hours of
publication, unchanged source/test selections and no open release blockers.
Every selected functional profile must pass without failures or skips. Retain
verified native, install/upgrade/uninstall, strict metadata/import/provider/PTX/
RECORD and hosted functional CPU evidence, with explicit artifact/source identity.
Policy/controller changes alone do not require rebuilding unchanged runtime bytes.
A runtime/artifact change requires affected checks and a new reviewed candidate.

`tools/stable_release.py --check` verifies concrete readiness and explicitly
reports that owner publication sign-off is still required. It neither records
approval nor publishes. Byte/source-only `--candidate` is a diagnostic mode.
Neither mode is allowed in the publisher.

The user must separately approve the exact final source, identities, message,
destinations, settings, tag, normal release, wheel and PyPI transaction. Executing
that approved transaction publishes the normal v0.2.0 release through the
maintainer's existing dependanz account. The GitHub Actions publisher requires
the actual published-release event from dependanz/gffx: sender and author dependanz,
v0.2.0, not draft/prerelease, exact reviewed commit and exact reviewed wheel asset.
It checks real timestamps, source/evidence identities and freshness, then tests
the attached final wheel on all five CPU minors before the existing OIDC upload.
This event is the recorded owner release sign-off; it is not a substitute for
obtaining the user's approval before operating the account. Preparation and unit
test fixtures never count as actual publication approval.

No automatic publication, model fit, reference campaign or heartbeat restart
follows technical readiness. Retain all failures and report partial publication
honestly. Published files and tags are never replaced to repair a failed release.

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
