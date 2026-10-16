# gffx 0.2.0 release evidence

Initial candidate checkpoint: 2026-10-16T09:30:07.306593+00:00. This is an unpublished Windows x64
candidate. The initial local matrix passes; subsequent source/hosted CI results
and the explicitly amended admission policy are recorded below. Publication
requires final user approval after the concrete checks pass.

## Exact artifact and scope

`gffx-0.2.0-cp310-abi3-win_amd64.whl`: 3,185,318 bytes, SHA256
`0385d27a028b04b98500766be11b81d95b7e3415436c681cd040329ccebe2ed4`.

The [release policy](RELEASE_POLICY_0_2_0.md) admits the declared eager CPU
geometry, animation, camera and image surface on ordinary-GIL Python 3.10-3.14
with separately installed Torch 2.10.0+cpu. Windows 11 x64 is the actual local
host; Windows Server 2022 is the prepared hosted CI representative. The optional
CUDA provider retains its exact experimental hardware/runtime profile. Wider
framework/platform, filtered renderer, performance and higher-order claims are
outside this release admission.

## Completed initial checks

| Profile | Actual Python | Actual Torch | Passed / failed / skipped |
|---|---|---|---|
| cpu310 | 3.10.11 | 2.10.0+cpu | 323 / 0 / 0 |
| cpu311 | 3.11.8 | 2.10.0+cpu | 323 / 0 / 0 |
| cpu312 | 3.12.1 | 2.10.0+cpu | 323 / 0 / 0 |
| cpu313 | 3.13.16 | 2.10.0+cpu | 323 / 0 / 0 |
| cpu314 | 3.14.6 | 2.10.0+cpu | 323 / 0 / 0 |
| cuda-torch-cpu314 | 3.14.6 | 2.10.0+cu126 | 323 / 0 / 0 |
| gpu314 | 3.14.6 | 2.10.0+cu126 | 285 / 0 / 0 |

Every profile installs the same artifact and verifies all 84 package payload
members. The explicit finite cohort includes operation assertions, independent
gradient checks, camera/image coverage and boundary/refusal controls. CPU cohorts
explicitly deselect CUDA-only nodes; the actual GPU cohort accepts no skips.
GPU execution used RTX 3080/compute 8.6/driver 610.88; the provider is built with
nvcc 12.6.20/PTX 8.5/sm_75. These finite checks do not prove all-input correctness.

Additional checks pass: 12 native analytical tests against the exact wheel's
C11 DLL; four installation/upgrade/uninstall scenarios with 30 successful
commands (fresh base, public 0.1.4, 0.2.0a1 and 0.2.0a2); strict Twine,
PE imports, provider/PTX identity and RECORD verification. The focused
foundation/release-control cohort passes 72 tests with one intentional
below-Torch-floor compatibility skip. The advertised runtime is Torch 2.10;
that skip is separate from the no-skip functional acceptance matrix.

The old historical camera fixture's NumPy/private-file failures and two rejected
GPU launcher runs are retained in the local evidence packet. A portable synthetic
exact-rational/byte-order camera fixture replaces the private dependency in public
CI. The GPU launcher now sets the existing camera fixture's device correctly;
no mathematical tolerance or runtime geometry/rendering code changed. Stale
packaging tests and the development pytest pin were repaired, not hidden.

## Revised 0.2.0 admission

On 16 October 2026 the maintainer replaced the initial 14-day/seven-clean-day
calendar hold with explicit final release sign-off after concrete checks. No
elapsed or completed soak is claimed. Earlier clock/validation evidence remains
preserved in project memory and the release manifest's history.

The approved source commit935816fb2e22711ce61085ae80e31b79a0c52568 is on both
main/m1-camera. Actual Windows functional CI38885078757 succeeded: build plus all
five CPU profiles,323 passing tests each with no failures/skips; six artifacts
were retained. That CI built one CPU test wheel; it is distinct from the final
provider-containing candidate identified above. The final publishing workflow
independently checks and tests that exact final wheel before upload.

Fresh final seven-profile validation completed at 2026-10-16T10:47:06.880944+00:00:
all five CPU-only profiles and CPU-with-CUDA-enabled-Torch pass323checks each;
actual GPU passes285; every profile has0failures/0skips and verifies84package
payloadmembers against the same final wheel. The source seal binds882files;
all187runtime/buildinputs remain unchanged. Fresh final evidence must remain
within24hours at publication. The revised release gate has44passing controls;
the combined focused cohort passes100tests with one separate below-Torch-floor
compatibility skip. The first cohort's obsolete14-day documentation assertion
failure is preserved and was corrected to the explicitly amended policy.
Strict Twine/nativeimports/provider/PTX/RECORD audit passes again. Existing
12native and4installation/30command checks remain valid for these unchanged bytes.

Technical readiness is complete under the revised policy. `--check` reports technical readiness without granting
publication approval. The publisher requires the real owner normal-release event
for the reviewed source and exact asset. No sign-off is fabricated in advance.

Private reporting/newtagpolicy/normalrelease/PyPIupload require final approval.
Published a1/a2 and all failures remain preserved. The reference campaign and
heartbeat stay paused indefinitely; no model/photograph or physical-reference
work is required for this release.
