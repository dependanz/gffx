# gffx 0.2.0 release evidence

Initial candidate checkpoint: 2026-10-16T09:30:07.306593+00:00. This is an unpublished Windows x64
candidate. The initial local matrix passes; the retained soak and hosted CI are
not yet complete. Publication requires approval after all gates pass.

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

## Outstanding admission gates

The real soak started at 2026-10-16T09:30:07.306593+00:00; midpoint checks are due no earlier than
2026-10-23T09:30:07.306593+00:00, and stable eligibility no earlier than
2026-10-30T09:30:07.306593+00:00. Seven clean days after the last
blocking change are also required. Run the same seven exact-artifact profiles at
each checkpoint and record actual outcomes. Age alone cannot pass admission.

`releases/0.2.0.json` seals the wheel, implementation/test/controller sources and
actual initial results. `python tools/stable_release.py --dist <wheel-folder>`
must refuse the current early/incomplete state. `--candidate` checks bytes/source
only and is never allowed in the publisher. Synthetic unit-test dates are not
soak evidence. Blocking code/dependency/workflow/artifact changes reset the
appropriate clock; factual checkpoint additions alone do not change the code.

The new `windows-functional` workflow builds one CPU adapter wheel and tests it
on all five minors. It is prepared and statically reviewed, not a hosted passing
run. Daily checks require its reviewed integration into default branch `main`.
The stable publisher independently tests the attached exact final wheel on all
five CPU minors before OIDC upload. It also requires private vulnerability
reporting to be enabled; that setting is prepared but has not been changed.

Published a1/a2 tags and artifacts are preserved. The original filtered campaign
and heartbeat remain paused indefinitely. No photograph fitting, physical
reference campaign, shared framework change or automatic publication is started.
