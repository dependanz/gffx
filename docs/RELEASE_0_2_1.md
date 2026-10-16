# gffx 0.2.1 technical release evidence

This records actual artifact validation; public availability is independently
verified by the approved publication transaction. See [PyPI](https://pypi.org/project/gffx/0.2.1/)
and the [normal release](https://github.com/dependanz/gffx/releases/tag/v0.2.1).

## Exact Windows artifact

`gffx-0.2.1-cp310-abi3-win_amd64.whl` — 3,194,417 bytes, SHA256 `05d7b0fb2c713f1b42f1fb6a33b829ec2842f5dcc1274e4b709d7055f9ba7a9d`.
One wheel includes the CPU core, Stable-ABI Torch adapter, lazy CUDA provider and
22-line runnable example. Direct requirement: `torch>=2.10.0,<=2.14.1`.
The current default is2.14.1; existing compatible2.10CUDA stays installed.

## Actual exact-wheel matrix

| Profile | Python | Torch | Operations pass / fail / skip | Scene checks pass / fail / skip |
|---|---|---|---|---|
| cpu310 | 3.10.11 | 2.10.0+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| cpu311 | 3.11.8 | 2.10.0+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| cpu312 | 3.12.1 | 2.10.0+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| cpu313 | 3.13.16 | 2.10.0+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| cpu314 | 3.14.6 | 2.10.0+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| cuda-torch-cpu314 | 3.14.6 | 2.10.0+cu126 | 323 / 0 / 0 | 14 / 0 / 0 |
| gpu314 | 3.14.6 | 2.10.0+cu126 | 285 / 0 / 0 | 17 / 0 / 0 |
| current-cpu310 | 3.10.11 | 2.14.1+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| current-cpu311 | 3.11.8 | 2.14.1+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| current-cpu312 | 3.12.1 | 2.14.1+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| current-cpu313 | 3.13.16 | 2.14.1+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| current-cpu314 | 3.14.6 | 2.14.1+cpu | 323 / 0 / 0 | 14 / 0 / 0 |
| current-cuda-torch-cpu314 | 3.14.6 | 2.14.1+cu126 | 323 / 0 / 0 | 14 / 0 / 0 |
| current-gpu314 | 3.14.6 | 2.14.1+cu126 | 285 / 0 / 0 | 17 / 0 / 0 |

All14profiles verified89installed payload members and rendered the included
512px example. Each CPU scene selection explicitly deselects CUDA nodes; no
selected test skips are accepted. Current/retained GPU execution is actual
RTX3080/CC8.6/driver610.88 withcu126. Final summary: `2026-10-16T13:14:55.519092+00:00`.
The current GPU example reserved299892736bytes; timings are instrumentation,
not a performance qualification. Visibility is discrete and lighting direct,
with explicit ambient fill; no multi-bounce GI claim.

Additional concrete controls:94release/metadata/native-ownership/base-import
tests;12native analytical tests on the exact core DLL; four native-only
install/upgrade/uninstall scenarios with30accepted commands; strictTwine,
PE/import/PTX/RECORD checks. A genuinely empty environment installed the local
exact wheel through ordinary dependency resolution from PyPI, selectedTorch2.14.1,
rendered the included CPU example and passedpipcheck. Both existing tested CUDA
builds were retained by normal dependency resolution. Officialcu126index
resolution is separately checked. Prior failed setups/control spellings remain
preserved privately and charged; they are not passing numerical evidence.

Publication additionally requires actual successful sourceCI at the reviewed
commit (builder plus10CPUminor/frameworkjobs), owner normalrelease sign-off,
10exact-release-asset CPU jobs and OIDC upload. No future hostedCI success is
predeclared here. [Policy](RELEASE_POLICY_0_2_1.md) separates usability/eagerCPU
scope from experimentalrenderer/CUDA and broader unqualified capabilities.
The campaign and heartbeat remain paused indefinitely.
