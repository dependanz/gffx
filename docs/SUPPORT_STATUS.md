# Current release support status

A candidate is admitted only after its exact release gates and publication pass.
The [0.2.1 policy](RELEASE_POLICY_0_2_1.md) and [evidence](RELEASE_0_2_1.md)
define the current bounded Windows contract; older release evidence is preserved.
Scene rendering and CUDA remain experimental.

| Surface | Exact boundary |
|---|---|
| Core/import | Windows x64, ordinary-GIL Python3.10–3.14; independent C11 core and lazy import; no GPU probe |
| Eager CPU geometry/camera/animation/image | The retained operation list on currentTorch2.14.1+cpu for each Python minor; separate retained2.10.0+cpu compatibility evidence |
| CPU under CUDA Torch | ExactPython3.14.6/cu1262.14.1 andretained2.10 profiles |
| CUDA | Experimental exactRTX3080/CC8.6/driver610.88/Python3.14.6/cu1262.14.1 andretained2.10 |
| Scene/material/deferred/filtered/physical-reference | Experimental; short example/device/PNG/gradient evidence is finite, not physical GI or general all-input qualification |
| Compile/export/higher derivatives/streaming/performance/real-time | Not admitted |
| Other platforms/framework versions/devices | Unqualified; metadata compatibility or an ABI tag does not grant support |

The distribution automatically declares a bounded compatibleTorch requirement;
fresh installs select current2.14.1. StandardWindowsPyPITorch isCPU-only; the
officialcu126index supplies a one-command freshGPU install. Compatible existing
CUDA frameworks are retained. Intermediate permitted versions do not acquire
tested-support status merely because the dependency range accepts them.

Torch2.14.1 is current upstream at inspection. Older2.10 is upstream-unmaintained
and has [GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47).
That advisory affectsTorch<=2.12.1 and was firstfixedin2.13. GFFX has no TorchScript
call; compilation/export is excluded. Retained compatibility is distinct from
upstream security. Historical0.2.0/a1/a2 and their dependency-free installation
policy stay unchanged. M1 is complete for development readiness.
The reference campaign and heartbeat stay paused indefinitely.

The [alpha support history](ALPHA_SUPPORT_HISTORY.md) preserves the earlier
foundation/M1 inventory and artifact-specific limits.
