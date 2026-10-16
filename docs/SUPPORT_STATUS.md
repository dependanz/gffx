# Current release support status

A candidate is admitted only after the complete release gates pass.

The Windows0.2.0 support contract requires exact-artifact
validation, the14-day RC/seven-clean-day gates, functional release CI and final
publication approval. [Release evidence](RELEASE_0_2_0.md) records actual outcomes;
[the adopted contract](RELEASE_POLICY_0_2_0.md) defines the supported/experimental boundary.

| Surface | Target and present boundary |
|---|---|
| Base CPU import/capabilities | Windowsx64/ordinary-GIL Python3.10-3.14; no framework/GPU probe |
| Eager CPU geometry/camera/animation/image | Exact policy operation list; Torch2.10.0+cpu across five Python minors; admission requires the recorded exact-artifact matrix |
| Optional CUDA | Existing provider, finite experimental Python3.14.6/Torch2.10cu126/RTX3080/driver610.88 profile |
| Scene/material/deferred/filtered/physical-reference and unlisted APIs | Experimental/unqualified under existing feature records |
| Compile/export/higher derivatives/streaming/performance/real-time | No stable0.2 admission |
| Other platforms/Python/framework/device combinations | Separate targets; no admission by ABI inference |

Published0.2.0a1/a2 and their exact evidence remain available. Historical foundation
and M1 inventory is preserved in [alpha support history](ALPHA_SUPPORT_HISTORY.md);
those old rows are not a current release qualification. M1 remains complete for
development readiness. The original reference campaign/heartbeat remain paused indefinitely.
