# Proposed offline certification budget for Group 3

Status: user approved the policy in D515 (Q31 resolved). Implementation and qualification are authorized. This document does not authorize a full trial.

## What the review established

Opaque166 stopped at4096region evaluations after3526reference seconds. Its complete interval was2.927times the allowed width. Correction uncertainty is distributed across2048leaves;501account forhalf. The rendering result has not failed an independent accuracy comparison: its reference remains insufficiently precise.

Opaque167 compared16deterministic ranks spanning the saved width distribution, retaining all physical contributions. Existing alternative proof methods offer no improvement on15of16sampled leaves. Crossing terms dominate most sampled local-proof regions. Existing one-step root refinement reduced12local-proof widths by17–87percent with little extra cost. A more aggressive contracted-root prototype adds limited tightening and costs1.4–2.1times as much on those affected regions. It is not adopted. These samples neither certify the whole piece nor guarantee success with a larger budget.

## Exact proposed policy

For a separately named Group3 offline reference profile only:

- At most8192charged region evaluations per piece, replacing4096.
- Keep depth14,1GiBhost memory,one numerical thread and strictly serial workers.
- At most7200reference seconds,7230worker seconds,7260watchdog seconds per piece, replacing3600/3630/3660.
- At most22fresh piece workers and one aggregation worker;stop first failure,no retry or certificate reuse.
- Maximum aggregate180224region evaluations,158400reference seconds(44hours),159180worker seconds and159900watchdog seconds including120/180seconds for aggregation. These are absolute ceilings,not predicted duration or a success guarantee.
- Keep every exact input/area allocation,all positive contributions,basis radius5e-13,final RGB/albedo-Jacobian radius1e-12 and installed absolute1e-10plusrelative1e-9 comparison.
- Production rendering is unchanged:32candidate faces,16cuts,800sample slots/32blocks per batch,16batches/12800slots/512blocks per pixel,64MiBactive storage,1GiBCPU/2GiBCUDAhost,one thread,120worker/180watchdog seconds.
- Dashboard quota/history,coverage-remeasurement deferral and consumer boundaries are unchanged. No Git publication.

## Execution gates after policy approval

Implement the explicit opt-in profile with tests that bind every limit and preserve old profiles. Qualify the chosen checker on independent analytical fixtures and representative complete-region comparisons,including setup cost. Freeze its exact files,input,wheel,controller and output directory. Present that concrete full-trial preview for the separately required approval;do not execute merely because this policy was approved. Failure remains terminal and all reports must be kept.

The alternative is to retain4096regions/one hour and continue redesigning the checker before a trial. There is no reliable ETA for either option. The recommendation is the bounded offline extension because enforcing the same small development-checking budget has become a substantial obstacle independent of the library's rendering requirements.

Evidence: opaque166/campaign-analysis1.json and remaining-distribution1.json; opaque167/survey-summary1.json,root-survey1.json,contracted-survey1.json and design-review1.json beneath D:/devspace/gffx/qualification-20260913/m1-continuation-20260913. The survey1 Windows file-sharing interruption is retained; only the missing comparisons were continued.

## Approved campaign outcome

D515 approved the offline profile; D516 approved the exact opaque168 frozen trial. All22 fresh piece certificates and final aggregation passed. The selected original128 pixel(59,10) is independently qualified for fixed packed geometry, colour and the constant-albedo first Jacobian against preserved CPU/CUDA results. This is not full-frame or new moving-geometry admission.

Campaign controller10952.4180956seconds;15216charged regions;10773.2337283reference seconds;643297280peak host bytes. The previously blocked piece3passed6731regions/5405.6521327reference seconds. All six comparisons passed unchanged absolute1e-10plusrelative1e-9 allowance. No memory, depth, rendering or accuracy relaxation; no history deletion or Git publication.

Evidence: D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque168-20261002/campaign-analysis1.json and campaign1/aggregate.json. The campaign is terminal and must not be rerun or resumed. Remaining representative pixels and frame/scale gates are governed by PLAN.md.
