# Original-frame offline qualification proposal

Status: approved by the user on2026-10-03; D569 resolvesQ34.
Implementation and controls must pass before a frozen campaign is admitted.
The frozen opaque223 campaign has completed all four larger-frame profiles.
Group3 is qualified in scope; M1 remains unfinished.

## Verified checkpoint and the remaining distinction

Opaque222 installed wheel58d51c898e6f completes original forward plus constant-RGB-
albedo backward at128 and256 on both CPU/CUDA, retaining complete image bytes
and exact cross-device gradient bytes. CPU12830.2232468s;CUDA12856.4415175s;
CPU25672.938657s;CUDA256116.9730726s. These are whole-worker measured durations.
The separate historical hard/soft scale matrix is24/24complete.

D504 admits larger original frames only from two matching complete smaller sizes,
using the slower pixel rate,twice that rate,and30seconds reserve. Hash-checked
opaque222 anchors produce these forecasts; they are NOT measured larger renders:

| Profile |512x512 forecast|1920x1080 forecast|
|---|---:|---:|
|CPU forward+backward|997.1438976s|7680.2593462s|
|CUDA forward+backward|1836.1285600s|14316.7591172s|

Every profile is refused under the120second whole-frame rule. The old standalone
forward qualifications retain their own source/wheel identity and cannot be
silently reused as matching anchors for this candidate. Complete forward pixels
are already checked as part of the four passing backward profiles above.

## Choice A: retain current whole-frame admission

Continue performance engineering and qualify each repair under existing rules.
No new authority is needed for that work. Larger original rows stay unfinished
until they satisfy D504 and the120second complete-frame budget. There is no
reliable completion estimate; the smaller-frame rates remain far from admission.

## Choice B: add a separately labelled offline qualification profile

Authorize implementation, test-first qualification and a frozen serial campaign
for original512x512 and1920x1080 on CPU/CUDA. Each full frame consists of bounded,
non-overlapping row ranges, with at most32768pixels per rendering worker.
Each range performs both forward storage and matching albedo-backward replay.
A final bounded aggregation worker verifies completeness and publishes a final
manifest only after every range is verified.

|One device/frame|Row ranges|Range workers|Aggregation workers|Worker ceiling|Watchdog ceiling|
|---|---|---:|---:|---:|---:|
|512x512|8ranges of64rows|8|1|1080s (18min)|1620s (27min)|
|1920x1080|63ranges of17rows plus9rows|64|1|7800s (130min)|11700s (195min)|

All four profiles together have at most148workers,17760worker seconds
(4h56min) and26640watchdog seconds (7h24min). These are ceilings,not ETAs or
success guarantees. Workers remain serial; stop the campaign at the first
numerical,resource,receipt or integrity failure. Completed ranges are reusable
only after exact identity/hash/prefix verification. An exhausted range is not
restarted unchanged or given a fresh timer to conceal failure.

Every worker retains120seconds/180watchdog,one CPU thread,CPU1GiB orCUDA2GiB
host memory and64MiB active storage. All32candidate-face,16cut,32block/800slot
per-batch and16batch/512block/12800slot per-pixel limits remain. No numerical
acceptance, tiny contribution, full material derivative check or old failure is
removed. Total internal campaign storage is capped at1GiB; dashboard quotas and
history remain untouched. Require sufficient free disk before any launch.

The changed rule is the TOTAL offline full-frame allowance and its admission
procedure. This is not a claim that a full original512/1080frame meets120seconds.
The existing single-worker refusals remain valid and visible. If adopted, Group3
may count this explicitly named offline storage/recompute profile for those
larger original rows; it must continue reporting actual total frame duration,
including setup,all workers,transfers,checks and aggregation. It does not claim
interactive or real-time performance and does not implement either consumer.

## Required implementation and controls before execution

1. Keep the full global camera and exact scene/wheel/source identity. Ranges select
   global pixels; never substitute a cropped or rescaled camera. Preserve window
   coordinates,coverage and each pixel's integration choices.
2. Store immutable range manifests with global first pixel/count,output hashes,
   cotangent rule and input/output gradient-carry BYTES. The cotangent denominator
   uses the full frame. Pass the previous RGBcarry into the next range unchanged;
   never add a range subtotal. Validate its predecessor hash and exact prefix.
3. Preserve original forward-versus-recomputed pixel equality within each range.
   Qualify range boundaries,short final ranges,signed zero/subnormals and varying
   cotangents against the existing canonical complete small-frame implementation.
   Qualify CPU/CUDA separately,then compare final complete images and gradients.
4. Reject missing/duplicate/reordered ranges,mutated inputs or artifacts,wrong
   origins/cameras/cotangents,changed carries,cancellation,corrupt payloads and
   resource exhaustion. Interrupted or failed campaigns never acquire complete
   manifests or become timing anchors. No synthetic success or deleted history.
5. Before launch freeze the actual qualified executor,candidate wheel,source and
   evidence hashes plus exact range plan. Record every attempt and actual cost.
   Never use these multi-worker reports as D504 single-worker timing anchors.

Evidence: opaque222/frame-preview1.json,frame-preview2.json and the four complete
*-backward1.json reports;larger-admission1.json;offline-plan-proposal1.json.
No commit,push,new branch,new dependency,dashboard quota increase or external
service change is included. D518 excludes quota increases; ChoiceB therefore
requires an explicit new decision despite the standing continuation approval.

## Terminal qualification — 3 October 2026

D569 adopted Choice B. Opaque223 qualified25 controls on each device, then all148
serial campaign workers passed. Complete512 and1080 image/mask/gradient bytes
match CPU/CUDA. Summed supervised times are255.5136643/397.5760160seconds for512
CPU/CUDA and1209.5038635/1763.0971096seconds for1080 CPU/CUDA. Whole campaign wall
time is3669.2364028seconds. Existing resource and accuracy limits remain intact.
See M1_READINESS.md and opaque223/handoff-audit1.json for exact scope and receipts.
This completed trial must not be rerun just to refresh status. Frozen source
copies retain the pre-execution document and code; later prose updates are not
changes to the trial. Groups4/5 remain before M1 closeout.
