# Proposed offline certification of one streamed original-mesh pixel

Status: approved by the user on2026-10-01; recorded as devbrain decisionD505. Implementation controls must pass before the single frozen campaign launches.

## Problem and evidence

Group 3 production streaming executes the selected original pixels and CPU/CUDA agree exactly on their saved colour, mask and albedo-gradient bytes. Independent accuracy is a separate gate. The first new pixel (59,10) in the 128x128 original frame has22 exact visible integration pieces. The priorD503 profile shares256 charged cells and90 reference seconds across the entire pixel.

Opaque160 introduced optional localized smooth bounds, a conditional near-horizon base with the complete nonlinear correction, and exact cosine slope bounds. All historical defaults and the installed wheel remain unchanged. The last complete enclosure of trial3 has specular width7.0400216719083701e-16 and matte width3.0856023420287383e-8, about370times narrower than the opaque158 checkpoint, but above the required1e-12 basis width. It stopped at the reference deadline, with234 charged cells and a last complete232-cell/127-leaf cover. This is proof uncertainty, not a measured rendering error. Fourth-order saved-region corrections improve by roughly2–6times while costing substantially more; they are not a qualified whole-pixel replacement.

## Exact proposed scope

Authorize implementation and qualification of a NEW explicit offline checker profile for this one frozen pixel only. Do not broaden or rename the historical original13 profile. Do not change runtime rendering configuration or production frame admission.

Input: D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque158-20261001/pixel0-input2.json
SHA-256: 058e854ed69ba529c1e42f20b213cd9f9e693a043a71a24a25ad8897e7c9326c
Installed wheel SHA-256: 464050e528b5699d1788b7dcc520fcf948cea742a7a5f40391f80fc3ddc35a7b

- At most22 fresh piece workers, strictly serial, stopping at the first failure.
- Each retains256 charged cells, depth10,90 reference seconds,120 whole-worker seconds,180 watchdog seconds,1GiB host memory and one thread.
- One final bounded CPU aggregation/comparison worker, only after all22 pieces pass:1GiB,one thread,120 worker seconds,180 watchdog seconds.
- Maximum aggregate piece work:5632 charged cells and1980 reference seconds (33minutes). Including final aggregation, maximum whole-worker allowance is2760seconds (46minutes) and sum of watchdog ceilings4140seconds (69minutes). These are hard maxima, not an ETA or success guarantee.
- Conserved error allocation: use exact visible-area weights via the existing piece_tolerances function, rounding downward, with total specular and matte radius no greater than5e-13 each. No piece is granted the whole pixel tolerance. Aggregate RGB and per-channel albedo Jacobian radius remains at most1e-12.
- Bind every certificate to the frozen input, piece index/geometry, exact tolerance allocation, checker/driver/dependency hashes, arithmetic profile and worker result. Never merge reports from different candidates or from the laptop. Freeze the implementation after its controls pass, before launching a campaign.
- Preserve every nonzero piece, signed correction and failed report. Zero pieces need an exact geometric/sign certificate. No partial pixel acceptance.
- Independently validate the aggregate against the preserved installed CPU values and hash-verified CUDA byte agreement using the existing absolute/relative acceptance. No changed comparison threshold.
- Failures end this bounded campaign; no automatic second campaign, increased cap, altered input or allocation, or unchecked certificate reuse.

## What remains unchanged

All D502 rendering limits, including64MiB active storage,1GiB CPU/2GiB CUDA host,120s rendering worker,180s watchdog,one thread,32candidate faces,16cuts,16batches and12800slots per pixel, remain unchanged. No dashboard quota increase or history deletion. No full frame launches before independent accuracy/storage admission. The approved128/256 startup sizes remain approved. No new Git publication or branch creation.

## Required implementation controls before a campaign

Test the explicit profile and exact input identity; reject selector/input/candidate mismatches and historical-profile mixing. Prove downward allocation and complete22-piece aggregation, including missing/duplicate/reordered pieces, malformed bounds, tiny nonzero contributions, over-radius totals and failed workers. Verify the same installed wheel and existing supervisor enforcement. Preserve historical13-piece selectors and standard shared-pool behavior. Preview the frozen campaign files/hashes after these controls pass; this approval authorizes that single campaign within the exact policy above, not unrelated experiments or larger workloads.

## Alternative

Retain the existing whole-pixel256-cell/90s cap and continue mathematical/checker optimization. Current results do not establish that sufficient accuracy can be reached under that cap; they also do not prove it impossible.

## Outcome of the authorized campaign

The one campaign is now terminal; its authorization has been used. In opaque161-20261001/campaign1, piece index0 passed and piece index1 hit its90s reference limit. Exactly1of22 pieces is certified; the remaining20pieces and final aggregation did not run. There is no complete-pixel or frame admission. The failure analysis isolates shading-boundary correction uncertainty, not a measured renderer defect. Further saved-region diagnosis can prepare a repair, but another frozen numerical campaign requires a new concrete preview and approval. Preserve all reports and do not reuse this partial certificate with a changed candidate.

## Approved extended offline campaign — 2026-10-01

The user approved a new, separate profile in D506 after the original D505 campaign stopped. For this same frozen22-piece pixel, allow4096cells,depth14,900reference seconds,930whole-worker seconds,960watchdog seconds,1GiB and one thread per piece. Execution order is piece index1first,then0and2through21; every certificate is fresh. Stop at first failure. Only complete22-piece success permits the final aggregation worker, which retains120worker seconds/180watchdog seconds. Total ceilings90112cells,19800reference seconds,20580worker seconds and21300summed watchdog seconds. These are ceilings,not an estimate.

The exact original input, downward area shares,5e-13total basis radius,1e-12RGB/Jacobian radius and installed abs1e-10/rel1e-9 comparison stay unchanged. All production rendering limits and standard/historical checking profiles stay unchanged.65new and regression controls passed before freezing. This authorizes one new frozen campaign; no additional automatic campaign or further increase. Current evidence and frozen candidate live in opaque162-20261001.

### Extended campaign outcome

The approved D506 campaign is terminal at900reference seconds on its first scheduled piece(index1). It narrowed that piece's bound about1804times but remained3.37times above the unchanged assigned allowance.21later pieces and aggregation were not run. Full outcome: opaque162-20261001/failure-analysis1.json. A further time-only proposal is separately pending in M1_STREAMED_PIXEL_REFERENCE_HOUR_PROPOSAL.md; this record does not authorize it.
