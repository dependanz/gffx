# Proposed time-only extension for the original22-piece offline campaign

Status: explicitly approved by the user on2026-10-01; recorded as D507, resolving Q27. Qualify and freeze the new profile before one fresh campaign.

The approved15-minute D506 campaign is terminal. It ran the previously failing piece first and stopped at900reference seconds. The last complete1437-cell/719-leaf enclosure has matte width9.839841759955241e-15 against its unchanged2.918184203864486e-15 allowance:3.371905times too wide,about1803.57times narrower than the prior90-second result. Terminal work1438cells; worker903.5510821s; monitored peak596353024bytes. No new piece passed;21others and aggregation did not run. These are proof bounds,not measured rendering errors.

## Exact proposed change

Implement and qualify a new explicit streamed22-offline-hour-v1 profile for the identical frozen input and unchanged downward error allocation. Change only per-piece time ceilings:3600reference seconds,3630whole-worker seconds and3660watchdog seconds. Keep4096charged cells,depth14,1GiB,one thread,serial execution,every nonzero contribution and all accuracy requirements. Existing standard,13-piece and15-minute profiles remain unchanged.

After controls pass, freeze the checker,driver,dependencies,input,allocation and installed-wheel hashes, then run one fresh campaign: piece index1first,then0and2..21. Each piece exits as soon as it proves its allocated bound. Stop the campaign at the first failure. Only all22passed certificates permit the existing120s/180s bounded aggregation/comparison worker. Do not reuse partial certificates from other candidates or launch an automatic second campaign.

Maximum totals:90112cells,79200reference seconds(22hours),79980whole-worker seconds(22hours13minutes) and80700summed watchdog seconds(22hours25minutes),including aggregation where applicable. These are worst-case ceilings,not an ETA. The observed1438cells in900seconds shows why a900-second deadline can stop well before the already-approved4096-cell cap. More time still does not guarantee acceptance.

Input: D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque158-20261001/pixel0-input2.json
Input SHA256:058e854ed69ba529c1e42f20b213cd9f9e693a043a71a24a25ad8897e7c9326c
Wheel SHA256:464050e528b5699d1788b7dcc520fcf948cea742a7a5f40391f80fc3ddc35a7b
Total basis radius5e-13,finalRGB/albedo-Jacobian radius1e-12,installed comparison abs1e-10+rel1e-9,all unchanged.

No production-rendering limit changes,no full-frame launch without admission,no Git publication,no quota increase and no history deletion. This is one pixel's reference campaign,not completion of all Group3.

## Approval boundary

D506 expressly authorized900reference seconds per piece and prohibits an automatic further campaign or cap growth. The user supplied that new explicit decision,D507. The alternative is targeted mathematical/cost improvement under the existing15-minute limit.

## Terminal outcome — 2026-10-01

The approved opaque163 campaign has completed and stopped at its first failure. Its frozen manifest SHA256 is 196929c0a17f59b9a47bd07d6345f1a1f1f0fc5afb30cd6f83fd378e07ad2020. All 69 qualification controls passed before launch.

- Piece index1, previously stopped by the 15-minute deadline, passed: 2535 cells, 1663.0155368 reference seconds, 1667.5360907 worker seconds and 632442880 sampled peak host bytes.
- Piece index0 passed freshly: 77 cells, 44.0207629 reference seconds, 48.6675764 worker seconds and 614588416 sampled peak host bytes.
- Piece index2 stopped at the unchanged 4096-cell work cap: 2639.8085376 reference seconds, 2643.4684769 worker seconds and 594685952 sampled peak host bytes. The one-hour time ceiling and depth14 ceiling were not reached.
- The last complete cover has 4095 charged cells and 2048 retained leaves, at depths5–13. Matte width2.199015436766789e-11 exceeds its allocated3.352041094003098e-16 width by about65602.28times. Smooth matte width2.571672852584752e-19 is already small; correction width2.199015411050060e-11 dominates. This is unresolved proof uncertainty, not measured renderer error.
- Result: 2of22 fresh certificates, one stopped piece,19unrun pieces, aggregation unrun; no complete-pixel or frame admission. Controller elapsed4367.3215017seconds.

Evidence: D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque163-20261001/campaign1/ and campaign-analysis1.json. The analysis preserves the three largest remaining regions and the distribution of uncertainty over subdivision depths. The approved campaign is terminal and must not be restarted or resumed. Its partial certificates cannot be mixed into a different candidate.

Next: investigate the matte-correction fallback on those three saved regions, including both integration directions, and qualify a targeted bound/partition repair before proposing a newly frozen campaign. Do not increase time, work, depth, memory or accuracy allowances automatically. Production rendering, wheel5, dashboard quota/history and M1 scope remain unchanged.
