# Group 3 centered-correction campaign preview

Status: proposed, not launched. M1 and Group 3 remain unfinished.

## Qualified repair

The three largest saved regions of terminal opaque163 piece 2 reproduced the old loose matte-correction bounds. Natural interval arithmetic did not establish the needed cosine derivative sign, causing fallback to a loose envelope. Centered C2 mean-value derivative enclosures and integration of the first coordinate first provide a conditioned proof on all three regions. The same physical integrand, positive contributions, exact coverage, downward area shares and accuracy thresholds remain.

Complete-region matte widths changed from approximately 2.51e-14 to 3.55e-18, 1.48e-18 and 9.62e-19. Median local checking time fell 40.39%, 33.56% and 34.51% across three pairs per region. These local results do not establish whole-piece completion or whole-pixel accuracy. Independent closed-form controls and broader tests: 106 passed in 30.91 seconds; three teaching checks passed. Expected red tests and the corrected missing-fixture regression failure are preserved.

## Exact proposed transaction

- Frozen manifest: `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque164-20261001/campaign-preview1.json`
- Manifest SHA256: `529b3f4ccb5428cbc336226f1ab428e239424c49c3dad5d7e675763640726372`; binds 249 files.
- Named checker: `centered-correction-v1`, offline profile `streamed22-offline-hour-v1`.
- Fresh output: `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque164-20261001/campaign1`; it must not already exist.
- Order: piece 2, then 0, 1, 3 through 21. At most 22 fresh serial piece workers, then one aggregation worker. No old certificate reuse. Stop at first failure, no automatic retry.
- Each piece: 4096 cells, depth 14, 3600 reference seconds, 3630 worker seconds, 3660 watchdog seconds, 1 GiB host, one thread.
- Aggregate piece ceilings: 90,112 cells, 79,200 reference seconds (22 hours), 79,860 worker seconds, 80,520 watchdog seconds. Aggregation adds at most 120 worker / 180 watchdog seconds. These are maxima, not estimates.
- Basis radius 5e-13, final RGB/albedo-Jacobian radius 1e-12, installed comparison absolute 1e-10 plus relative 1e-9; all 22 must pass before comparison with preserved CPU/CUDA bytes can admit this pixel.
- No rendering-limit, dashboard-quota, production-wheel, Git-publication or consumer change.

## Approval boundary

PLAN immediate next action and D507 require explicit approval of each newly frozen campaign. Prior approvals apply to terminal campaigns. Approving this preview authorizes only this fresh campaign and recording its verified outcome. After approval, recheck every bound file and actual processes, then launch with the frozen manifest hash. Stop on mismatch or first failed worker; preserve all reports.

```powershell
$env:PYTHONPATH='C:/dev/gffx'
& D:/devspace/gffx/env/Scripts/python.exe -X utf8 -m tools.opaque_streamed_reference_campaign --fixtures D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque164-20261001/campaign-preview1.json --manifest-sha 529b3f4ccb5428cbc336226f1ab428e239424c49c3dad5d7e675763640726372 --output D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque164-20261001/campaign1
```

Production remains 32 candidate faces, 16 cuts, 800 slots/32 blocks per batch, 16 batches/12,800 slots/512 blocks per pixel, 64 MiB active storage, 1 GiB CPU/2 GiB CUDA host, one thread and 120/180-second worker/watchdog limits. Full-frame admission, remaining scale/backward cases, fitting readiness and the actual qualified SIGGRAPH-quality showcase remain open.

## Terminal outcome — 1 October 2026

The user approved this exact preview in D509/Q28 resolved and requested the dashboard Current work correction first. The live card now derives Group 3 of 5 from the roadmap; 44 dashboard controls pass. The frozen 249-file checker remained unchanged.

The campaign stopped at the first failure after fresh passes for pieces 2, 0 and 1. Piece 2, the prior blocker, passed 3,657 cells and 1,729.3438935 reference seconds. Piece 0 passed 63 cells and 41.9415711 seconds; piece 1 passed 2,553 cells and 1,734.5674134 seconds.

Piece 3 (face 683) reached 4,096 cells after 3,130.4997240 reference seconds, before its time/depth caps. Its last complete 4,095-cell cover has matte width 6.668271229293596e-15 versus allowed 1.316688490913195e-15, about 5.06443 times too wide. Correction width 6.665791159509693e-15 dominates smooth matte width 2.480069783902303e-18. Ordinary/boundary component accounting is incomplete because some retained leaves used fallback; do not infer a complete component attribution.

Eighteen pieces and aggregation are unrun. No complete-pixel or full-frame acceptance. Controller time 6,664.8402858 seconds. All reports and failures remain in opaque164-20261001/campaign1; campaign-analysis1.json preserves three dominant regions for a bounded follow-up investigation. This campaign is terminal and must not be resumed or repeated automatically. No limits, quotas, history or Git publication changed.
