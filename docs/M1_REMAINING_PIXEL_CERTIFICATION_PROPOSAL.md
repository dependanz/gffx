# Remaining original128 pixel certification scope

Status: approved by the user in D518, including bounded continuation through qualified frozen trials. The earlier separate routine trial-approval pause below is superseded for this scope; numerical, resource and first-failure safeguards remain.

The selected pixel (59,10) passed all22 pieces in opaque168. Preserve it without rerunning. The remaining preselected pixels (48,15) and (67,117) have33 and28 exact positive-area visible pieces respectively, from21 and18 candidate faces. Candidate faces and clipped integration pieces are different counts. Production remains within its32-candidate-face cap.

Opaque169 independently checked containment, depth ownership, all906 pairwise zero-area intersections, total mask area, and saved CPU/CUDA report identity. Both pass. Their RGB/albedo Jacobians still need independent numerical certification. Of61 pieces,51 are certified view-front,one view-back,and9 cross or need further view-horizon classification. No positive area was removed. A diagnostic polynomial-division error and its failed report are retained; corrected exact rational multiplication passes.

## Proposed bounded extension

Add explicit input-bound offline profiles for these two exact saved inputs only. Retain each existing per-piece ceiling:8192 cells,depth14,7200 reference seconds,7230 whole-worker seconds,7260 watchdog seconds,1GiB,one thread. Use the qualified horizon-correction checker;do not adopt experimental root contraction.

At most61 fresh serial piece workers and two120-second/180-watchdog aggregation workers,one complete aggregation per pixel. Stop the entire campaign at its first failure;no automatic retries,resume,substitution,or old certificates. Worst-case summed ceilings499712 cells,439200 reference seconds(122hours),441510 worker seconds,and443220 watchdog seconds. These are ceilings,not an ETA; the previous22-piece campaign finished in about3hours3minutes.

Extend the reference checker’s visible-piece selector to33 ONLY for the exact33-piece profile;the default32-piece reference cap and historical22-piece contract remain unchanged. Pin exact input bytes,canonical content,individual pieces,complete area allocations,and producer/dependency/installed-wheel identities. Reject missing,duplicate,reordered,mixed-profile,and altered certificates. Each pixel separately conserves basis radius5e-13 and finalRGB/albedo-Jacobian radius1e-12. Installed comparisons retain absolute1e-10 plus relative1e-9. Preserve every positive contribution.

Qualification before a full trial: test changed input/certificate/profile rejection,33/28 enumeration and complete aggregation,supervisor accounting and first-failure stop; run appropriate independent analytical controls and bounded saved-region checks. Then freeze the implementation and both exact inputs into one reviewable campaign preview. The existing plan requires separate approval of that exact frozen trial.

Production rendering stays unchanged:32candidate faces,16cuts,800slots/32blocks per batch,16batches/12800slots/512blocks per pixel,64MiBactive,1GiBCPU/2GiBCUDAhost,one thread,120-second worker/180-second watchdog. No Git publication,dashboard quota increase,or history deletion. Original128/256 startup approval D504 already exists;complete accuracy/storage/timing admission remains mandatory.

## Frozen input scope

- Case1: 33pieces; input `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque169-20261002/pixel1-input2.json`; bytesSHA `a148bf721ba478e1f52821bf8f4d48f9fcbd67e6aecf51390caea42076c22e86`; canonicalSHA `6b6f96ab16913c69216277c43ce600c824f8be52b12a0c148844e91a742dab55`.
- Case2: 28pieces; input `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque169-20261002/pixel2-input2.json`; bytesSHA `de96eae0e545dc3633587049ce82e31540b81006e0b7e2a60b9fe7eab0e671af`; canonicalSHA `4135fda3245db8d7a757b976e9f49b906d990c48d275975f8a91750fa087d78f`.

Evidence: `D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque169-20261002/partition-summary1.json` and the two partition-audit reports.
