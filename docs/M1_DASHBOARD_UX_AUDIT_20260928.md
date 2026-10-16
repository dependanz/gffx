# gffx dashboard UX audit — 28 September 2026

The dashboard was functioning as a developer evidence browser, but did not
explain the project to its owner. This audit used the actual in-app browser,
desktop and390x844 viewport settings. The browser's observed mobile CSS width
was355px with no horizontal document overflow. It was not a physical-phone or
touch-device acceptance test.

## Inspection scope

Visually inspected overview, newest pages, the current opaque engineering
checkpoint, rendering feature tests, history/storage, coverage, textbook index,
Kinetic Bloom, CHROMA, and the general artifact viewer. Read the complete catalog
inventory from the local API; saved as opaque138/dashboard-inventory1.json.
This covers the principal page types and representative actual visuals, not a
claim to have individually inspected thousands of generated test/check pages,
every historic run, every animation frame, or every textbook chapter.

## Findings and repairs

1. The old front door advertised6910 inspection pages and scattered pass badges,
   with no milestone roadmap. It now opens a dated five-group M1 checklist,
   scope, current blocker, and an explanation of why the opaque model changed.
   The page counts are not treated as completion percentages.
2. The rendering feature could say passed1/1 when its latest run covered only a
   square-root control. Badges now explicitly label selected checks/execution;
   the M1 overview remains unfinished regardless of that narrow pass.
3. The latest opaque demonstration was a text engineering checkpoint with all
   explanatory panels collapsed. Text-only checkpoints now show their content;
   visual demonstrations have a separate navigation entry and existing curated
   CHROMA, Bloom, camera-fitting and layer-recovery entry points.
4. The sidebar opened every category, overwhelming mobile navigation. Categories
   now start collapsed below the short primary navigation. Mobile navigation
   closes after choosing a destination; technical copying controls are hidden
   on the orientation page.
5. Raw gallery metadata and the long retention inventory were as prominent as
   the presentation. These remain available in technical disclosures. No rows,
   assets, failures or quotas were deleted or changed.
6. During indexing, an initial page request timed out with a raw AbortError.
   Existing loaded-page refresh preservation was already implemented and passes
   its regression. The gap was first load/navigation. Now an unavailable initial
   index can show only the clearly dated M1 explanation, with an explicit offline
   notice and no invented live test results. Unknown references still fail closed.
   An unnecessary directory request on the overview was also removed.
7. The general viewer at / presents raw producer artifacts, including JSON for a
   gallery. It is now labelled Raw artifact viewer from the testing shell. Use
   /testing/ as the phone entry point. The installed general viewer was not changed.
8. Coverage appeared wholly unmeasured for the selected recent execution, which
   could look like lost progress. It now states that remeasurement is deferred
   until after M1, retaining the distinction between missing and zero coverage.

## Visual assessment

CHROMA and Bloom show actual saved graphics; their provenance and limits are
visible. They are useful examples of working components. They are not the final
original-mesh SIGGRAPH-quality3D showcase. The remaining visual work is a curated
original-renderer narrative after its numerical/gradient/full-frame gates pass,
including meaningful before/after comparisons and readable mobile controls.
No generated substitute image or cosmetic retouching was used.

## Validation and retained failures

77 JavaScript viewer/navigation tests pass with the installed opaque128 assets;
265 selected checker/teaching/diagnostic/dashboard tests pass. All344 pytest
outcomes (332pass/12fail) remain, including the failed pre-cache control. Initial dashboard test invocations had a missing
GFFX_DASHBOARD_STATIC setting and then an unsupported Node flag; corrected
invocations are saved, and no claim is made for the invalid invocations. The
first checkpoint publisher rejected a path containing spaces; its failed run
is preserved and the corrected producer uses pc-opaque138-bounds2-20260928.

Paired stopped-service backup, full prior-row/asset preservation, SQLite
integrity and12 local/Tailscale HTTP checks pass (dashboard-audit2.json). Runtime,
installed wheel, historical certificates and user Git work remain unchanged.

## Remaining UX work

The archive still uses many technical test names, and a feature page still shows
one selected execution rather than a consolidated contract dashboard. These are
explicitly labelled and moved behind the orientation view. A future consolidated
acceptance view must resolve evidence scopes and revisions, not count unrelated
green runs together. Initial indexing can still be slow under writes; the new
fallback explains this but is not a backend latency qualification. No blanket
claim of visual acceptance for every page, device or historical artifact is made.


The attempted SQL rewrite preserved all94 test pages' ordered run lists but
was slower (3.707455s old versus58.760467s candidate in the read-only comparison),
so it was not adopted. The deployed improvement caches only the small unchanged
catalog projection, invalidates local writes and file changes, recomputes stale
worker flags and protects against caller mutation. Three sequential local
requests returned161 pages in0.0399605/0.0165034/0.007101seconds. These are warm
request observations, not a cold-load or physical-phone performance guarantee.
The cache and existing selected-execution rules pass five targeted regressions.
An attempted safety stop after the slow query comparison found it already
finished; no process was stopped by that command, and the result remains saved.
