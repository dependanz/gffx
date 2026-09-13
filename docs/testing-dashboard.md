# gffx testing dashboard

This is a **separate developer application** in `tools/testing_dashboard`, not a layout or
a `gffx.dashboard.testing` package. It uses the installed general dashboard's `Store`,
`Connection`, and `PANEL_KINDS` objects and its server for artifact transport.
The wheel must include the HTTP extension seam and component exports introduced with this app.

## Inspect pages, not executions

Open `/testing/`. There is no run dropdown. A checked-in catalog in
`tools/testing_dashboard/catalog.py` defines permanent, grouped pages such as
`camera/projection`, `mesh/face-geometry`, and `dashboard/panel-demo`. The page is the thing
being inspected; executions supply its measurements. Pages exist even before evidence arrives.
The catalog is not the project plan and does not grant operation-support claims.

The tester opens in light mode. Use **Dark mode / Light mode** in the header to switch;
the choice is saved in this browser, independently of the general dashboard. If browser storage
is disabled, switching still works for the current page. Theme changes do not recolour image or
mesh evidence or restart the viewer connection.

The hierarchy is **Feature → individual pytest test → named check**. Expand a feature's test
row and choose **Open test page**, then **Open check page**. Every collected pytest node ID,
including parameterized instances, gets a stable test page. Named checks are explicit reporting
scopes, not automatically inferred assertions or pytest setup/call/teardown phases. Each check
has its own outcome and artifacts; it never inflates the suite's test count. A passing check does
not imply its parent passed. A missing check in a newer execution says not measured, never falls
back to old success. Test and check pages have parent links, execution history, and copyable URLs.
Unpinned feature/test pages leave check-owned visuals on the check pages; old exact artifact
links remain supported. The first instrumented operation test is TA-01 face-geometry parity:
two oracle cases × normals/areas/validity = six checks, with eight evidence panels total.

Each test page selects the newest execution **by start time** that collected a matching test.
New pending, failed, skipped or interrupted results never fall back to an older green result.
Counts cover only that execution's collected subset, not every possible fixture for a feature.
Different pages may show different revisions, so the overview deliberately does not claim a
combined suite pass. Expand **Execution details** to inspect provenance or up to nine earlier
evidence selections. No execution selector is needed to move between features and demonstrations.

**Newest pages** lists pages by their immutable catalog `created_on` date, newest first.
The overview also shows the five most recently created pages and a link to the full list.
The directory includes feature, test and named-check pages, paginated at fifty entries.
For derived test/check pages, creation means first registration as an inspection page, not the
creation date of its source code. Existing stored tests get pages when this version starts;
those page creation dates then survive restarts and subsequent executions.
Creation dates are independent of test execution dates. Same-day entries are alphabetical;
unknown dates appear last. The initial thirteen pages share September 5, 2026; exact creation
times were not recorded.

To add a page, add its stable ID, creation date, group, description and explicit node-ID patterns to the catalog.
Patterns are first-match-wins; unassigned tests stay visible under Infrastructure. Existing stored
tests are rebound on server startup without changing outcomes, timestamps or provenance.
The current finite panel demo owns `dashboard/panel-demo`; additional demonstration producers
need an explicit catalog/metadata binding before their own pages are introduced.

Page-local checks expand into phase durations and failure messages; raw producer metadata is
collapsed. Shared visual panels appear above the checks (at most 32 per page). Missing artifacts
remain explicit; older artifacts are not substituted. Mobile navigation begins collapsed and
closes after selecting a page. Pages are read-only inspection views, not browser test launchers.
Polling preserves unchanged filters, summary and pagination controls. A temporary connection
failure retains the last received evidence with an offline warning; an invalid reference still
fails visibly. Retained offline content is not a claim that the producer remains live.

## Start

From the gffx repository, using the environment containing the verified wheel:

```powershell
python -m tools.testing_dashboard serve --root dist/testing-dashboard --port 8791 --also-loopback
```

The testing interface is `/testing/`; the general-purpose interface remains `/`.
The server retains gffx's Tailscale-only binding policy, with optional additional loopback.
Port 8791 is an example that avoids an already-running general dashboard on 8790.
On the laptop use `http://127.0.0.1:8791/testing/`. Phone WebGPU needs a private **HTTPS**
Tailscale Serve proxy to this server; plain tailnet HTTP is insufficient. Do not overwrite another
service's Serve configuration. Server startup does not start tests or demo generators.

## Run tests

```powershell
python -m tools.testing_dashboard run --address http://127.0.0.1:8791 -- tests -q
```

Alternatively load `-p tools.testing_dashboard.pytest_plugin` and set
`GFFX_TEST_DASHBOARD=http://127.0.0.1:8791`. The runner selects a fresh UUID-based run.
An explicit `--run` must be unused. This is a **serial pytest** baseline, not xdist aggregation.
The original `gffx.dashboard.pytest_plugin` remains available for general dashboards;
do not load both reporters for the same session.

The test index reports collection, setup/call/teardown, terminal exit status and heartbeat.
Completion counts tests after teardown, not individual phase reports. Teardown failure overrides
call success. An interrupted or stale run is never promoted to green.
A separate heartbeat arrives every five seconds; after fifteen seconds without an event the
viewer shows stale liveness. Serial HTTP reporting has a one-second request timeout and disables
itself after a transport rejection; this is not a zero-overhead reporting claim.
Diagnostics use stderr, not Python warnings that `-W error` could turn into failures.
Evidence refusal is reported separately and cannot change an assertion's result.

## Attach visual evidence

```python
def test_projection(dashboard):
    # Make the assertion independently of the optional viewer.
    actual = 0.5
    assert actual == 0.5
    dashboard.scalar("measurement", actual, step=0)
    dashboard.text("explanation", "Closed-form synthetic fixture", step=0)
    # dashboard.image("difference", image_tensor, step=0)
    # dashboard.mesh("result", vertices, faces, step=0)
```

For independently inspectable checks, use the check-scoped evidence object:

```python
def test_projection(dashboard):
    actual = 0.5
    with dashboard.check("pixel position") as check:
        check.record("comparison", {"actual": actual, "expected": 0.5})
        # check.image("difference", image_tensor, step=0)
        assert actual == 0.5
```

The block reports pending on entry, passed on normal exit, or failure/skip/interruption while
preserving the original exception. Name checks uniquely within each test execution; nested
checks are not supported. Assertions not placed in a named block remain part of their test only.
Without the reporter plugin, tests need an optional instrumentation path (TA-01 demonstrates
this); the `dashboard` fixture itself is supplied by the opt-in plugin.

The fixture supports scalar, text, record, image, mesh and points sends. These are static
evidence sends, not an invitation to persist a large mesh on every training step. For streaming
experiments use the general dashboard's configurable queues and sparse snapshots.

Open its feature page to see its checks and shared visualization panels; expand a check for details.
An execution with no published visuals says so.
Artifact paths use SHA-256 of the exact node ID and evidence label to avoid sanitized-name
collisions. A live page reference is simply `/testing/?page=camera/projection`; it does not
depend on an execution. An artifact reference also carries its execution, path and optional step,
and pinned steps require both execution and path. **copy page reference** normally copies just
the page label and URL. **copy evidence details** includes producer-reported revision/dirty state
and evidence caveats; artifact and historical references retain these details automatically.
Send references to an agent separately; the application has no messaging system.
Missing retained artifacts and unknown page IDs are explicit, not substituted with another test.

## Plain-language test guides

Every maintained test page must explain **what it checks**, **how it checks it**, and
**what a pass means and does not mean**. These are reviewed explanations, not generated claims
from a function name. Add/update the test entry in `tools/testing_dashboard/explanations.py`
with every new or materially changed test. Parameterized instances share an explanation but
keep distinct IDs and results. Add each explicit named-check explanation to `CHECK_GUIDES`;
describe its actual comparison, artifact meaning and limits. Do not imply an unreported
assertion passed. The repository test-inventory gate rejects missing test guides, including
test class methods; the named-check regression covers the current six TA-01 checks.

The guide is labelled **current author guide**. It describes the current test source, not a
documentation snapshot captured by a historical producer. It never changes saved outcomes or
producer metadata. Unknown/external tests and unregistered checks explicitly say that their
explanation is missing; authors must resolve that before accepting a maintained page.

The opaque SHA-256 is just a stable page address. The visible reference and copied reference
use the readable page title, with full IDs and source node IDs under **Technical identifiers**.
Existing links continue to work. A passing configuration test does not mean CI ran or a GPU
operation succeeded; an expected failure is not feature acceptance, and a skipped test was not
exercised. Read the result separately from the guide.

## Coverage

Coverage.py is **optional development tooling**, never a gffx runtime dependency.
The `testing-dashboard` development group pins coverage.py 7.10.7.

```powershell
python -m tools.testing_dashboard run --address http://127.0.0.1:8791 --coverage -- tests -q
```

Coverage outputs stay under `build/testing-runs/<run>/`. The runner collects Python branch
coverage and imports its JSON after pytest, without changing pytest's exit status.
No coverage tool or failed import means **Not measured**, not zero and not 100 percent.
The coverage page shows per-file covered/executable lines, missing lines and branch counts for
the newest test execution (or an explicitly referenced execution). If that execution has no
report, it shows Not measured rather than displaying an older coverage report.
The responsive view uses an overall line-coverage bar and per-file line/branch bars with exact
covered/total counts. Filenames and package-relative directories lead each card; expand
**Source & missing lines** for the full measured path and compact missing-line ranges.
Zero covered lines out of a positive total is 0%; an empty denominator or missing measurement
has no percentage bar. Bars measure executed code, not correctness or an acceptance threshold.
It measures only the selected **Python gffx package**, not native C/CUDA and not this tool's own
code. It does not claim per-test coverage attribution. Child pytest processes are not
automatically combined into that report.

An existing report can be attached explicitly to its corresponding run:

```powershell
python -m tools.testing_dashboard coverage --address http://127.0.0.1:8791 --run RUN coverage.json
```

The importer validates counts and records the exact input artifact SHA-256. The user is responsible
for associating an imported report with the correct run; the server cannot independently verify
a producer's source claims.

## Demonstrations

```powershell
python -m tools.testing_dashboard demo --address http://127.0.0.1:8791
```

This publishes a tiny synthetic triangle checkpoint, a four-pixel image, twenty curve values and
a description, then **exits**. It does not launch an endless generator. The permanent panel-demo page
labels these as demonstrations; none contributes to test counts or proves numerical correctness.

## Storage and boundaries

The chosen root contains `testing.sqlite` and `viewer/<run>/` (general dashboard logs/identity).
The index has no lifetime execution-count cap. It defaults to 32 active metadata executions,
10,000 tests/execution, 1 MiB/event and a configurable 64 MiB SQLite budget.
`--history-keep 100` retains 100 extra unprotected older executions in addition to latest,
active and pinned evidence; it is a cleanup eligibility policy, not an admission wall.
See [pages, execution history and safe cleanup](TESTING_RETENTION.md) for History & storage,
durable pins, configurable limits and explicit preview/apply maintenance.
The SQLite rollback journal, request buffers, Python objects and GPU allocations are additional;
these are not RSS limits. Test and coverage lists return at most fifty rows/page.
Quotas reject writes without silently deleting evidence. Explicit cleanup can retire only
eligible unprotected executions; page IDs persist. SQLite metadata and pins survive a restart.
Named checks are capped at 100 per test execution; the inspection-page registry at 100,000 pages.
The existing 32-artifact metadata limit per test includes its check artifacts; the shared
128-artifact-path per-execution limit still applies. These are bounded developer defaults, not
a claim that every assertion can publish a mesh. Missing terminal test outcomes fail closed even
if a teardown report arrived; a zero pytest exit code cannot replace missing outcome evidence.
The general viewer's defaults independently cap active runs, 128 artifact paths/run, retained
bytes and log bytes. Test results do **not** consume artifact paths. Large evidence collections
can hit those limits and will report a warning; they are not an unlimited artifact archive.

The web application reads its metadata and dashboard artifacts only. It cannot start tests,
execute shell commands, fetch arbitrary local files, or send instructions to agents.
Tailnet members can publish data under the existing development trust model.
Cross-origin JSON writes are refused. Do not expose this application publicly.

Automated checks validate backend behavior and frontend state/wiring. Physical phone layout,
WebGPU framebuffer equivalence (DB-11), and sustained CPU/network budgets remain separate gates.
