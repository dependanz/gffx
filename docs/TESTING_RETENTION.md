# Pages, executions and live frames

The testing application is page-first. A page identifies a test, named check,
demonstration or article; it does not consume a session slot each time you run it.
An execution records one test invocation or demo publication. Mesh frames and
measurements inside it use the general dashboard's separate recent-window and
sparse-checkpoint controls (normally 100 recent updates, one snapshot every 1000
source steps). None of those settings is a lifetime test limit.

## Server policy

`python -m tools.testing_dashboard serve --root <saved-root> --port 8791`
accepts `--history-keep 100`, `--max-active 32`, and `--metadata-mib 64`.
The history count means extra **unprotected** older executions, plus all protected
evidence. It is an eligibility policy for explicit cleanup, not a hard admission
limit. There is no 100-execution lifetime ceiling. Active and storage limits remain
hard bounds; stale sessions are not silently marked finished. Restarting applies
the supplied policy but never removes history.

Protected evidence includes active executions, durable pins, the latest collected
and completed result of each test, the latest completed result of each named check,
latest and finished demo results, and the latest measured coverage per layer.
Latest failures are not replaced by old success. A missing check or coverage layer
in a newer execution stays unmeasured even when older evidence is retained.
An execution is protected as a whole, including its saved artifacts.

Use **History & storage** to inspect counts and limits. Under a page's **Execution
details**, Pin protects that execution across restarts; Unpin removes only this
manual protection, not automatic latest/active protection.

## Explicit cleanup

Preview without changing anything:

```powershell
python -m tools.testing_dashboard history --address http://127.0.0.1:8791
```

Review `candidates`, exact allowlisted files and `blocked` reasons. Back up evidence
you want outside this store. Apply only the returned token:

```powershell
python -m tools.testing_dashboard history --address http://127.0.0.1:8791 --apply-token <token>
```

Any changed preview requires a fresh token. Leased viewers/senders, recording
warnings, unsaved live tails, links/junctions and unknown files block removal.
Cleanup removes only selected metadata, log.bin, identity.json and saved layouts;
page identities and creation dates remain. Retired execution IDs cannot be reused.
Historical execution-specific links become unavailable, never point to new data.
This is deletion, not a reversible archive: recovery needs an external backup.

Metadata retirement is transactional. File removal follows it; a crash or file
error may leave orphan files, explicitly reported as cleanup errors when observed.
They are not automatically adopted or replayed as current evidence. Inspect these
files separately before any further removal. Shared content-addressed presentation
assets are not pruned: they can belong to multiple executions.

The metadata budget, general saved-log budget (256 MiB total / 64 MiB per log), and
shared presentation store (512 MiB pre-write check) are separate, not one total
filesystem or process-memory ceiling. SQLite reuses freed pages; its file need not
shrink. Protected evidence filling a budget stops further writes visibly rather
than being discarded. Pinning is not a reservation of disk capacity. No background
cleanup, external dependency, browser shell execution, or general-dashboard policy
change is introduced.
