# Live windows and saved snapshots

## Referencing a testing page

Each run has a persistent UUID in `identity.json`; each saved layout has a stable page UUID
within that run. The page header shows both IDs and producer-reported revision/dirty state.
Use **copy page reference** for context to paste into your coding-agent conversation, or
**copy link** for a direct URL. Each panel's **ref** button identifies its path; mesh/image
panels also include the frame actually shown. If clipboard permission is denied, a selectable
text field provides a manual-copy fallback. There are no messaging boards or agent actions.

Pass provenance explicitly from the producing checkout/build, not from the dashboard host:

```python
import gffx.dashboard as gd

dash = gd.connect(address, run="camera-test-001", provenance={
    "revision": full_commit_sha,
    "dirty": True,
    "artifact_sha256": wheel_sha256,
})
```

These are **producer-reported**, not independently verified identities. Omitted fields remain
unknown. Conflicting provenance is refused for an existing run; use a new run for another build.
An unknown first producer cannot be retroactively relabelled as a known revision. Keep the run
directory, including `identity.json` and layouts, when moving saved results between devices.
Legacy runs without an identity file initially derive their ID from the absolute run directory;
opening them persists that ID before future moves.

Links carry run name, run UUID, page UUID and optional artifact path/step. Explicit page selection
takes precedence over browser preferences. Wrong-run/missing-page references fail visibly rather
than selecting a substitute. Missing or evicted steps report unavailable; an identifier does not
make an ephemeral frame durable. Mesh/image links pin the requested step; other panels expose
live/path references, not snapshot replay. A changed saved layout keeps its identity; saving under
another name creates another page identity. Unsaved layout edits are not a portable saved page.
Default auto/proposed views also have IDs but remain dynamic views, not immutable evidence.

Opening an existing run automatically replays its log. The saved layout selector provides page
navigation in this slice; a collapsible navigation sidebar and full development-monitoring
workspace (coverage, lifecycle/heartbeat and scalable test catalog) are not implemented here.

## Streaming setup

`gffx.dashboard` is an optional development monitor. It adds no runtime dependencies and
does not start a server on import. Connect to a dashboard server you have explicitly started.

For high-rate meshes, points or images, use a queue rather than `dash.mesh(...)` / `set`,
which records every accepted call:

```python
import gffx.dashboard as gd

dash = gd.connect("YOUR_TAILSCALE_IP:8790", run="experiment")  # replace with your server address
meshes = dash.queue(
    "fit/mesh", "mesh",
    recent_window=100,       # maximum recent updates held in server memory
    snapshot_every=1000,     # persist one full frame at steps 0, 1000, 2000, ...
    max_hz=10,               # optional cap on ordinary live sends; None disables it
)

for step in range(10_000):
    vertices, faces = compute_mesh(step)  # your experiment
    meshes.push(vertices, faces, step=step)

# Changes apply on subsequent pushes. They do not erase existing saved snapshots.
meshes.configure(recent_window=200, snapshot_every=5000, max_hz=5)
dash.close()
```

`recent_window=100` and `snapshot_every=1000` are the defaults. `bound` remains an alias
for `recent_window`; do not supply both. Set `snapshot_every=0` to disable snapshot logging,
and `max_hz=None` to disable producer-side rate limiting. The browser's Low/Standard/Paused
delivery settings are independent and never change recording policy.

## Steps, rate limiting and capacity

- Explicit steps are nonnegative, strictly increasing integers per queue path. Omit `step`
  to count calls to that Queue object from zero, **including throttled calls**. On resuming a
  run, explicitly continue beyond the server's latest recorded step; a new Queue object cannot
  infer the old producer's counter. Use one writer per path.
- The interval is measured in **source steps**, not seconds or browser messages. A snapshot is
  selected when `step % snapshot_every == 0`, including step zero. If the producer does not
  call `push` at a selected step, that snapshot does not exist; the dashboard does not invent it.
- Snapshot selection precedes rate limiting and array copying. Selected snapshots bypass
  `max_hz`. Ordinary throttled calls return before converting/copying arrays; `meshes.throttled`
  counts them. This does not reduce the cost of `compute_mesh` itself.
- The recent window holds the most recent **accepted** source updates, not necessarily
  consecutive source steps. Server byte limits may retain fewer frames: defaults are 16 MiB
  per path and 64 MiB per run, with a server ceiling of 2,048 entries per path.
- Sender buffering is bounded by count and bytes. Selected snapshots are protected from
  ordinary drop-oldest eviction. If the queue contains only protected snapshots, a new frame
  may still be refused. Inspect `dash.dropped`, `dash.checkpoint_dropped`, `dash.errors` and
  `dash.last_error`; nonblocking is not a lossless guarantee.
- `wait_acks=True` waits for server acceptance and raises on rejection. A checkpoint ack means
  its log write was flushed, not a power-loss/fsync guarantee. Connection errors are visible;
  automatic retry and exactly-once delivery are not claimed.

## Browsing and persistence

The mesh/image panel's normal scrubber addresses the recent server window, including retained
steps that were skipped by browser-delivery coalescing. **Snapshots** switches to a page of
up to 100 saved steps; **older** loads an earlier page; **live** returns to recent updates.
The policy label shows the effective recent-window limit and configured snapshot interval.
Physical-browser verification is separate from automated state/transport tests.

```python
page = gd.PageConnection("YOUR_TAILSCALE_IP:8790", run="experiment")  # replace with your server address
page.open()
saved = page.checkpoints("fit/mesh", limit=100)
frame = page.history("fit/mesh", saved["steps"][0])
older = page.checkpoints("fit/mesh", before=saved["steps"][0], limit=100)
page.close()
```

Each saved snapshot is one **complete** frame, not the whole recent window. The initial
implementation repeats topology in those sparse frames for independent replay; browser mesh
delta compression never enters the archive. No recent-window dump is added at shutdown for
new queues, even with snapshots disabled. Legacy protocol-1 senders without snapshot fields
retain their old clean-shutdown-tail behavior, and their old logs remain readable.

Checkpoints are read on demand from the bounded append-only run log; pagination retains at
most 101 step IDs, not an unbounded in-memory archive. Scans use a fixed log-length boundary
without holding the producer lock during disk I/O. They are not constant-time indexed lookups.
Replay restores the most recent *saved* frames within the live limits, not every ephemeral
frame seen before stopping. Deleting a path hides its older checkpoints from browsing but
does not erase the log bytes.

Disk quotas remain 64 MiB per run log and 256 MiB across run logs by default. A due checkpoint
that exceeds quota is rejected before live-state mutation, with a recording warning. Ordinary
non-checkpoint pushes may continue. Nothing automatically deletes or compacts existing logs;
raising server limits or archiving logs is a separate explicit action. Saved layouts and other
runtime overhead are outside these log quotas and retained-payload figures.
