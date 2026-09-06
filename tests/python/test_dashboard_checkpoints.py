"""Sparse durable snapshots are independent of recent live retention and page pacing."""
import pytest
from gffx import dashboard as gd
from gffx.dashboard import _server, _client


@pytest.fixture
def stream(tmp_path):
    server = gd.Server(tmp_path, allow_loopback_for_tests=True, port=0).start()
    client = gd.connect(server.address, run="snapshots", wait_acks=True)
    try:
        yield server, client
    finally:
        client.close()
        server.stop()


def test_recent_window_and_sparse_checkpoints_are_independent(stream):
    server, client = stream
    q = client.queue("mesh", "scalar", recent_window=3, snapshot_every=5)
    for step in range(13):
        q.push(float(step), step=step)
    assert list(server.state.series("mesh")) == [10, 11, 12]
    page = gd.PageConnection(server.address, run="snapshots")
    try:
        snap = page.open()
        assert snap["paths"]["mesh"]["snapshot_every"] == 5
        assert page.checkpoints("mesh")["steps"] == [0, 5, 10]
        assert page.history("mesh", 0)["value"] == 0
        assert page.checkpoints("mesh", before=10, limit=1)["steps"] == [5]
        with pytest.raises(gd.ProtocolError):
            page.history("mesh", 1)  # evicted, not a checkpoint
    finally:
        page.close()


def test_checkpoint_replay_without_shutdown_tail(stream):
    server, client = stream
    q = client.queue("value", "scalar", recent_window=2, snapshot_every=3)
    for step in range(8):
        q.push(step, step=step)
    run = server.state.run("snapshots")
    log = run.directory / "log.bin"
    before = log.read_bytes()
    assert run.record_queue_tails() == 0
    assert log.read_bytes() == before
    fresh = gd.Server(server.root, allow_loopback_for_tests=True, port=0)
    fresh.replay("snapshots")
    assert list(fresh.state.series("value")) == [3, 6]
    restored = fresh.state.run("snapshots")
    assert restored.checkpoints("value")["steps"] == [0, 3, 6]
    assert restored.history("value", 0)[2] == 0.0


def test_disabled_checkpoints_keep_no_durable_frames(stream):
    server, client = stream
    q = client.queue("value", "scalar", recent_window=2, snapshot_every=0)
    for i in range(6):
        q.push(i, step=i)
    run = server.state.run("snapshots")
    assert run.checkpoints("value")["steps"] == []
    assert run.record_queue_tails() == 0
    assert list(run.tree.series("value")) == [4, 5]


def test_source_throttle_never_skips_due_checkpoint(monkeypatch, stream):
    server, client = stream
    monkeypatch.setattr(_client.time, "monotonic", lambda: 100.0)
    q = client.queue("value", "scalar", recent_window=3, snapshot_every=5, max_hz=1)
    for i in range(12):
        q.push(i, step=i)
    assert q.throttled == 9
    assert list(server.state.series("value")) == [0, 5, 10]
    assert server.state.run("snapshots").checkpoints("value")["steps"] == [0, 5, 10]


@pytest.mark.parametrize("options", [
    {"recent_window": 0}, {"recent_window": True}, {"recent_window": 1.5},
    {"snapshot_every": -1}, {"snapshot_every": True}, {"snapshot_every": 2.5},
    {"max_hz": 0}, {"max_hz": float("nan")}, {"bound": 3, "recent_window": 4},
])
def test_invalid_queue_policy_rejected(stream, options):
    _, client = stream
    with pytest.raises((ValueError, gd.ProtocolError)):
        client.queue("value", "scalar", **options)


def test_settings_can_change_without_erasing_checkpoints(stream):
    server, client = stream
    q = client.queue("value", "scalar", recent_window=4, snapshot_every=5)
    for i in range(6):
        q.push(i, step=i)
    q.configure(recent_window=2, snapshot_every=3)
    for i in range(6, 10):
        q.push(i, step=i)
    run = server.state.run("snapshots")
    assert list(run.tree.series("value")) == [8, 9]
    assert run.checkpoints("value")["steps"] == [0, 5, 6, 9]


def test_checkpoint_quota_error_is_visible_and_does_not_mutate_live(stream):
    server, client = stream
    q = client.queue("value", "scalar", snapshot_every=2)
    q.push(1, step=1)
    run = server.state.run("snapshots")
    run.log.max_bytes = run.log.path.stat().st_size
    with pytest.raises(gd.ProtocolError, match="disk quota"):
        q.push(2, step=2)
    assert run.tree.get("value")["step"] == 1
    q.push(3, step=3)
    assert run.tree.get("value")["step"] == 3


def test_defaults_and_source_steps_before_copy(monkeypatch, stream):
    server, client = stream
    q = client.queue("value", "scalar")
    assert q.recent_window == 100 and q.snapshot_every == 1000
    q.push(0, step=1000)
    before = server.state.snapshot()
    for step in (999, 1000, -1, True, 1.5):
        with pytest.raises((ValueError, gd.ProtocolError)):
            q.push(0, step=step)
    assert server.state.snapshot() == before


def test_mesh_checkpoint_self_contained_after_live_eviction(stream):
    import struct
    server, client = stream
    q = client.queue("mesh", "mesh", recent_window=1, snapshot_every=2)
    faces = gd.ArrayView("uint32", (1, 3), struct.pack("<3I", 0, 1, 2))
    for i in range(4):
        vertices = gd.ArrayView("float32", (3, 3), struct.pack("<9f", *([float(i)] * 9)))
        q.push(vertices, faces, step=i)
    run = server.state.run("snapshots")
    kind, step, value = run.history("mesh", 0)
    assert kind == "mesh" and step == 0
    assert value["faces"].data == faces.data
    assert value["vertices"].data == bytes(36)
    client.delete("mesh")
    assert run.checkpoints("mesh")["steps"] == []
    with pytest.raises(KeyError):
        run.history("mesh", 0)


def test_pending_checkpoints_are_protected_and_overload_is_visible():
    dash = _client.Dashboard(None, "unit", "unit", wait_acks=False, queue_bound=1, timeout=1)
    dash.connected = True  # no worker: deterministic full-queue test
    q = dash.queue("value", "scalar", snapshot_every=2)
    q.push(0, step=0)
    q.push(1, step=1)
    assert dash.dropped == 1 and dash.checkpoint_dropped == 0
    q.push(2, step=2)
    assert dash.dropped == 2 and dash.checkpoint_dropped == 1
    assert "protected checkpoints" in dash.last_error
    assert dash._queue.qsize() == 1 and dash._inflight == 1


def test_due_checkpoint_evicts_ordinary_pending_update():
    dash = _client.Dashboard(None, "unit", "unit", wait_acks=False, queue_bound=1, timeout=1)
    dash.connected = True
    q = dash.queue("value", "scalar", snapshot_every=2)
    q.push(1, step=1)
    q.push(2, step=2)
    assert dash.dropped == 1 and dash.checkpoint_dropped == 0
    assert dash._queue.get_nowait()[2] is True


def test_recent_window_still_respects_bytes_and_checkpoint_pages_are_bounded(tmp_path):
    run = _server.Run("unit", tmp_path, limits=gd.ResourceLimits(path_bytes=1024))
    run.open_log()
    try:
        for step in range(130):
            event = {"t": "push", "path": "x", "kind": "scalar", "bound": 100,
                     "snapshot_every": 1, "step": step, "value": step, "checkpoint": True}
            run.apply(event)
            run.record_text(event)
        assert len(run.tree.series("x")) <= 3
        assert run.tree.resources()["retained_bytes"] <= 1024
        result = run.checkpoints("x")
        assert result == {"steps": list(range(30, 130)), "has_more": True}
        assert run.checkpoints("x", before=30) == {"steps": list(range(30)), "has_more": False}
        assert run.history("x", 0)[2] == 0.0
    finally:
        run.log.close()


def test_slow_checkpoint_scan_does_not_hold_producer_lock(monkeypatch, stream):
    import threading
    server, client = stream
    client.queue("x", "scalar").push(0, step=0)
    run = server.state.run("snapshots")
    entered, release = threading.Event(), threading.Event()
    original = _server.RunLog.read
    def slow_read(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        yield from original(*args, **kwargs)
    monkeypatch.setattr(_server.RunLog, "read", slow_read)
    thread = threading.Thread(target=lambda: run.checkpoints("x"))
    thread.start()
    try:
        assert entered.wait(2)
        assert run.lock.acquire(timeout=0.5), "archive scan blocked producer"
        run.lock.release()
    finally:
        release.set()
        thread.join(3)
