"""Resource safety is distinct from display conformance and performance claims."""
import pytest
from gffx import dashboard as gd
from gffx.dashboard import _server, _state


def limits(**kwargs):
    return gd.ResourceLimits(**kwargs)


def test_set_history_and_queue_are_bounded_and_report_evictions():
    tree = _state.StateTree(limits=limits(history_entries=3, text_entries=2))
    for step in range(10):
        tree.set("loss", "scalar", step, float(step))
        tree.set("note", "text", step, str(step))
    assert list(tree.series("loss")) == [7, 8, 9]
    assert list(tree.series("note")) == [8, 9]
    assert tree.snapshot()["resources"]["evicted_entries"] == 15
    assert tree.snapshot()["paths"]["loss"]["retained_from"] == 7


def test_byte_budget_evicts_old_values_and_rejects_oversize_without_mutation():
    tree = _state.StateTree(limits=limits(path_bytes=1024, run_bytes=2048))
    for step in range(20):
        tree.push("text", "text", "x" * 400, bound=100)
    assert tree.snapshot()["resources"]["retained_bytes"] <= 1024
    before = tree.snapshot()
    with pytest.raises(gd.ProtocolError, match="byte"):
        tree.set("huge", "text", 0, "z" * 4096)
    assert tree.snapshot() == before


def test_total_run_budget_and_path_count_are_bounded():
    tree = _state.StateTree(limits=limits(path_bytes=1024, run_bytes=1500, paths=3))
    for i in range(3):
        tree.set(str(i), "text", i, "x" * 500)
    assert tree.snapshot()["resources"]["retained_bytes"] <= 1500
    with pytest.raises(gd.ProtocolError, match="path"):
        tree.set("extra", "text", 0, "small")


def test_log_quota_preserves_existing_bytes(tmp_path):
    path = tmp_path / "log.bin"
    log = _server.RunLog(path, max_bytes=100)
    try:
        log.append_text({"t": "hello"})
        before = path.read_bytes()
        with pytest.raises(gd.ProtocolError, match="disk"):
            log.append_text({"value": "x" * 100})
        assert path.read_bytes() == before
    finally:
        log.close()


def test_page_delivery_coalesces_and_hidden_page_keeps_no_pending_frames(tmp_path):
    run = _server.Run("test", tmp_path)
    page = _server._PageSession(None, run)
    for i in range(1000):
        page.enqueue({"t": "update", "path": "mesh", "step": i, "value": i}, None)
    assert len(page.pending) == 1
    assert page.pending["mesh"][0]["step"] == 999
    page.configure({"paused": True})
    page.enqueue({"t": "update", "path": "mesh", "step": 1001}, None)
    assert not page.pending
    assert page.pending_bytes == 0


def test_mesh_delta_is_connection_local_and_changed_topology_is_resent():
    from gffx.dashboard._delivery import MeshDeltaCache
    view = lambda data: gd.ArrayView("uint8", (len(data),), data)
    cache = MeshDeltaCache(max_bytes=1024)
    arrays = [("vertices", view(b"abc")), ("faces", view(b"123")), ("colors", view(b"rgb"))]
    h, sent = cache.encode({"path": "mesh", "step": 1, "kind": "mesh"}, arrays)
    assert len(sent) == 3 and not h.get("mesh_delta")
    h, sent = cache.encode({"path": "mesh", "step": 2, "kind": "mesh"}, arrays)
    assert [name for name, _ in sent] == ["vertices"] and h["base_step"] == 1
    changed = [arrays[0], ("faces", view(b"321"))]  # colors removed
    h, sent = cache.encode({"path": "mesh", "step": 3, "kind": "mesh"}, changed)
    assert [name for name, _ in sent] == ["vertices", "faces"]
    assert h["mesh_fields"] == ["vertices", "faces"]
    cache.clear()
    h, sent = cache.encode({"path": "mesh", "step": 4, "kind": "mesh"}, arrays)
    assert len(sent) == 3 and not h.get("mesh_delta")


def test_resource_limits_validate_positive_integers():
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            limits(history_entries=value)


def test_disk_limit_rejection_does_not_change_live_state(tmp_path):
    server = gd.Server(tmp_path, limits=limits(log_bytes=1024), allow_loopback_for_tests=True, port=0).start()
    client = gd.connect(server.address, run="limited", wait_acks=True)
    try:
        client.text("note", "ok", step=1)
        with pytest.raises(Exception, match="disk"):
            client.text("note", "x" * 2000, step=2)
        assert server.state.get("note", run="limited")["value"] == "ok"
    finally:
        client.close(); server.stop()


def test_sender_queue_has_a_byte_budget():
    from gffx.dashboard._client import ByteQueue
    import queue
    pending = ByteQueue(64, max_bytes=100)
    pending.put_nowait((1, [b"x" * 60]))
    with pytest.raises(queue.Full):
        pending.put_nowait((2, [b"y" * 60]))
    assert pending.bytes == 60
    pending.get_nowait()
    assert pending.bytes == 0


def test_slow_page_socket_does_not_hold_sender_queue_lock(tmp_path):
    import threading
    entered, release, enqueued = threading.Event(), threading.Event(), threading.Event()
    class Socket:
        def send_text(self, payload):
            entered.set(); release.wait(3)
    page = _server._PageSession(Socket(), _server.Run("test", tmp_path))
    page.start()
    try:
        page.enqueue({"t": "update", "path": "a", "value": 1}, None)
        assert entered.wait(2)
        worker = threading.Thread(target=lambda: (page.enqueue({"t": "update", "path": "b", "value": 2}, None), enqueued.set()))
        worker.start()
        assert enqueued.wait(1), "slow socket blocked sender"
        worker.join(1)
    finally:
        release.set(); page.stop(); page._writer.join(2)


def test_page_pacing_and_pause_resume_over_wire(tmp_path):
    import time
    server = gd.Server(tmp_path, allow_loopback_for_tests=True, port=0).start()
    client = gd.connect(server.address, run="rate", wait_acks=True)
    page = gd.PageConnection(server.address, run="rate", timeout=3)
    try:
        page.open(view={"fps": 2, "bytes_per_second": 131072})
        start = time.monotonic()
        for i in range(3):
            client.scalar("value%d" % i, i, step=i)
        for _ in range(3):
            page.next_update(timeout=3)
        assert time.monotonic() - start >= 0.9
        page._request({"t": "view_config", "paused": True}, ("ack",))
        client.scalar("value0", 9, step=9)
        with pytest.raises(TimeoutError):
            page.next_update(timeout=0.2)
        page._request({"t": "view_config", "paused": False}, ("ack",))
        values = [page.next_update(timeout=3) for _ in range(3)]
        assert any(v["path"] == "value0" and v["value"] == 9 for v in values)
    finally:
        page.close(); client.close(); server.stop()


def test_full_disk_does_not_partially_append_shutdown_tail(tmp_path):
    run = _server.Run("tail", tmp_path / "tail", limits=limits(log_bytes=64))
    run.open_log()
    run.apply({"t": "push", "path": "q", "kind": "scalar", "value": 1})
    before = run.log.path.read_bytes()
    with pytest.warns(RuntimeWarning, match="NOT saved"):
        assert run.record_queue_tails() == 0
    assert run.log.path.read_bytes() == before
    run.log.close()


def test_negotiated_mesh_delta_and_resume_send_full_frame(tmp_path):
    server = gd.Server(tmp_path, allow_loopback_for_tests=True, port=0).start()
    client = gd.connect(server.address, run="mesh", wait_acks=True)
    page = gd.PageConnection(server.address, run="mesh")
    verts, faces = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]]
    try:
        page.open(view={"fps": 30, "mesh_deltas": True})
        stream = client.queue("mesh", "mesh", bound=4)
        stream.push(verts, faces)
        first = page.next_update()
        assert "faces" in first["arrays"] and not first.get("mesh_delta")
        stream.push(verts, faces)
        second = page.next_update()
        assert second["mesh_delta"] and second["base_step"] == first["step"]
        assert list(second["arrays"]) == ["vertices"]
        page._request({"t": "view_config", "paused": True}, ("ack",))
        page._request({"t": "view_config", "paused": False}, ("ack",))
        resumed = page.next_update()
        assert "faces" in resumed["arrays"] and not resumed.get("mesh_delta")
    finally:
        page.close(); client.close(); server.stop()


def test_disk_budget_is_shared_across_runs(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = _server.RunLog(tmp_path / "a" / "log.bin", total_bytes=100)
    b = _server.RunLog(tmp_path / "b" / "log.bin", total_bytes=100)
    try:
        a.append_text({"text": "x" * 50})
        with pytest.raises(gd.ProtocolError, match="disk"):
            b.append_text({"text": "y" * 50})
        assert b.path.stat().st_size == 0
    finally:
        a.close(); b.close()


def test_page_byte_budget_rejects_oversize_and_keeps_latest(tmp_path):
    run = _server.Run("test", tmp_path, limits=limits(page_bytes=1024))
    page = _server._PageSession(None, run)
    for i in range(20):
        page.enqueue({"path": str(i), "value": "x" * 200}, None)
    assert page.pending_bytes <= 1024
    assert "19" in page.pending
    page.enqueue({"path": "huge", "value": "x" * 2048}, None)
    assert "huge" not in page.pending
    assert page.dropped > 0
