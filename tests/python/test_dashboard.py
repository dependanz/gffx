"""DASHBOARD_CONTRACT_V0_1.md: the dashboard.

Red baseline for the `dashboard` subproject's Phase 0. Every fixture here is specified in section
11 and fails on the absence of `gffx.dashboard`, not on a version gate and not on an assertion.
DB-11 needs a headless browser with WebGPU and skips, visibly, where none is present.

These fixtures drive the server in-process on the loopback interface with the Tailscale rule
explicitly relaxed (`allow_loopback_for_tests=True`), because a test must not depend on the
tailnet being up; DB-07 is the one fixture that exercises the rule itself.
"""

from __future__ import annotations

import os
import shutil
import threading
import time

import pytest
import torch

try:
    from gffx import dashboard as gd  # the facade of API_CONTRACT_V0_1.md section 2.1
    _RED = None
except ImportError as _absent:  # the red baseline: the module does not exist yet
    gd = None
    _RED = str(_absent)

# While `gffx.dashboard` is absent every fixture here is an expected failure, strictly: the moment the
# module exists these run for real, and an unexpected pass is reported as a failure so the marker
# cannot outlive the baseline it guards.
pytestmark = pytest.mark.xfail(_RED is not None, reason="red baseline: " + str(_RED), strict=True)


def _require_module():
    if gd is None:
        pytest.xfail("red baseline: " + _RED)



@pytest.fixture
def server(tmp_path):
    _require_module()
    srv = gd.Server(root=tmp_path, allow_loopback_for_tests=True, port=0)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def client(server):
    _require_module()
    dash = gd.connect(server.address, run="test-run", name="pytest", wait_acks=True)
    try:
        yield dash
    finally:
        dash.close()


def test_db01_set_reaches_a_page_as_snapshot_then_update(server, client):
    page = gd.PageConnection(server.address, run="test-run")
    client.scalar("fit/loss", 0.5, step=1)
    snap = page.open()
    assert snap["paths"]["fit/loss"]["kind"] == "scalar"
    assert snap["paths"]["fit/loss"]["latest"]["step"] == 1
    assert snap["paths"]["fit/loss"]["latest"]["value"] == 0.5
    client.scalar("fit/loss", 0.25, step=2)
    upd = page.next_update(timeout=2.0)
    assert upd["t"] == "update" and upd["path"] == "fit/loss" and upd["step"] == 2
    assert upd["value"] == 0.25
    page.close()


def test_db02_mesh_survives_binary_framing_bit_for_bit(server, client):
    g = torch.Generator().manual_seed(0)
    verts = torch.rand(5023, 3, generator=g)
    faces = torch.randint(0, 5023, (9976, 3), generator=g, dtype=torch.int32)
    colors = torch.rand(5023, 3, generator=g)
    client.mesh("fit/head", verts, faces, colors=colors, step=7)
    got = server.state.get("fit/head", step=7)
    assert got["kind"] == "mesh"
    assert torch.equal(torch.from_numpy(got["vertices"]), verts)
    assert torch.equal(torch.from_numpy(got["faces"]).to(torch.int32), faces)
    assert torch.equal(torch.from_numpy(got["colors"]), colors)
    assert got["array_order"] == ["vertices", "faces", "colors"]


def test_db03_push_past_the_bound_evicts_the_oldest(server, client):
    q = client.queue("fit/stream", "scalar", bound=4)
    for i in range(10):
        q.push(float(i))
    client.flush()
    series = server.state.series("fit/stream")
    assert len(series) == 4
    assert [v["value"] for v in series.values()] == [6.0, 7.0, 8.0, 9.0]


def test_db04_a_control_set_on_the_page_reaches_the_client_as_a_live_value(server, client):
    lr = client.control("controls/lr", 1e-3, kind="slider", range=(1e-5, 1e-1))
    assert lr.value == 1e-3
    page = gd.PageConnection(server.address, run="test-run")
    page.open()
    t0 = time.perf_counter()
    page.control_set("controls/lr", 5e-4)
    deadline = t0 + 2.0
    while lr.value != 5e-4 and time.perf_counter() < deadline:
        time.sleep(0.005)
    assert lr.value == 5e-4
    assert (time.perf_counter() - t0) < 0.1  # the contract's 95th-percentile bound, on loopback
    series = server.state.series("controls/lr")
    assert list(series.values())[-1]["value"] == 5e-4
    page.close()


def test_db05_a_run_log_replays_to_the_same_tree(server, client, tmp_path):
    client.scalar("a/x", 1.0, step=0)
    client.text("a/note", "hello", step=0)
    client.scalar("a/y", 2.0, step=0)
    client.delete("a/y")
    client.scalar("a/x", 3.0, step=1)
    client.flush()
    before = server.state.snapshot()
    log = server.run_dir("test-run") / "log.bin"
    assert log.is_file()
    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    shutil.copytree(server.run_dir("test-run"), fresh_root / "test-run")
    fresh = gd.Server(root=fresh_root, allow_loopback_for_tests=True, port=0)
    fresh.replay("test-run")
    assert fresh.state.snapshot() == before
    assert "a/y" not in fresh.state.snapshot()["paths"]


def test_db06_layouts_round_trip(server):
    page = gd.PageConnection(server.address, run="test-run")
    page.open()
    spec = {"name": "fit", "columns": 12, "panels": [
        {"kind": "mesh", "path": "fit/head", "col": 0, "row": 0, "w": 8, "h": 6},
        {"kind": "plot", "path": "fit/loss", "col": 8, "row": 0, "w": 4, "h": 3},
    ]}
    page.layout_save(spec)
    assert "fit" in page.layout_list()
    assert page.layout_load("fit") == spec
    page.close()


def test_db07_the_server_binds_only_the_tailscale_interface(tmp_path):
    for bad in ("0.0.0.0", "8.8.8.8", "127.0.0.1"):
        with pytest.raises(gd.BindRefused) as excinfo:
            gd.Server(root=tmp_path, host=bad, port=0).start()
        assert "Tailscale" in str(excinfo.value)
    addr = gd.tailscale_address()
    if addr is None:
        pytest.skip("no Tailscale address in the device catalog for this host")
    srv = gd.Server(root=tmp_path, host=addr, port=0)
    srv.start()
    try:
        assert srv.address.startswith(addr)
    finally:
        srv.stop()


def test_db08_the_pytest_plugin_posts_outcomes_and_evidence(server, tmp_path, pytester):
    pytester.makepyfile(
        """
        import torch
        def test_pass(dashboard):
            dashboard.scalar("metric", 1.0)
            assert True
        def test_fail():
            assert False
        """
    )
    os.environ["GFFX_DASHBOARD"] = server.address
    try:
        result = pytester.runpytest("-p", "gffx.dashboard.pytest_plugin", "-q")
    finally:
        del os.environ["GFFX_DASHBOARD"]
    result.assert_outcomes(passed=1, failed=1)
    runs = server.state.runs()
    run = next(r for r in runs if r != "test-run")
    tree = server.state.snapshot(run)["paths"]
    passed = next(v for k, v in tree.items() if k.startswith("tests/") and k.endswith("test_pass"))
    failed = next(v for k, v in tree.items() if k.startswith("tests/") and k.endswith("test_fail"))
    assert passed["latest"]["value"]["outcome"] == "passed"
    assert failed["latest"]["value"]["outcome"] == "failed"
    assert any(k.endswith("test_pass/evidence/metric") for k in tree)


def test_db09_no_address_means_every_call_is_a_no_op(monkeypatch):
    monkeypatch.delenv("GFFX_DASHBOARD", raising=False)
    monkeypatch.setattr(gd, "tailscale_address", lambda: None)
    dash = gd.connect()
    assert dash.connected is False
    t0 = time.perf_counter()
    dash.mesh("x", torch.zeros(3, 3), torch.zeros(1, 3, dtype=torch.int32))
    dash.scalar("y", 1.0)
    c = dash.control("controls/lr", 1e-3)
    assert c.value == 1e-3
    dash.close()
    assert (time.perf_counter() - t0) < 0.05


def test_db10_a_full_queue_costs_the_sender_under_the_budget(server, tmp_path):
    verts = torch.rand(5023, 3)
    faces = torch.randint(0, 5023, (9976, 3), dtype=torch.int32)

    def loop(dash, n):
        t0 = time.perf_counter()
        for i in range(n):
            if dash is not None:
                dash.mesh("fit/head", verts, faces, step=i)
            _ = verts.sum()
        return time.perf_counter() - t0

    baseline = min(loop(None, 60) for _ in range(3))
    dash = gd.connect(server.address, run="budget", name="loop", queue_bound=8)
    try:
        with_server = min(loop(dash, 60) for _ in range(3))
        assert dash.dropped >= 0
    finally:
        dash.close()
    assert with_server <= baseline * 1.01 + 0.02  # 1 per cent, plus a constant for timer noise


@pytest.mark.skipif(shutil.which("chrome") is None and shutil.which("chromium") is None
                    and shutil.which("msedge") is None, reason="no headless browser with WebGPU on PATH")
def test_db11_the_viewport_matches_a_gffx_cpu_render_to_a_display_tolerance(server, client):
    import gffx.torch as gt
    verts = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    client.mesh("view/tri", verts, faces, colors=colors, step=0)
    ndc = torch.cat([verts[:, :2], torch.zeros(3, 1)], dim=-1)
    fi, bary, _, _ = gt.render.rasterize(ndc_vertices=ndc, faces=faces, image_height=128, image_width=128,
                                         faces_per_pixel=1, blur_radius_px=0.0, cull_mode=gt.render.CULL_NONE)
    reference = gt.render.interpolate(fi, bary, colors[faces])[0, :, :, 0, :]
    mask = fi[0, :, :, 0] >= 0
    frame = gd.headless.render_viewport(server.address, run="test-run", path="view/tri",
                                        width=128, height=128, camera="identity")
    got_mask = frame[..., 3] > 0
    iou = (got_mask & mask).sum().item() / max(1, (got_mask | mask).sum().item())
    assert iou > 0.98
    diff = (frame[..., :3][mask] - reference[mask]).abs().mean().item()
    assert diff < 0.05


def test_db12_mixed_kinds_on_one_path_are_rejected_and_the_connection_survives(server, client):
    client.scalar("k/x", 1.0, step=0)
    with pytest.raises(gd.ProtocolError):
        client.text("k/x", "no", step=1)
    client.scalar("k/x", 2.0, step=1)
    assert server.state.get("k/x", step=1)["value"] == 2.0
