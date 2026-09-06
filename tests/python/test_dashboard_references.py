"""Stable page references and explicit run provenance, independent of messaging agents."""
import pytest
from gffx import dashboard as gd


def test_page_identity_survives_save_and_restart(tmp_path):
    ids = []
    for restart in range(2):
        server = gd.Server(tmp_path, allow_loopback_for_tests=True, port=0).start()
        page = None
        try:
            page = gd.PageConnection(server.address, run="test")
            snap = page.open()
            ids.append(snap["run_id"])
            page.layout_save({"name": "camera projection", "panels": []})
            saved = page.layout_load("camera projection")
            ids.append(saved["page_id"])
            page.layout_save({**saved, "columns": 8})
            assert page.layout_load("camera projection")["page_id"] == saved["page_id"]
        finally:
            if page: page.close()
            server.stop()
    assert ids[:2] == ids[2:]


def test_page_ids_distinguish_runs_and_layouts(tmp_path):
    with gd.Server(tmp_path, allow_loopback_for_tests=True, port=0) as server:
        found = set()
        for run in ("one", "two"):
            page = gd.PageConnection(server.address, run=run)
            try:
                snap = page.open()
                found.add(snap["pages"][0]["page_id"])
                for name in ("camera", "shading"):
                    page.layout_save({"name": name, "panels": []})
                    found.add(page.layout_load(name)["page_id"])
            finally:
                page.close()
        assert len(found) == 6


def test_run_provenance_is_reported_and_cannot_be_rewritten(tmp_path):
    with gd.Server(tmp_path, allow_loopback_for_tests=True, port=0) as server:
        provenance = {"revision": "a" * 40, "dirty": True, "artifact_sha256": "b" * 64}
        client = gd.connect(server.address, run="test", provenance=provenance, wait_acks=True)
        client.close()
        page = gd.PageConnection(server.address, run="test")
        try:
            assert page.open()["provenance"] == provenance
        finally:
            page.close()
        with pytest.raises(ConnectionError, match="provenance"):
            gd.connect(server.address, run="test", provenance={"revision": "c" * 40})
    fresh = gd.Server(tmp_path, allow_loopback_for_tests=True, port=0).start()
    page = None
    try:
        page = gd.PageConnection(fresh.address, run="test")
        assert page.open()["provenance"] == provenance
    finally:
        if page: page.close()
        fresh.stop()


def test_unknown_provenance_is_not_inferred_from_server(tmp_path):
    with gd.Server(tmp_path, allow_loopback_for_tests=True, port=0) as server:
        client = gd.connect(server.address, run="unknown")
        client.close()
        page = gd.PageConnection(server.address, run="unknown")
        try:
            assert page.open()["provenance"] == {}
        finally:
            page.close()


def test_invalid_or_conflicting_provenance_does_not_poison_log(tmp_path):
    with gd.Server(tmp_path, allow_loopback_for_tests=True, port=0) as server:
        client = gd.connect(server.address, run="test", provenance={"revision": "a" * 40})
        client.close()
        path = server.run_dir("test") / "log.bin"
        before = path.read_bytes()
        with pytest.raises(gd.ProtocolError, match="provenance"):
            gd.connect(server.address, run="test", provenance={"revision": "not a hash"})
        with pytest.raises(ConnectionError, match="provenance"):
            gd.connect(server.address, run="test", provenance={"revision": "c" * 40})
        assert path.read_bytes() == before
