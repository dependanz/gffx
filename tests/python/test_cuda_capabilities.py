"""Python rendering tests for the explicit CUDA capability surface."""

from __future__ import annotations


def test_full_capability_records_receive_stable_names(monkeypatch):
    from gffx import _capabilities

    class FakeCore:
        @staticmethod
        def runtime_capabilities(*, include_sensitive=False):
            assert include_sensitive is False
            return {
                "query_flags": 1,
                "result_flags": 3,
                "include_sensitive": False,
                "records": [
                    {
                        "category": 4,
                        "subject_index": 0,
                        "key": 13,
                        "value_type": 3,
                        "flags": 0,
                        "sensitive": False,
                        "value": "loaded",
                    },
                    {
                        "category": 6,
                        "subject_index": 1,
                        "key": 23,
                        "value_type": 3,
                        "flags": 0,
                        "sensitive": False,
                        "value": "Synthetic GPU",
                    },
                ],
            }

    monkeypatch.setattr(_capabilities, "_core", FakeCore())
    monkeypatch.setattr(_capabilities, "_core_attempted", True)
    report = _capabilities.full_capabilities()
    assert report["gpu_probed"] is True
    assert report["status"] == "loaded"
    assert report["result_flags"] == {
        "static": True,
        "runtime_probed": True,
        "optional_provider_absent": False,
        "partial_failure": False,
        "raw": 3,
    }
    assert report["records"][0]["category_name"] == "backend"
    assert report["records"][0]["key_name"] == "cuda_provider_status"
    assert report["records"][1]["category_name"] == "device"
    assert report["records"][1]["key_name"] == "cuda_device_name"


def test_full_capabilities_reports_an_unavailable_native_core(monkeypatch):
    from gffx import _capabilities

    monkeypatch.setattr(_capabilities, "_core", None)
    monkeypatch.setattr(_capabilities, "_core_attempted", True)
    monkeypatch.setattr(_capabilities, "_core_error", "synthetic missing core")
    report = _capabilities.full_capabilities()
    assert report["probe_attempted"] is True
    assert report["gpu_probed"] is False
    assert report["status"] == "native core unavailable"
    assert report["detail"] == "synthetic missing core"
    assert report["records"] == []


def test_public_cuda_namespace_delegates_only_on_call(monkeypatch):
    import gffx.cuda
    from gffx import _capabilities

    sentinel = {"status": "sentinel"}

    def fake_full_capabilities(*, include_sensitive=False):
        assert include_sensitive is True
        return sentinel

    monkeypatch.setattr(_capabilities, "full_capabilities", fake_full_capabilities)
    assert gffx.cuda.capabilities(include_sensitive=True) is sentinel
