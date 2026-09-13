"""Exact installed-artifact identity remains visible alongside source revision."""
from types import SimpleNamespace

import pytest

from tools.testing_dashboard import pytest_plugin


@pytest.mark.parametrize('digest', ['a' * 64, 'bad-hash'])
def test_test_session_reports_only_valid_artifact_identity(monkeypatch, tmp_path, digest):
    events = []
    class Reporter:
        failed = False
        def __init__(self, *args):
            pass
        def send(self, event, **values):
            events.append((event, values))
            return True
    monkeypatch.setattr(pytest_plugin, 'Reporter', Reporter)
    monkeypatch.setattr(pytest_plugin.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setenv('GFFX_TEST_DASHBOARD', 'http://127.0.0.1:8791')
    monkeypatch.setenv('GFFX_TEST_ARTIFACT_SHA256', digest)
    session = pytest_plugin.Session(SimpleNamespace(rootpath=tmp_path, option=SimpleNamespace()))
    try:
        provenance = events[0][1]['provenance']
        if len(digest) == 64:
            assert provenance['artifact_sha256'] == digest
        else:
            assert 'artifact_sha256' not in provenance
    finally:
        session.close()
