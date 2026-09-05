"""The pytest plugin of DASHBOARD_CONTRACT_V0_1.md section 6.

Inert unless ``GFFX_DASHBOARD`` names a server. Then a run opens for the session, every test's
outcome lands at ``tests/<nodeid>`` as it is reported, and the ``dashboard`` fixture lets a test
attach evidence under ``tests/<nodeid>/evidence/<name>``. The plugin adds no assertion and changes
no outcome.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import pytest

_SEGMENT_BAD = re.compile(r"[^A-Za-z0-9_.-]+")


def _path_of(nodeid: str) -> str:
    """A nodeid as a state path: ``tests/python/test_x.py::TestA::test_b[1]`` → ``tests/python/test_x.py/TestA/test_b_1_``."""
    parts = nodeid.replace("\\", "/").split("::")
    segments = []
    for part in parts:
        for piece in part.split("/"):
            cleaned = _SEGMENT_BAD.sub("_", piece)
            if cleaned:
                segments.append(cleaned)
    return "tests/" + "/".join(segments)


class _Evidence:
    """What the ``dashboard`` fixture hands a test: sends scoped beneath its evidence path."""

    def __init__(self, dashboard: Any, base: str) -> None:
        self._dashboard = dashboard
        self._base = base
        self.connected = dashboard is not None and dashboard.connected

    def _at(self, name: str) -> str:
        return "%s/evidence/%s" % (self._base, _SEGMENT_BAD.sub("_", name))

    def scalar(self, name: str, value: Any, *, step: int | None = None) -> None:
        if self._dashboard is not None:
            self._dashboard.scalar(self._at(name), value, step=step)

    def image(self, name: str, image: Any, *, step: int | None = None) -> None:
        if self._dashboard is not None:
            self._dashboard.image(self._at(name), image, step=step)

    def mesh(self, name: str, vertices: Any, faces: Any, *, colors: Any = None, normals: Any = None, step: int | None = None) -> None:
        if self._dashboard is not None:
            self._dashboard.mesh(self._at(name), vertices, faces, colors=colors, normals=normals, step=step)

    def points(self, name: str, positions: Any, *, colors: Any = None, radii: Any = None, step: int | None = None) -> None:
        if self._dashboard is not None:
            self._dashboard.points(self._at(name), positions, colors=colors, radii=radii, step=step)

    def text(self, name: str, text: str, *, markdown: bool = False, step: int | None = None) -> None:
        if self._dashboard is not None:
            self._dashboard.text(self._at(name), text, markdown=markdown, step=step)

    def record(self, name: str, record: dict[str, Any], *, step: int | None = None) -> None:
        if self._dashboard is not None:
            self._dashboard.record(self._at(name), record, step=step)


class _Session:
    def __init__(self) -> None:
        self.dashboard: Any = None
        self.index = 0

    def open(self, config: pytest.Config) -> None:
        address = os.environ.get("GFFX_DASHBOARD")
        if not address:
            return
        from . import connect
        run = "pytest-" + time.strftime("%Y%m%d-%H%M%S") + "-%d" % os.getpid()
        name = "pytest " + " ".join(str(item) for item in config.invocation_params.args)
        try:
            self.dashboard = connect(address, run=run, name=name.strip(), wait_acks=True)
        except (ConnectionError, OSError) as error:  # a missing server never fails a test session
            config.issue_config_time_warning(pytest.PytestWarning("gffx.dashboard plugin: %s" % (error,)), stacklevel=2)
            self.dashboard = None
            return
        self.dashboard.text("tests", "session started", step=0)

    def close(self) -> None:
        if self.dashboard is not None:
            self.dashboard.close()
            self.dashboard = None


_session = _Session()


def pytest_configure(config: pytest.Config) -> None:
    _session.open(config)


def pytest_unconfigure(config: pytest.Config) -> None:
    _session.close()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    dashboard = _session.dashboard
    if dashboard is None:
        return
    # One value per test: the call phase decides, except a setup or teardown failure, which reports itself.
    if report.when == "call" or (report.when != "call" and report.outcome != "passed"):
        _session.index += 1
        message = None
        if report.failed and report.longrepr is not None:
            message = str(report.longrepr)[-2000:]
        elif report.skipped and report.longrepr is not None:
            message = str(report.longrepr[-1] if isinstance(report.longrepr, tuple) else report.longrepr)[-500:]
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        value = {"outcome": outcome, "duration": float(report.duration), "when": report.when, "message": message}
        dashboard.record(_path_of(report.nodeid), value, step=_session.index)


@pytest.fixture
def dashboard(request: pytest.FixtureRequest) -> _Evidence:
    """Evidence sends for the running test, landing under ``tests/<nodeid>/evidence/``; inert without a server."""
    return _Evidence(_session.dashboard, _path_of(request.node.nodeid))
