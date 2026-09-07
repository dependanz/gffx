"""Dependency-free frontend behavior checks against the selected package's static assets.

Node is optional test tooling, never a dashboard runtime dependency. These tests model pointer
events, not browser gesture arbitration or a WebGPU framebuffer; real-device acceptance remains
separate. conftest's installed-wheel/source selection applies to the assets too.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import gffx.dashboard


def test_dashboard_touch_contracts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("dashboard pointer contracts need Node test tooling; not mobile acceptance")
    version = subprocess.check_output([node, "--version"], text=True, timeout=10).strip()
    if int(version.lstrip("v").split(".")[0]) < 22:
        pytest.skip("dashboard pointer contracts need Node 22+ test tooling")
    static = Path(gffx.dashboard.__file__).parent / "static"
    script = Path(__file__).resolve().parents[1] / "dashboard" / "touch.test.mjs"
    result = subprocess.run(
        [node, "--experimental-default-type=module", "--test", str(script)],
        env={**os.environ, "GFFX_DASHBOARD_STATIC": str(static)},
        text=True, encoding='utf-8', capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
