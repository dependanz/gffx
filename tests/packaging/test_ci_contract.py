from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _load_ci_matrix():
    module_path = TOOLS / "ci_matrix.py"
    spec = importlib.util.spec_from_file_location("gffx_ci_matrix", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_pins_are_exact_and_immutable() -> None:
    matrix = _load_ci_matrix()
    assert matrix.ACTION_PINS == {
        "actions/checkout": ("v7.0.1", "3d3c42e5aac5ba805825da76410c181273ba90b1"),
        "actions/setup-python": ("v7.0.0", "5fda3b95a4ea91299a34e894583c3862153e4b97"),
        "actions/upload-artifact": ("v7.0.1", "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"),
        "actions/download-artifact": ("v8.0.0", "70fc10c6e5e1ce46ad2ea6f2b72d43f7d47b13c3"),
        "pypa/cibuildwheel": ("v4.2.0", "1828c10ab37f080699c7b81cea34097c684a7074"),
    }


def test_pull_request_matrix_is_the_seven_adopted_lanes() -> None:
    matrix = _load_ci_matrix()
    actual = matrix.pull_request_matrix()
    assert len(actual) == 7
    assert [entry["name"] for entry in actual] == [
        "linux-x64-floor-floor", "linux-x64-floor-current",
        "linux-x64-current-floor", "linux-x64-current-current",
        "windows-x64-current-current", "linux-arm64-current-current",
        "macos-arm64-current-current",
    ]
    assert [(entry["python"], entry["pytorch"]) for entry in actual] == [
        ("3.10", "2.10.0"), ("3.10", "2.13.0"),
        ("3.14", "2.10.0"), ("3.14", "2.13.0"),
        ("3.14", "2.13.0"), ("3.14", "2.13.0"),
        ("3.14", "2.13.0"),
    ]


def test_nightly_matrix_is_36_blocking_lanes_with_iso_week_rotation() -> None:
    matrix = _load_ci_matrix()
    odd = matrix.nightly_matrix(iso_week=35)
    even = matrix.nightly_matrix(iso_week=36)
    for actual in (odd, even):
        assert len(actual) == 36
        assert all(entry["blocking"] for entry in actual)
        assert {entry["platform"] for entry in actual} == set(matrix.PLATFORMS)
        assert len({entry["name"] for entry in actual}) == 36
    assert {entry["pytorch"] for entry in odd} == {"2.10.0", "2.11.0", "2.13.0"}
    assert {entry["pytorch"] for entry in even} == {"2.10.0", "2.12.1", "2.13.0"}
    assert sum(entry["lane"] == "current" for entry in odd) == 20
    assert sum(entry["lane"] == "floor" for entry in odd) == 8
    assert sum(entry["lane"] == "intermediate" for entry in odd) == 8


def test_release_candidate_matrix_builds_four_and_tests_eighty() -> None:
    matrix = _load_ci_matrix()
    builds = matrix.release_candidate_build_matrix()
    tests = matrix.release_candidate_test_matrix()
    assert len(builds) == 4
    assert {entry["platform"] for entry in builds} == set(matrix.PLATFORMS)
    assert all(entry["python"] == "3.10" for entry in builds)
    assert all(entry["pytorch"] == "2.10.0" for entry in builds)
    assert sum(bool(entry["build_sdist"]) for entry in builds) == 1
    assert len(tests) == 80
    assert len({(e["platform"], e["python"], e["pytorch"]) for e in tests}) == 80


def test_workflows_are_exactly_the_three_cadences_and_validate() -> None:
    matrix = _load_ci_matrix()
    workflows = ROOT / ".github" / "workflows"
    assert matrix.validate_workflows(ROOT) == []
    assert not (workflows / "wheels.yml").exists()
    assert {path.name for path in workflows.glob("*.yml")} == {
        "package-foundation-pr.yml", "package-foundation-nightly.yml",
        "package-foundation-rc.yml",
    }


def test_workflows_are_least_privilege_pinned_and_scope_truthful() -> None:
    workflows = ROOT / ".github" / "workflows"
    joined = "\n".join(
        path.read_text(encoding="utf-8") for path in workflows.glob("*.yml")
    ).lower()
    assert "tools/verify_foundation.py" in joined
    assert "tools/ci_smoke.py" in joined
    assert "package-foundation-required" in joined
    assert "permissions:\n  contents: read" in joined
    assert "persist-credentials: false" in joined
    for forbidden in (
        "pull_request_target", "@main", "@master", "@latest",
        "ubuntu-latest", "windows-latest", "macos-latest", "twine", "pypi",
        "id-token: write", "contents: write", "torch.compile", "torch.export",
        "opcheck", "face_geometry",
    ):
        assert forbidden not in joined


def test_rc_installs_downloaded_wheels_without_rebuilding_source() -> None:
    rc = (ROOT / ".github" / "workflows" / "package-foundation-rc.yml").read_text(
        encoding="utf-8"
    )
    assert "needs: build-artifacts" in rc
    assert "actions/download-artifact@" in rc
    assert "tools/ci_smoke.py installed" in rc
    assert "tools/ci_smoke.py uninstalled" in rc
    assert "python -m pip uninstall --yes gffx" in rc
    assert "python -m pip wheel ." not in rc.split("artifact-tests:", maxsplit=1)[1]


def test_ci_smoke_exposes_installed_and_uninstalled_modes() -> None:
    script = (TOOLS / "ci_smoke.py").read_text(encoding="utf-8")
    assert 'add_parser("installed"' in script
    assert 'add_parser("uninstalled"' in script
    assert "gpu_probed" in script
    assert "_foundation_probe" in script
