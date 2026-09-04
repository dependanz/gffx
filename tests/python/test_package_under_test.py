"""Asserts the suite is testing the package it is supposed to be testing.

Every other test here assumes `import gffx` reaches the intended package. When that assumption
broke, the symptom was 36 import errors across `tests/pytorch` that took two continuous-integration
rounds to read, because the failures described a missing submodule rather than the wrong package.
This test converts that into one named failure, checked before the suites that depend on it.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def _installed_distribution_exists() -> Path | None:
    """Return an installed `gffx` package directory, ignoring the source tree.

    `sys.path` is scanned directly rather than asked through `importlib`, because by the time this
    runs the source tree may already sit at the front and would answer for any installed copy.
    """
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry).resolve()
        if candidate == SRC_DIR.resolve():
            continue
        if (candidate / "gffx" / "__init__.py").is_file():
            return candidate / "gffx"
    return None


def test_the_suite_imports_the_installed_distribution_when_one_exists() -> None:
    if os.environ.get("GFFX_TEST_SOURCE_TREE") == "1":
        pytest.skip("GFFX_TEST_SOURCE_TREE=1 selects the source tree deliberately")

    installed = _installed_distribution_exists()
    if installed is None:
        pytest.skip("no installed gffx on sys.path; the source tree is the only candidate")

    import gffx

    origin = Path(gffx.__file__).resolve()
    assert SRC_DIR not in origin.parents, (
        "the suite imported the source tree while an installed distribution exists at "
        f"{installed}. The source tree holds no compiled extensions unless it was built in "
        "place, so every test needing gffx._core or gffx._torch will fail with a missing "
        "submodule rather than reporting this. Set GFFX_TEST_SOURCE_TREE=1 to test the source "
        "tree on purpose."
    )


def test_the_selected_package_carries_its_compiled_core() -> None:
    """A package without its extension is a package that cannot answer for the library.

    Kept separate from the selection check above so the two failures stay distinguishable: the
    wrong package was chosen, or the right one was never built.
    """
    import gffx

    package_dir = Path(gffx.__file__).resolve().parent
    extensions = sorted(
        path.name
        for path in package_dir.iterdir()
        if path.suffix in {".pyd", ".so", ".dylib"} or ".so." in path.name
    )
    assert any(name.startswith("_core") for name in extensions), (
        f"{package_dir} contains no compiled _core extension; found {extensions or 'none'}. "
        "An installed distribution should carry one, and a source tree only carries one after an "
        "in-place build."
    )
