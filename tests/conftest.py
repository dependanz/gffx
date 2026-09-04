"""Selects which gffx the suite imports, and says so out loud.

Two gffx packages can exist at once: the source tree at `src/gffx`, and an installed distribution
in site-packages. Only one wins, because Python resolves a package to a single directory and then
looks for every submodule inside it alone. There is no fallback: a `gffx` resolved to the source
tree cannot see `gffx._core` or `gffx._torch` in site-packages, however complete that installation
is.

This file used to place the source tree first unconditionally, which pairs with the supported
in-place build workflow that `.gitignore` provides for (`src/gffx/*.pyd`, `src/gffx/*.dll`): build
the extensions beside their sources and import them without installing anything. That is a real
convenience and it is preserved.

What it also did was override CI. Every hosted lane builds a wheel, installs it, and intends to
test it; the insertion silently redirected the suite to a source tree that had never been built, so
`tests/pytorch` failed with `No module named 'gffx._torch'` while the artifact under test sat
untouched in site-packages. The same override made a local run pass for the wrong reason, because
hand-copied binaries in `src/gffx` accidentally reproduce an in-place build.

So the default is now the installed distribution when one exists, and the mode that can mislead is
opt-in rather than automatic. An environment variable that CI must remember to set would have
restored the same failure the first time somebody added a lane and forgot it.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

#: Set `GFFX_TEST_SOURCE_TREE=1` to test the working tree, which is what an in-place build wants.
FORCE_SOURCE_TREE = os.environ.get("GFFX_TEST_SOURCE_TREE") == "1"

#: Resolved before `sys.path` is touched, so it answers whether an installed gffx exists rather
#: than whether this file has already shadowed one.
INSTALLED_SPEC = importlib.util.find_spec("gffx")

USING_SOURCE_TREE = FORCE_SOURCE_TREE or INSTALLED_SPEC is None

if USING_SOURCE_TREE and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_report_header(config) -> str:
    """Name the package under test in every run's header.

    This defect survived weeks of green runs because nothing ever reported which of the two
    candidates had been imported; both look identical from the outside, and the wrong one fails
    only where a compiled submodule is needed. One line makes that ambiguity impossible to hold.
    """
    try:
        import gffx
    except Exception as error:  # noqa: BLE001 - a header must never break collection
        return f"gffx under test: not importable ({error})"

    origin = Path(gffx.__file__).resolve()
    if SRC_DIR in origin.parents:
        source = "source tree"
        if FORCE_SOURCE_TREE:
            source += ", forced by GFFX_TEST_SOURCE_TREE=1"
        elif INSTALLED_SPEC is None:
            source += ", no installed distribution found"
    else:
        source = "installed distribution"
    return f"gffx under test: {origin} ({source})"
