"""Phase 1 Step 8 contracts for the lazy PyTorch Stable-ABI adapter."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import REPO_ROOT, SRC_DIR, USING_SOURCE_TREE


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run a snippet in a clean interpreter against the in-tree package."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    # The subprocess must resolve the same gffx the suite selected. Forcing the source tree onto
    # PYTHONPATH unconditionally is how these checks kept testing the working tree while an
    # installed wheel sat unexamined in site-packages; when the source tree is the selection, it
    # still goes on the path so an in-place build works with no install.
    if USING_SOURCE_TREE:
        prefix = os.fspath(SRC_DIR)
        env["PYTHONPATH"] = prefix + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )


def run_installed_python(code: str) -> subprocess.CompletedProcess:
    """Run outside the checkout so an installed wheel, never the source tree, is exercised."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="gffx-installed-wheel-") as temporary_directory:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=temporary_directory,
        )


def release_tuple(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    assert match is not None, "unrecognized PyTorch version: " + version
    return int(match.group(1)), int(match.group(2))


def test_import_gffx_exposes_a_lazy_torch_namespace_without_loading_it():
    code = (
        "import sys\n"
        "import gffx\n"
        "print('torch' in dir(gffx))\n"
        "print(','.join(str(name in sys.modules) for name in "
        "('torch', 'gffx.torch', 'gffx._torch')))\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["True", "False,False,False"]


def test_below_floor_pytorch_fails_only_adapter_access_with_detected_version():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")

    import torch

    if release_tuple(torch.__version__) >= (2, 10):
        pytest.skip("this is the explicit below-floor compatibility lane")

    code = (
        "import gffx\n"
        "print('BASE:' + gffx.__version__)\n"
        "try:\n"
        "    gffx.torch\n"
        "except ImportError as error:\n"
        "    print('IMPORTERROR:' + str(error).replace(chr(10), ' | '))\n"
        "else:\n"
        "    print('NO_ERROR')\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    output = result.stdout.splitlines()
    assert output[0].startswith("BASE:")
    assert output[1].startswith("IMPORTERROR:")
    assert "requires PyTorch 2.10 or newer" in output[1]
    assert torch.__version__ in output[1]
    assert "base package" in output[1]


def test_registration_source_uses_only_the_libtorch_stable_api():
    source = (Path(REPO_ROOT) / "adapters" / "pytorch" / "register.cpp").read_text(
        encoding="utf-8"
    )
    assert "#include <Python.h>" in source
    assert "#include <torch/csrc/stable/library.h>" in source
    assert "STABLE_TORCH_LIBRARY(gffx_internal" in source
    assert "PyInit__torch" in source
    for forbidden in ("<torch/torch.h>", "<ATen/", "pybind11"):
        assert forbidden not in source


def test_adapter_build_pins_the_python_and_libtorch_abi_floors():
    cmake = (Path(REPO_ROOT) / "cmake" / "GffxPyTorch.cmake").read_text(encoding="utf-8")
    assert "USE_SABI 3.10" in cmake
    assert "Py_LIMITED_API=0x030A0000" in cmake
    assert "TORCH_TARGET_VERSION=0x020a000000000000" in cmake
    assert "CXX_STANDARD 17" in cmake


def test_supported_pytorch_loads_the_private_registration_probe():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")

    import torch

    if release_tuple(torch.__version__) < (2, 10):
        pytest.skip("PyTorch Stable ABI begins at 2.10")

    code = (
        "import importlib.util\n"
        "import sys\n"
        "if importlib.util.find_spec('gffx') is None:\n"
        "    print('SKIP:no installed gffx wheel')\n"
        "    raise SystemExit(0)\n"
        "import gffx\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'gffx._torch' not in sys.modules\n"
        "adapter = gffx.torch\n"
        "import torch\n"
        "assert adapter is gffx.torch\n"
        "assert sys.modules['torch'] is torch\n"
        "assert 'gffx._torch' in sys.modules\n"
        "assert hasattr(torch.ops.gffx_internal, '_foundation_probe')\n"
        "print('OK:' + torch.__version__)\n"
    )
    result = run_installed_python(code)
    if result.stdout.startswith("SKIP:"):
        pytest.skip("a PyTorch-ready GFFX wheel is not installed in this environment")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "OK:" + torch.__version__
