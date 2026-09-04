"""Phase 1 Step 6: base-package import semantics.

Every requirement of Step 6 is asserted here:

* `import gffx` pulls in no framework, array library, accelerator runtime, visualization, I/O,
  progress, or test package -- checked in a subprocess so an already-imported module cannot mask
  a real dependency. This environment has PyTorch installed, so the check is meaningful rather
  than vacuous.
* the base import is lazy and loads no native library.
* package/ABI/capability state is reportable without probing any GPU library.
* a missing framework fails only its explicit adapter import, with an actionable error.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, SRC_DIR, USING_SOURCE_TREE

# Frameworks, array libraries, accelerator runtimes, visualization, I/O, progress, and test
# packages that the dependency-free base import must never pull in.
FORBIDDEN_MODULES = (
    "torch",
    "numpy",
    "jax",
    "tensorflow",
    "cupy",
    "matplotlib",
    "PIL",
    "trimesh",
    "tqdm",
    "pytest",
    "scipy",
)


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run a snippet in a clean interpreter that can import the in-tree package."""
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


def test_source_fallback_version_matches_pyproject():
    """The static source-tree fallback must not drift from the PEP 621 authority."""
    from gffx._version import SOURCE_TREE_FALLBACK_VERSION

    pyproject = os.path.join(REPO_ROOT, "pyproject.toml")
    with open(pyproject, encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None, "pyproject.toml has no static version"
    assert SOURCE_TREE_FALLBACK_VERSION == match.group(1)


def test_version_is_a_nonempty_string():
    import gffx

    assert isinstance(gffx.__version__, str)
    assert gffx.__version__.strip() != ""


def test_base_import_pulls_in_no_forbidden_module():
    code = (
        "import sys\n"
        "import gffx\n"
        "forbidden = %r\n"
        "leaked = sorted(n for n in forbidden if n in sys.modules)\n"
        "print(','.join(leaked))\n"
    ) % (FORBIDDEN_MODULES,)
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", "import gffx leaked: " + result.stdout.strip()


def test_base_import_loads_no_native_library():
    """The private Limited-API extension must not be imported by the base import."""
    code = (
        "import sys\n"
        "import gffx\n"
        # Sample sys.modules immediately, before touching any attribute: reading
        # gffx.native_core_is_loaded itself legitimately imports the capability module.
        "core = 'gffx._core' in sys.modules\n"
        "caps = 'gffx._capabilities' in sys.modules\n"
        "assert gffx.native_core_is_loaded() is False\n"
        # The capability module is what would pull the extension in. In a source checkout the
        # compiled extension is absent, so checking only for 'gffx._core' cannot detect an eager
        # load; checking that the capability module itself stayed unimported can.
        "print('%s,%s' % (core, caps))\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False,False", "base import reached the native loading path"


def test_base_import_does_not_import_the_adapter_namespace():
    code = ("import sys\nimport gffx\n"
            "print('%s,%s' % ('gffx.cuda' in sys.modules, 'gffx.torch' in sys.modules))\n")
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False,False"


def test_capabilities_reports_state_without_probing_a_gpu():
    import gffx

    report = gffx.capabilities()
    for key in (
        "package_version",
        "python_version",
        "platform",
        "pointer_bits",
        "native_core",
        "backends",
        "gpu_probed",
    ):
        assert key in report, "capability report is missing " + key
    assert report["gpu_probed"] is False
    assert report["backends"]["cuda"] == "optional; explicit probe required"
    assert report["pointer_bits"] in (32, 64)
    assert isinstance(report["native_core"]["available"], bool)


def test_capabilities_tolerates_an_absent_native_core():
    """An unbuilt core is reported, never raised."""
    import gffx

    report = gffx.capabilities()
    native = report["native_core"]
    if not native["available"]:
        assert native["abi_version"] is None
        assert native["path"] is None
        assert native["limited_api"] is None
        assert "detail" in native and native["detail"].strip() != ""
    else:
        assert re.match(r"^\d+\.\d+$", native["abi_version"])
        assert os.path.isfile(native["path"])
        # The extension must be built against the CPython 3.10 stable ABI floor.
        assert native["limited_api"] == "0x030A0000"


def test_unknown_attribute_raises_attribute_error():
    import gffx

    with pytest.raises(AttributeError) as excinfo:
        gffx.definitely_not_a_real_attribute
    assert "gffx" in str(excinfo.value)


def test_dir_reports_the_public_surface():
    import gffx

    assert dir(gffx) == sorted(
        ["__version__", "abi_version", "capabilities", "native_core_is_loaded", "cuda", "torch"]
    )


def test_cuda_namespace_is_lazy_and_probe_is_explicit():
    code = (
        "import sys\n"
        "import gffx\n"
        "cuda = gffx.cuda\n"
        "print('%s,%s' % ('gffx._core' in sys.modules, 'gffx._capabilities' in sys.modules))\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False,False"


def test_missing_framework_fails_only_the_adapter_import_with_an_actionable_error():
    code = (
        "import importlib.util\n"
        "_real = importlib.util.find_spec\n"
        "def _fake(name, *args, **kwargs):\n"
        "    if name == 'torch':\n"
        "        return None\n"
        "    return _real(name, *args, **kwargs)\n"
        "importlib.util.find_spec = _fake\n"
        "import gffx\n"
        "assert gffx.capabilities()['gpu_probed'] is False\n"
        "try:\n"
        "    import gffx.torch\n"
        "except ImportError as error:\n"
        "    print('IMPORTERROR:' + str(error).replace(chr(10), ' | '))\n"
        "else:\n"
        "    print('NO_ERROR')\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    output = result.stdout.strip()
    assert output.startswith("IMPORTERROR:"), output
    # Actionable: names the missing framework, how to install it, and that the base package is fine.
    assert "PyTorch" in output
    assert "pip install torch" in output
    assert "base package" in output


def test_native_core_loads_only_when_capability_state_is_requested():
    """Laziness is observable: the extension is imported only after a real request."""
    code = (
        "import sys\n"
        "import gffx\n"
        "before = 'gffx._core' in sys.modules\n"
        "gffx.capabilities()\n"
        "after = 'gffx._core' in sys.modules\n"
        "print('%s,%s,%s' % (before, after, gffx.native_core_is_loaded()))\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    before, after, loaded = result.stdout.strip().split(",")
    assert before == "False"
    # In a source checkout the compiled extension is absent, so it may legitimately stay unloaded.
    # What must never happen is it loading before anything asked for it.
    assert after == loaded
