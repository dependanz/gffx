"""Explicit PyTorch adapter namespace.

Accessing this namespace is the opt-in boundary: it imports PyTorch, enforces the supported floor,
loads the private Stable-ABI registration module, and verifies its internal foundation schema.
Plain `import gffx` does none of those things.
"""

from __future__ import annotations

import importlib.util
import re

if importlib.util.find_spec("torch") is None:
    raise ImportError(
        "gffx.torch requires PyTorch, which is not installed in this environment.\n"
        "Install a build appropriate for your platform, for example:\n"
        "    pip install torch\n"
        "See https://pytorch.org/get-started/locally/ for CPU and CUDA specific commands.\n"
        "The gffx base package does not depend on PyTorch: `import gffx` and "
        "`gffx.capabilities()` keep working without it."
    )

import torch as _torch_framework


def _release_tuple(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise ImportError(
            "gffx.torch could not interpret the installed PyTorch version %r. "
            "The gffx base package remains usable with `import gffx`." % version
        )
    return int(match.group(1)), int(match.group(2))


if _release_tuple(_torch_framework.__version__) < (2, 10):
    raise ImportError(
        "gffx.torch requires PyTorch 2.10 or newer; detected PyTorch %s. "
        "Upgrade PyTorch to activate the adapter. The gffx base package remains usable with "
        "`import gffx`." % _torch_framework.__version__
    )

try:
    importlib.import_module("gffx._torch")
except ImportError as error:
    raise ImportError(
        "gffx found a supported PyTorch %s installation, but its private Stable-ABI adapter "
        "binary could not be loaded. Install a PyTorch-ready gffx wheel or rebuild gffx with "
        "GFFX_BUILD_PYTORCH=ON. The base package remains usable with `import gffx`."
        % _torch_framework.__version__
    ) from error

if not hasattr(_torch_framework.ops.gffx_internal, "_foundation_probe"):
    raise ImportError(
        "gffx._torch loaded but did not register its private Stable-ABI foundation schema. "
        "Reinstall gffx; the base package remains usable with `import gffx`."
    )

from . import mesh, stream  # noqa: E402  (the adapter gate above must run first)

__all__: list[str] = ["mesh", "stream"]
