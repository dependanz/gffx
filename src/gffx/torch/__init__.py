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
    # The underlying loader error is included rather than only chained. When this failed on every
    # hosted lane on 2026-09-03 the real cause was a missing runtime search path, and the Linux
    # lanes were diagnosable only because pytest happened to print the chained cause while the
    # Windows lane did not. Advice to rebuild is also actively wrong in that case: the adapter was
    # built, and could not find the core beside it.
    raise ImportError(
        "gffx found a supported PyTorch %s installation, but its private Stable-ABI adapter "
        "binary could not be loaded: %s\n"
        "Install a PyTorch-ready gffx wheel or rebuild gffx with GFFX_BUILD_PYTORCH=ON. If the "
        "adapter is present but a library beside it could not be found, the installation's "
        "runtime search path is wrong rather than the build. The base package remains usable "
        "with `import gffx`." % (_torch_framework.__version__, error)
    ) from error

if not hasattr(_torch_framework.ops.gffx_internal, "_foundation_probe"):
    raise ImportError(
        "gffx._torch loaded but did not register its private Stable-ABI foundation schema. "
        "Reinstall gffx; the base package remains usable with `import gffx`."
    )

from . import io, mesh, points, render, stream, transforms  # noqa: E402

__all__: list[str] = ["io", "mesh", "points", "render", "stream", "transforms"]
