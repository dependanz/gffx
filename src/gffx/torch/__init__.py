"""Explicit PyTorch adapter namespace.

Phase 1 Step 6 owns only the import semantics enforced below: a missing framework must fail this
adapter import alone, with an actionable message, and must never affect `import gffx`. The adapter
itself is Step 8, so nothing here imports or uses PyTorch yet -- presence is checked through the
import system rather than by importing the framework.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise ImportError(
        "gffx.torch requires PyTorch, which is not installed in this environment.\n"
        "Install a build appropriate for your platform, for example:\n"
        "    pip install torch\n"
        "See https://pytorch.org/get-started/locally/ for CPU and CUDA specific commands.\n"
        "The gffx base package does not depend on PyTorch: `import gffx` and "
        "`gffx.capabilities()` keep working without it."
    )

__all__: list = []
