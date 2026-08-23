"""Portable differentiable graphics and mesh operations.

The base import is dependency-free and lazy. Importing `gffx` requires no autodiff framework,
array library, accelerator runtime, visualization, I/O, progress, or test package, and loads no
native library or backend provider. Package, ABI, and static capability state are reachable
through `gffx.capabilities()`, which loads the native core on demand and never probes a GPU.

Framework support lives behind explicit adapter imports such as `gffx.torch`. A missing framework
fails only that adapter import, with an actionable message; it never breaks `import gffx`.

No public graphics or geometry operation is advertised in Phase 1.
"""

from __future__ import annotations

__all__ = ["__version__", "abi_version", "capabilities", "native_core_is_loaded"]


def __getattr__(name: str):
    # PEP 562 module-level lazy attributes: nothing below is imported until first use.
    if name == "__version__":
        from ._version import version

        return version()
    if name in ("abi_version", "capabilities", "native_core_is_loaded"):
        from . import _capabilities

        return getattr(_capabilities, name)
    raise AttributeError("module 'gffx' has no attribute '%s'" % name)


def __dir__():
    return sorted(__all__)
