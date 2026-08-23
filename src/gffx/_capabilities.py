"""Base-package capability reporting.

Phase 1 Step 6/7 contract:

* importing this module imports no autodiff framework, array library, or accelerator runtime;
* nothing here loads a GPU driver, CUDA runtime, or backend provider;
* the private `gffx._core` CPython Limited-API loader is imported lazily, only when a caller
  actually asks for capability or ABI state, and its absence is reported rather than raised.

The report is deliberately a plain dict of built-in types so the base package needs no schema
dependency. Full runtime probing (which may load drivers and block) is a separate explicit action
that arrives with the CUDA plugin in a later phase; it is never reachable from `import gffx`.
"""

from __future__ import annotations

import struct
import sys
from typing import Any, Dict, Optional

__all__ = ["abi_version", "capabilities", "native_core_is_loaded"]

# Populated only by _load_core(). Remains None for the whole life of a process that never asks for
# capability state, which is what keeps `import gffx` free of native loading.
_core: Optional[Any] = None
_core_path: Optional[str] = None
_core_error: Optional[str] = None
_core_attempted = False


def _load_core() -> None:
    """Import the private Limited-API loader once. Never raises; records failure instead."""
    global _core, _core_path, _core_error, _core_attempted
    if _core_attempted:
        return
    _core_attempted = True

    try:
        from . import _core as loaded
    except ImportError as error:
        _core_error = (
            "the private gffx._core extension is not available in this installation, so "
            "capability values that require the native core are reported as unavailable: %s"
            % (error,)
        )
        return

    _core = loaded
    _core_path = getattr(loaded, "__file__", None)


def native_core_is_loaded() -> bool:
    """Report whether the native core has actually been loaded into this process."""
    return _core is not None


def abi_version() -> Optional[str]:
    """Return the native ABI version as 'major.minor', or None when the core is unavailable."""
    _load_core()
    if _core is None:
        return None
    encoded = int(_core.abi_version())
    return "%d.%d" % ((encoded >> 16) & 0xFFFF, encoded & 0xFFFF)


def capabilities() -> Dict[str, Any]:
    """Return static package, host, and ABI state without probing any GPU library."""
    from ._version import version

    _load_core()
    native: Dict[str, Any] = {
        "available": _core is not None,
        "path": _core_path,
        "abi_version": abi_version(),
        "limited_api": (
            "0x%08X" % int(_core.limited_api_version()) if _core is not None else None
        ),
    }
    if _core_error is not None:
        native["detail"] = _core_error

    return {
        "package_version": version(),
        "python_version": "%d.%d.%d" % sys.version_info[:3],
        "platform": sys.platform,
        "pointer_bits": struct.calcsize("P") * 8,
        "native_core": native,
        "backends": {
            "cpu": "available" if _core is not None else "native core not present",
            "cuda": "not built",
        },
        # Explicit and always false at this level: the base package never enumerates devices.
        "gpu_probed": False,
    }
