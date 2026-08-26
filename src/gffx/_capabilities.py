"""Base-package capability reporting.

Base capability contract:

* importing this module imports no autodiff framework, array library, or accelerator runtime;
* nothing here loads a GPU driver, CUDA runtime, or backend provider;
* the private `gffx._core` CPython Limited-API loader is imported lazily, only when a caller
  actually asks for capability or ABI state, and its absence is reported rather than raised.

The report is deliberately a plain dict of built-in types so the base package needs no schema
dependency. Full runtime probing (which may load drivers and block) is available only through the
explicit `gffx.cuda.capabilities()` diagnostic; it is never reachable from `import gffx`.
"""

from __future__ import annotations

import struct
import sys
from typing import Any, Dict, Optional

__all__ = ["abi_version", "capabilities", "full_capabilities", "native_core_is_loaded"]

_CATEGORIES = {
    1: "build", 2: "host", 3: "cpu", 4: "backend", 5: "driver", 6: "device",
    7: "operation",
}
_KEY_NAMES = {
    1: "abi_version", 2: "package_version", 3: "pointer_bits", 4: "target_os",
    5: "target_arch", 6: "compiler", 7: "endianness", 8: "cpu_backend",
    9: "cuda_backend", 10: "dtype_mask", 11: "device_mask", 12: "operation_count",
    13: "cuda_provider_status", 14: "provider_status", 15: "cuda_plugin_path",
    16: "cuda_plugin_build_id", 17: "cuda_plugin_abi_version",
    18: "cuda_plugin_compatible", 19: "cuda_driver_status", 20: "cuda_driver_version",
    21: "cuda_device_count", 22: "cuda_toolkit_version", 23: "cuda_device_name",
    24: "cuda_device_uuid", 25: "cuda_device_pci_bus_id",
    26: "cuda_compute_capability_major", 27: "cuda_compute_capability_minor",
    28: "cuda_total_memory_bytes", 29: "cuda_multiprocessor_count", 30: "cuda_warp_size",
    31: "cuda_max_threads_per_block", 32: "cuda_max_block_dim_x",
    33: "cuda_max_block_dim_y", 34: "cuda_max_block_dim_z", 35: "cuda_max_grid_dim_x",
    36: "cuda_max_grid_dim_y", 37: "cuda_max_grid_dim_z",
    38: "cuda_shared_memory_per_block", 39: "cuda_registers_per_block",
    40: "cuda_clock_rate_khz", 41: "cuda_memory_clock_rate_khz",
    42: "cuda_memory_bus_width_bits", 43: "cuda_l2_cache_bytes",
    44: "cuda_max_threads_per_multiprocessor", 45: "cuda_unified_addressing",
    46: "cuda_managed_memory", 47: "cuda_concurrent_managed_access",
    48: "cuda_pageable_memory_access", 49: "cuda_cooperative_launch",
    50: "cuda_compute_mode", 51: "cuda_kernel_timeout", 52: "cuda_integrated",
    53: "cuda_can_map_host_memory", 54: "cuda_async_engine_count", 55: "cuda_ecc_enabled",
    56: "cuda_tcc_driver", 57: "cuda_compute_preemption",
    58: "cuda_max_shared_memory_per_block_optin",
    59: "cuda_max_blocks_per_multiprocessor", 60: "cuda_memory_pools_supported",
    61: "cuda_gpu_direct_rdma_supported",
}


def _named_record(record: Dict[str, Any]) -> Dict[str, Any]:
    named = dict(record)
    named["category_name"] = _CATEGORIES.get(record["category"], "unknown")
    named["key_name"] = _KEY_NAMES.get(record["key"], "unknown_%d" % record["key"])
    return named


def full_capabilities(*, include_sensitive: bool = False) -> Dict[str, Any]:
    """Explicitly load providers/drivers and return the verbose typed runtime snapshot."""
    _load_core()
    if _core is None:
        return {
            "gpu_probed": False,
            "probe_attempted": True,
            "status": "native core unavailable",
            "detail": _core_error,
            "result_flags": {},
            "records": [],
        }
    raw = _core.runtime_capabilities(include_sensitive=bool(include_sensitive))
    flags = int(raw["result_flags"])
    records = [_named_record(record) for record in raw["records"]]
    provider_status = next(
        (record["value"] for record in records if record["key"] == 13), "not reported"
    )
    return {
        "gpu_probed": bool(flags & 2),
        "probe_attempted": True,
        "status": provider_status,
        "include_sensitive": bool(raw["include_sensitive"]),
        "query_flags": int(raw["query_flags"]),
        "result_flags": {
            "static": bool(flags & 1),
            "runtime_probed": bool(flags & 2),
            "optional_provider_absent": bool(flags & 4),
            "partial_failure": bool(flags & 8),
            "raw": flags,
        },
        "records": records,
    }

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
            "cuda": "optional; explicit probe required",
        },
        # Explicit and always false at this level: the base package never enumerates devices.
        "gpu_probed": False,
    }
