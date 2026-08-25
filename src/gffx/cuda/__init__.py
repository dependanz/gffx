"""Explicit optional CUDA backend diagnostics.

Importing this namespace is inert. Calling :func:`capabilities` is an explicit setup/diagnostic
action that may load the isolated plugin and NVIDIA driver, block, and enumerate devices. It is
not safe for a real-time frame path and advertises no graphics or geometry operation in Phase 1.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["capabilities"]


def capabilities(*, include_sensitive: bool = False) -> Dict[str, Any]:
    """Return the verbose runtime snapshot, redacting paths and stable identifiers by default."""
    from .._capabilities import full_capabilities

    return full_capabilities(include_sensitive=include_sensitive)
