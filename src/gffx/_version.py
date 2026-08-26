"""Installed-version access for the dependency-free base package.

Uses only the standard library. The authority is the PEP 621 metadata written by the build; the
static fallback below covers running directly from a source checkout, where no installed
distribution exists to query. `tests/python/test_base_import.py` asserts the fallback still equals
the `pyproject.toml` version, so the two cannot drift silently.
"""

from __future__ import annotations

__all__ = ["SOURCE_TREE_FALLBACK_VERSION", "version"]

SOURCE_TREE_FALLBACK_VERSION = "0.2.0.dev0"


def version() -> str:
    """Return the installed distribution version, or the source-tree fallback."""
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _distribution_version
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.10+
        return SOURCE_TREE_FALLBACK_VERSION
    try:
        return _distribution_version("gffx")
    except PackageNotFoundError:
        return SOURCE_TREE_FALLBACK_VERSION
