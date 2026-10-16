"""Portable differentiable graphics and mesh operations.

The base import is dependency-free and lazy. Importing `gffx` requires no autodiff framework,
array library, accelerator runtime, visualization, I/O, progress, or test package, and loads no
native library or backend provider. Package, ABI, and static capability state are reachable
through `gffx.capabilities()`, which loads the native core on demand and never probes a GPU.

Framework support lives behind explicit adapter imports such as `gffx.torch`. A missing framework
fails only that adapter import, with an actionable message; it never breaks `import gffx`.

Scene construction and rendering select the current adapter explicitly through
the lazy scene types or namespaces.
"""

from __future__ import annotations

__all__ = [
    "Scene", "Mesh", "Material", "Light", "Camera", "io", "__version__", "abi_version", "capabilities", "native_core_is_loaded", "cuda", "torch", "camera", "shade", "material", "scene", "image", "render", "mesh", "transforms", "skeleton"
]


def __getattr__(name: str):
    # PEP 562 module-level lazy attributes: nothing below is imported until first use.
    if name in ("Scene", "Mesh", "Material", "Light", "Camera"):
        from importlib import import_module
        module = import_module(".camera" if name == "Camera" else ".scene", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "io":
        from importlib import import_module
        module = import_module(".io", __name__)
        globals()[name] = module
        return module
    if name == "__version__":
        from ._version import version

        return version()
    if name in ("abi_version", "capabilities", "native_core_is_loaded"):
        from . import _capabilities

        return getattr(_capabilities, name)
    if name in ("camera", "shade", "material", "scene", "image", "render", "mesh", "transforms", "skeleton"):
        from importlib import import_module
        facade = import_module("." + name, __name__)
        globals()[name] = facade
        return facade
    if name == "cuda":
        # Importing the namespace is still inert. Only cuda.capabilities() explicitly loads a
        # plugin/driver and enumerates devices.
        from importlib import import_module

        backend = import_module(".cuda", __name__)
        globals()[name] = backend
        return backend
    if name == "torch":
        # Preserve `import gffx` as the one stable entry point while keeping framework activation
        # explicit: PyTorch and the native adapter load only when this attribute is first used.
        from importlib import import_module

        adapter = import_module(".torch", __name__)
        globals()[name] = adapter
        return adapter
    raise AttributeError("module 'gffx' has no attribute '%s'" % name)


def __dir__():
    return sorted(__all__)
