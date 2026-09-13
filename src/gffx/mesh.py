"""Lazy mesh namespace; all validation and mathematics belong to the adapter."""


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .torch import mesh as backend
    return getattr(backend, name)


def __dir__():
    from .torch import mesh as backend
    return sorted(backend.__all__)
