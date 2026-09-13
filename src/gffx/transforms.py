"""Lazy transform namespace with no duplicate arithmetic."""


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .torch import transforms as backend
    return getattr(backend, name)


def __dir__():
    from .torch import transforms as backend
    return sorted(backend.__all__)
