"""Instance-owned setup-time extension bindings, never plugin discovery."""
import re
from collections.abc import Mapping
from .shade import _inputs


def parameters(values, owner):
    if not isinstance(values, Mapping):
        raise TypeError('parameters must be a mapping')
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise TypeError('parameter keys must be nonempty strings')
        _inputs(owner=owner, parameter=value)


class Registry:
    """Local strong references. Resolve once before capture or concurrent use."""
    def __init__(self):
        self._functions = {}

    @staticmethod
    def _name(name):
        if not isinstance(name, str):
            raise TypeError('registry name must be a string')
        if (len(name) > 128 or re.fullmatch(r'[a-z][a-z0-9_-]*/[a-z][a-z0-9_-]*', name) is None
                or name.startswith('gffx/')):
            raise ValueError('registry name is malformed or reserved')

    def register(self, name, function):
        self._name(name)
        if not callable(function):
            raise TypeError('registry function must be callable')
        if name in self._functions:
            raise ValueError('registry name already exists')
        self._functions[name] = function

    def resolve(self, name):
        self._name(name)
        return self._functions[name]

    def unregister(self, name):
        self._name(name)
        del self._functions[name]

    def names(self):
        return tuple(sorted(self._functions))
