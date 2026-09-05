"""The state tree of DASHBOARD_CONTRACT_V0_1.md section 1.

A tree maps a path to a series; a series maps steps to values of one kind. A queue is a series
with a bound, whose steps the server assigns. Every mutation happens under one lock, and the
mutation methods are the only code path for both live messages and run-log replay, which is how
replay reproduces a tree exactly.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from ._framing import BINARY_KINDS, ArrayView, ProtocolError, json_safe

KINDS = frozenset({"mesh", "points", "camera", "image", "scalar", "text", "record", "control"})
DEFAULT_BOUND = 256
_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def check_path(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise ProtocolError("path must be a nonempty string")
    for segment in path.split("/"):
        if not _SEGMENT.match(segment):
            raise ProtocolError("path segment %r is not [A-Za-z0-9_.-]+" % (segment,))
    return path


def check_kind(kind: Any) -> str:
    if kind not in KINDS:
        raise ProtocolError("kind %r is not one of %s" % (kind, ", ".join(sorted(KINDS))))
    return kind


class Series:
    __slots__ = ("kind", "bound", "entries", "next_step")

    def __init__(self, kind: str, bound: int | None) -> None:
        self.kind = kind
        self.bound = bound
        self.entries: dict[int, Any] = {}
        self.next_step = 0

    def latest_step(self) -> int | None:
        return max(self.entries) if self.entries else None


class StateTree:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._paths: dict[str, Series] = {}
        self.current_step = 0

    # -- mutation: the one path for live events and replay --------------------------------------

    def set(self, path: str, kind: str, step: int, value: Any) -> None:
        check_path(path)
        check_kind(kind)
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ProtocolError("step must be an integer >= 0, received %r" % (step,))
        with self._lock:
            series = self._paths.get(path)
            if series is None:
                series = self._paths[path] = Series(kind, None)
            elif series.kind != kind:
                raise ProtocolError("path %r holds kind %r; a %r value is refused" % (path, series.kind, kind))
            series.entries[step] = value
            series.next_step = max(series.next_step, step + 1)
            if step > self.current_step:
                self.current_step = step

    def push(self, path: str, kind: str, value: Any, bound: int | None = None) -> int:
        check_path(path)
        check_kind(kind)
        with self._lock:
            series = self._paths.get(path)
            if series is None:
                if bound is None:
                    bound = DEFAULT_BOUND
                if not isinstance(bound, int) or bound < 1:
                    raise ProtocolError("bound must be an integer >= 1, received %r" % (bound,))
                series = self._paths[path] = Series(kind, bound)
            elif series.kind != kind:
                raise ProtocolError("path %r holds kind %r; a %r value is refused" % (path, series.kind, kind))
            elif series.bound is None:
                raise ProtocolError("path %r was created by set and is not a queue" % (path,))
            step = series.next_step
            series.entries[step] = value
            series.next_step = step + 1
            while len(series.entries) > series.bound:
                del series.entries[min(series.entries)]
            return step

    def delete(self, path: str) -> bool:
        check_path(path)
        with self._lock:
            return self._paths.pop(path, None) is not None

    # -- reads -----------------------------------------------------------------------------------

    def kind(self, path: str) -> str | None:
        with self._lock:
            series = self._paths.get(path)
            return None if series is None else series.kind

    def get(self, path: str, step: int | None = None) -> dict[str, Any]:
        """The value at ``step`` (latest when omitted), flattened with its kind and step.

        Arrays come back as NumPy arrays when NumPy is importable and as ``ArrayView`` otherwise.
        """
        with self._lock:
            series = self._paths.get(path)
            if series is None:
                raise KeyError(path)
            if step is None:
                step = series.latest_step()
                if step is None:
                    raise KeyError("%s has no entries" % (path,))
            if step not in series.entries:
                raise KeyError("%s has no entry at step %d" % (path, step))
            return self._flatten(series.kind, step, series.entries[step])

    def raw(self, path: str, step: int | None = None) -> tuple[str, int, Any]:
        with self._lock:
            series = self._paths.get(path)
            if series is None:
                raise KeyError(path)
            if step is None:
                step = series.latest_step()
                if step is None:
                    raise KeyError("%s has no entries" % (path,))
            return series.kind, step, series.entries[step]

    def series(self, path: str) -> dict[int, dict[str, Any]]:
        """Every entry of a path by ascending step, each flattened as ``get`` returns it."""
        with self._lock:
            series = self._paths.get(path)
            if series is None:
                raise KeyError(path)
            return {step: self._flatten(series.kind, step, series.entries[step]) for step in sorted(series.entries)}

    def snapshot(self) -> dict[str, Any]:
        """A JSON-safe view: per path its kind, bound, steps, and latest value with arrays as metadata."""
        with self._lock:
            paths: dict[str, Any] = {}
            for path, series in self._paths.items():
                steps = sorted(series.entries)
                entry: dict[str, Any] = {"kind": series.kind, "steps": steps}
                if series.bound is not None:
                    entry["bound"] = series.bound
                if steps:
                    entry["latest"] = {"step": steps[-1], "value": json_safe(series.entries[steps[-1]])}
                paths[path] = entry
            return {"paths": paths, "current_step": self.current_step}

    @staticmethod
    def _flatten(kind: str, step: int, value: Any) -> dict[str, Any]:
        if kind in BINARY_KINDS:
            out: dict[str, Any] = {"kind": kind, "step": step}
            for key, item in value.items():
                out[key] = _materialize(item) if isinstance(item, ArrayView) else item
            return out
        if kind == "camera":
            return {"kind": kind, "step": step, **{k: (_materialize(v) if isinstance(v, ArrayView) else v) for k, v in value.items()}}
        return {"kind": kind, "step": step, "value": value}


def _materialize(view: ArrayView) -> Any:
    try:
        return view.to_numpy()
    except ImportError:
        return view


class Runs:
    """Every run the server holds, and the default run reads resolve against.

    The default is the run most recently opened by a client; tests and tools that address one run
    pass ``run=`` explicitly.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, Any] = {}
        self._default: str | None = None

    def register(self, name: str, run: Any, *, default: bool = True) -> None:
        with self._lock:
            self._runs[name] = run
            if default or self._default is None:
                self._default = name

    def run(self, name: str | None = None) -> Any:
        with self._lock:
            key = self._default if name is None else name
            if key is None or key not in self._runs:
                raise KeyError("no run named %r" % (key,))
            return self._runs[key]

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._runs

    def runs(self) -> list[str]:
        with self._lock:
            return list(self._runs)

    def snapshot(self, run: str | None = None) -> dict[str, Any]:
        return self.run(run).tree.snapshot()

    def get(self, path: str, step: int | None = None, run: str | None = None) -> dict[str, Any]:
        return self.run(run).tree.get(path, step)

    def series(self, path: str, run: str | None = None) -> dict[int, dict[str, Any]]:
        return self.run(run).tree.series(path)
