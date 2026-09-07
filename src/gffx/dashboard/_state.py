"""The state tree of DASHBOARD_CONTRACT_V0_1.md section 1.

A tree maps a path to a series; a series maps steps to values of one kind. A queue is a series
with a bound, whose steps the server assigns. Every mutation happens under one lock, and the
mutation methods are the only code path for both live messages and run-log replay, which is how
replay reproduces a tree exactly.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any

from ._framing import BINARY_KINDS, ArrayView, ProtocolError, json_safe
from ._limits import ResourceLimits, value_bytes

KINDS = frozenset({"mesh", "points", "camera", "image", "scalar", "text", "record", "control"})
DEFAULT_BOUND = 256
_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def check_window_policy(bound, snapshot_every):
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
        raise ProtocolError("recent_window must be an integer >= 1")
    if isinstance(snapshot_every, bool) or not isinstance(snapshot_every, int) or snapshot_every < 0:
        raise ProtocolError("snapshot_every must be an integer >= 0 (0 disables snapshots)")


def check_path(path: Any) -> str:
    if not isinstance(path, str) or not path or len(path) > 1024:
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
    __slots__ = ("kind", "bound", "entries", "next_step", "sizes", "bytes", "snapshot_every")

    def __init__(self, kind: str, bound: int | None) -> None:
        self.kind = kind
        self.bound = bound
        self.entries: dict[int, Any] = {}
        self.next_step = 0
        self.sizes = {}
        self.bytes = 0
        self.snapshot_every = None  # None denotes the legacy clean-shutdown-tail protocol.

    def latest_step(self) -> int | None:
        return max(self.entries) if self.entries else None


class StateTree:
    def __init__(self, *, limits: ResourceLimits | None = None) -> None:
        self._lock = threading.RLock()
        self.limits = limits or ResourceLimits()
        self.retained_bytes = 0
        self.evicted_entries = 0
        self._age = OrderedDict()
        self._paths: dict[str, Series] = {}
        self.current_step = 0

    def _check_size(self, path, value):
        size = 256 + value_bytes(value)
        if not isinstance(value, dict) and size > 65536:
            raise ProtocolError("non-array value exceeds 64 KiB byte budget")
        if size > min(self.limits.path_bytes, self.limits.run_bytes):
            raise ProtocolError("value exceeds dashboard byte budget")
        if path not in self._paths and len(self._paths) >= self.limits.paths:
            raise ProtocolError("dashboard path limit reached")
        return size

    def _remove(self, path, step, *, eviction=True):
        series = self._paths[path]
        size = series.sizes.pop(step)
        del series.entries[step]
        series.bytes -= size
        self.retained_bytes -= size
        self._age.pop((path, step), None)
        self.evicted_entries += int(eviction)

    def _retain(self, path, step, value, size):
        series = self._paths[path]
        if step in series.entries:
            self._remove(path, step, eviction=False)
        series.entries[step] = value
        series.sizes[step] = size
        series.bytes += size
        self.retained_bytes += size
        self._age[(path, step)] = None
        bound = series.bound or (self.limits.text_entries if series.kind in ("text", "record") else self.limits.history_entries)
        bound = min(bound, self.limits.history_entries)
        while len(series.entries) > bound or series.bytes > self.limits.path_bytes:
            self._remove(path, min(series.entries))
        while self.retained_bytes > self.limits.run_bytes:
            old_path, old_step = next(iter(self._age))
            self._remove(old_path, old_step)

    def resources(self):
        return {"retained_bytes": self.retained_bytes, "evicted_entries": self.evicted_entries,
                "run_byte_limit": self.limits.run_bytes, "path_byte_limit": self.limits.path_bytes,
                "history_entry_limit": self.limits.history_entries, "text_entry_limit": self.limits.text_entries}

    def retention(self, path):
        with self._lock:
            series = self._paths[path]
            return {"retained_from": min(series.entries) if series.entries else None,
                    "history_limit": min(series.bound or (self.limits.text_entries if series.kind in ("text", "record") else self.limits.history_entries), self.limits.history_entries),
                    "recent_window": series.bound, "snapshot_every": series.snapshot_every,
                    **({"recent_steps": sorted(series.entries)} if series.bound is not None else {})}

    # -- mutation: the one path for live events and replay --------------------------------------

    def set(self, path: str, kind: str, step: int, value: Any) -> None:
        check_path(path)
        check_kind(kind)
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ProtocolError("step must be an integer >= 0, received %r" % (step,))
        with self._lock:
            size = self._check_size(path, value)
            series = self._paths.get(path)
            if series is None:
                series = self._paths[path] = Series(kind, None)
            elif series.kind != kind:
                raise ProtocolError("path %r holds kind %r; a %r value is refused" % (path, series.kind, kind))
            self._retain(path, step, value, size)
            series.next_step = max(series.next_step, step + 1)
            if step > self.current_step:
                self.current_step = step

    def push(self, path: str, kind: str, value: Any, bound: int | None = None,
             *, _restore_step: int | None = None, snapshot_every: int | None = None) -> int:
        check_path(path)
        check_kind(kind)
        if snapshot_every is not None:
            check_window_policy(bound, snapshot_every)
        if _restore_step is not None and (
            not isinstance(_restore_step, int) or isinstance(_restore_step, bool) or _restore_step < 0
        ):
            raise ProtocolError("restored queue step must be an integer >= 0")
        with self._lock:
            size = self._check_size(path, value)
            series = self._paths.get(path)
            if series is None:
                if bound is None:
                    bound = DEFAULT_BOUND
                if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
                    raise ProtocolError("bound must be an integer >= 1, received %r" % (bound,))
                series = self._paths[path] = Series(kind, bound)
            elif series.kind != kind:
                raise ProtocolError("path %r holds kind %r; a %r value is refused" % (path, series.kind, kind))
            elif series.bound is None:
                raise ProtocolError("path %r was created by set and is not a queue" % (path,))
            if snapshot_every is not None:
                series.bound = bound
                series.snapshot_every = snapshot_every
            # Tail records can repeat across clean shutdowns. Restore the original key directly,
            # without allocating a new step or evicting an entry before replacing that key.
            step = series.next_step if _restore_step is None else _restore_step
            self._retain(path, step, value, size)
            series.next_step = max(series.next_step, step + 1)
            return step

    def delete(self, path: str) -> bool:
        check_path(path)
        with self._lock:
            if path in self._paths:
                for step in list(self._paths[path].entries):
                    self._remove(path, step, eviction=False)
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

    def queue_tails(self) -> list[tuple[str, str, int, list[tuple[int, Any]]]]:
        """Every bounded path with its retained entries, oldest first, for writing at shutdown."""
        with self._lock:
            return [
                (path, series.kind, series.bound, [(step, series.entries[step]) for step in sorted(series.entries)])
                for path, series in self._paths.items() if series.bound is not None and series.snapshot_every is None
            ]

    def snapshot(self) -> dict[str, Any]:
        """A JSON-safe view: per path its kind, bound, steps, and latest value with arrays as metadata."""
        with self._lock:
            paths: dict[str, Any] = {}
            for path, series in self._paths.items():
                steps = sorted(series.entries)
                entry: dict[str, Any] = {"kind": series.kind, "steps": steps}
                entry.update(self.retention(path))
                if series.bound is not None:
                    entry["bound"] = series.bound
                if steps:
                    entry["latest"] = {"step": steps[-1], "value": json_safe(series.entries[steps[-1]])}
                if series.kind in ("scalar", "control"):
                    # Small enough to ship whole, so a plot never has to fetch its history step by step.
                    entry["values"] = [[step, series.entries[step]] for step in steps]
                paths[path] = entry
            return {"paths": paths, "current_step": self.current_step, "resources": self.resources()}

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

    def unregister(self, name: str) -> None:
        """Release a loaded tree only; its owner checks leases and durability first."""
        with self._lock:
            del self._runs[name]
            if self._default == name:
                self._default = next(reversed(self._runs), None)

    def runs(self) -> list[str]:
        with self._lock:
            return list(self._runs)

    def snapshot(self, run: str | None = None) -> dict[str, Any]:
        return self.run(run).tree.snapshot()

    def get(self, path: str, step: int | None = None, run: str | None = None) -> dict[str, Any]:
        return self.run(run).tree.get(path, step)

    def series(self, path: str, run: str | None = None) -> dict[int, dict[str, Any]]:
        return self.run(run).tree.series(path)
