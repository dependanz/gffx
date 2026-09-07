"""The dashboard server of DASHBOARD_CONTRACT_V0_1.md: HTTP for the page, WebSocket for state.

Standard library only. One ``ThreadingHTTPServer`` answers static requests and upgrades ``/ws``
connections; a client session (opened by ``hello``) writes the state tree through the run log and
receives acks and forwarded controls, and a page session (opened by ``open``) receives a snapshot
and every later update, and sends controls, history requests and layout operations.
"""

from __future__ import annotations

import ipaddress
import json
import os
import queue
import socket
import struct
import threading
import time
import warnings
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ._framing import (
    BINARY_KINDS, ArrayView, ProtocolError, decode_binary, decode_text, encode_binary, encode_text,
    json_safe, split_value, value_from_arrays,
)
from ._state import KINDS, Runs, StateTree, check_kind, check_path
from ._limits import ResourceLimits
from ._limits import value_bytes
from ._delivery import MeshDeltaCache
from ._ws import ConnectionClosed, WebSocket, server_accept

PROTOCOL = 1
DEFAULT_PORT = 8790
STATIC_DIR = Path(__file__).resolve().parent / "static"
_TAILNET = ipaddress.ip_network("100.64.0.0/10")
_LOOPBACK = ("127.0.0.1", "::1", "localhost")
_PAGE_QUEUE = 256
_LOG_TEXT, _LOG_BINARY = 0, 1


class BindRefused(RuntimeError):
    """The server was asked to bind an interface the contract forbids."""


def tailscale_address() -> str | None:
    """This machine's Tailscale IPv4 address, or ``None`` when it has none.

    Tailscale hands out addresses from the carrier-grade NAT block 100.64.0.0/10, so the interface
    is recognised by its address rather than by name. ``GFFX_DASHBOARD_TAILSCALE`` overrides the
    lookup for a machine whose address is not discoverable this way.
    """
    override = os.environ.get("GFFX_DASHBOARD_TAILSCALE")
    if override:
        return override
    candidates: list[str] = []
    try:
        candidates.extend(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.append(info[4][0])
    except OSError:
        pass
    for address in candidates:
        try:
            if ipaddress.ip_address(address) in _TAILNET:
                return address
        except ValueError:
            continue
    return None


class RunLog:
    """An append-only record of accepted events, replayable through the same mutation path."""

    def __init__(self, path: Path, *, max_bytes: int = 64 * 1024 * 1024,
                 total_bytes: int = 256 * 1024 * 1024, lock=None) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.total_bytes = total_bytes
        self._lock = lock or threading.RLock()
        self._file = open(path, "ab")

    def check_capacity(self, bytes_needed):
        own = self.path.stat().st_size
        total = sum(p.stat().st_size for p in self.path.parent.parent.glob("*/log.bin"))
        if own + bytes_needed > self.max_bytes or total + bytes_needed > self.total_bytes:
            raise ProtocolError("dashboard disk quota reached; existing logs preserved; archive logs or explicitly raise limits")

    def append_text(self, envelope: dict[str, Any]) -> None:
        self._append(_LOG_TEXT, encode_text(envelope).encode("utf-8"))

    def append_binary(self, frame: bytes) -> None:
        self._append(_LOG_BINARY, frame)

    def _append(self, tag: int, payload: bytes) -> None:
        with self._lock:
            self.check_capacity(len(payload) + 5)
            self._file.write(struct.pack("<IB", len(payload) + 1, tag))
            self._file.write(payload)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()

    @staticmethod
    def read(path: Path, *, end: int | None = None):
        with open(path, "rb") as handle:
            while True:
                if end is not None and handle.tell() + 5 > end:
                    return
                head = handle.read(5)
                if len(head) < 5:
                    return
                length, tag = struct.unpack("<IB", head)
                if length < 1 or length > 8 * 1024 * 1024 + 1:
                    raise ProtocolError("log record exceeds 8 MiB; legacy file preserved, replay refused")
                if end is not None and handle.tell() + length - 1 > end:
                    return
                payload = handle.read(length - 1)
                if len(payload) < length - 1:
                    return  # a torn final record: the process died mid-write; everything before it stands
                yield tag, payload


def check_provenance(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"revision", "dirty", "artifact_sha256"}:
        raise ProtocolError("provenance needs revision, dirty and/or artifact_sha256")
    if "dirty" in value and value["dirty"] is not None and not isinstance(value["dirty"], bool):
        raise ProtocolError("provenance dirty must be boolean or null")
    for key, lengths in (("revision", (40, 64)), ("artifact_sha256", (64,))):
        item = value.get(key)
        if item is not None and (not isinstance(item, str) or len(item) not in lengths or any(c not in "0123456789abcdefABCDEF" for c in item)):
            raise ProtocolError("provenance %s must be a full hexadecimal hash or null" % key)
    return dict(value)


class Run:
    def __init__(self, name: str, directory: Path, *, limits=None, disk_lock=None) -> None:
        self.name = name
        self.directory = directory
        self.limits = limits or ResourceLimits()
        self.disk_lock = disk_lock or threading.RLock()
        self.tree = StateTree(limits=self.limits)
        self.recording_warning = None
        self.log: RunLog | None = None
        self.layouts_dir = directory / "layouts"
        self.controls: dict[str, dict[str, Any]] = {}
        self.proposed_layout: dict[str, Any] | None = None
        self.clients: set[WebSocket] = set()
        self.pages: set["_PageSession"] = set()
        self.lock = threading.RLock()
        self.names: list[str] = []
        import uuid
        identity = directory / "identity.json"
        self.run_id = str(uuid.UUID(json.loads(identity.read_text(encoding="utf-8"))["run_id"])) if identity.exists() else str(uuid.uuid5(uuid.NAMESPACE_URL, directory.resolve().as_uri()))
        self.provenance = None

    def open_log(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.layouts_dir.mkdir(exist_ok=True)
        import uuid
        identity = self.directory / "identity.json"
        if not identity.exists():
            # Preserve path-derived IDs for legacy logs; new runs receive a portable random ID.
            if not (self.directory / "log.bin").exists():
                self.run_id = str(uuid.uuid4())
            with identity.open("x", encoding="utf-8") as handle:
                json.dump({"run_id": self.run_id}, handle)
        self.log = RunLog(self.directory / "log.bin", max_bytes=self.limits.log_bytes,
                          total_bytes=self.limits.total_log_bytes, lock=self.disk_lock)

    def resources(self):
        result = self.tree.resources()
        result.update({"log_bytes": self.log.path.stat().st_size if self.log else 0,
                       "log_byte_limit": self.limits.log_bytes, "recording_warning": self.recording_warning})
        return result

    def page_id(self, name):
        import uuid
        return str(uuid.uuid5(uuid.UUID(self.run_id), "layout:" + name))

    def reference_metadata(self):
        names = ["auto", *self.layout_list()]
        if self.proposed_layout is not None:
            names.append("@proposed")
        return {"run_id": self.run_id, "provenance": self.provenance or {},
                "pages": [{"name": name, "page_id": self.page_id(name)} for name in names]}

    # -- the one mutation path -------------------------------------------------------------------

    def apply(self, envelope: dict[str, Any], arrays: list[tuple[str, ArrayView]] | None = None) -> dict[str, Any] | None:
        """Apply one accepted event to the tree and return the page-facing update, if any."""
        kind_of = envelope.get("t")
        if kind_of == "hello":
            provenance = check_provenance(envelope.get("provenance"))
            if self.provenance is not None and provenance is not None and provenance != self.provenance:
                raise ProtocolError("run provenance cannot change; use a new run")
            name = envelope.get("name")
            if isinstance(name, str) and name not in self.names:
                if len(name) > 256:
                    raise ProtocolError("client display name exceeds 256 characters")
                if len(self.names) >= 32:
                    raise ProtocolError("run name-history limit reached")
                self.names.append(name)
            if self.provenance is None:
                self.provenance = provenance or {}
            return None
        if kind_of == "set":
            path = check_path(envelope.get("path"))
            kind = check_kind(envelope.get("kind"))
            step = envelope.get("step")
            value = self._value(kind, envelope, arrays)
            self.tree.set(path, kind, step, value)
            return {"t": "update", "path": path, "kind": kind, "step": step, "value": json_safe(value)}
        if kind_of == "push":
            path = check_path(envelope.get("path"))
            kind = check_kind(envelope.get("kind"))
            value = self._value(kind, envelope, arrays)
            if "snapshot_every" in envelope:
                from ._state import check_window_policy
                check_window_policy(envelope.get("bound"), envelope["snapshot_every"])
                source_step = envelope.get("step")
                if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step < 0:
                    raise ProtocolError("queue source step must be an integer >= 0")
                series = self.tree._paths.get(path)
                if series is not None and not envelope.get("checkpoint") and source_step < series.next_step:
                    raise ProtocolError("queue source steps must increase; resume after the last recorded step")
                step = self.tree.push(path, kind, value, envelope.get("bound"),
                                      _restore_step=source_step, snapshot_every=envelope["snapshot_every"])
            elif envelope.get("tail") and isinstance(envelope.get("step"), int):
                # a retained entry written at shutdown keeps its original step on replay
                step = self.tree.push(path, kind, value, envelope.get("bound"),
                                      _restore_step=envelope["step"])
            else:
                step = self.tree.push(path, kind, value, envelope.get("bound"))
                envelope["step"] = step
            return {"t": "update", "path": path, "kind": kind, "step": step, "value": json_safe(value), "queue": True}
        if kind_of == "delete":
            path = check_path(envelope.get("path"))
            self.tree.delete(path)
            return {"t": "update", "path": path, "deleted": True}
        if kind_of == "control_def":
            path = check_path(envelope.get("path"))
            definition = {
                "kind": envelope.get("kind") or _infer_control_kind(envelope.get("default")),
                "default": envelope.get("default"),
                "range": envelope.get("range"),
                "options": envelope.get("options"),
                "label": envelope.get("label"),
            }
            if len(encode_text(definition).encode("utf-8")) > 16384:
                raise ProtocolError("control definition exceeds 16 KiB")
            if self.tree.kind(path) is None or not self.tree.series(path):
                self.tree.set(path, "control", self.tree.current_step, envelope.get("default"))
            self.controls[path] = definition
            return {"t": "update", "path": path, "kind": "control", "control": definition,
                    "step": self.tree.current_step, "value": self.tree.raw(path)[2]}
        if kind_of == "control_set":
            path = check_path(envelope.get("path"))
            step = envelope.get("step")
            if not isinstance(step, int):
                step = envelope["step"] = self.tree.current_step
            self.tree.set(path, "control", step, envelope.get("value"))
            return {"t": "update", "path": path, "kind": "control", "step": step, "value": envelope.get("value")}
        if kind_of == "layout":
            spec = envelope.get("spec")
            if not isinstance(spec, dict):
                raise ProtocolError("layout needs a spec object")
            if len(encode_text(spec).encode("utf-8")) > 65536:
                raise ProtocolError("layout exceeds 64 KiB")
            self.proposed_layout = spec
            return {"t": "layout_proposed", "spec": spec}
        raise ProtocolError("unknown event type %r" % (kind_of,))

    @staticmethod
    def _value(kind: str, envelope: dict[str, Any], arrays: list[tuple[str, ArrayView]] | None) -> Any:
        if kind in BINARY_KINDS:
            if not arrays:
                raise ProtocolError("kind %r must arrive in a binary frame" % (kind,))
            return value_from_arrays(kind, envelope, arrays)
        if arrays:
            raise ProtocolError("kind %r does not carry arrays" % (kind,))
        if kind == "camera":
            value = envelope.get("value")
            if not isinstance(value, dict) or "world_to_clip" not in value:
                raise ProtocolError("camera needs a value with world_to_clip")
            return value
        if kind == "scalar":
            value = envelope.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProtocolError("scalar needs a number")
            return float(value)
        if kind == "text":
            value = envelope.get("value")
            if not isinstance(value, str):
                raise ProtocolError("text needs a string")
            return {"text": value, "markdown": bool(envelope.get("markdown"))} if envelope.get("markdown") else value
        if kind == "record":
            value = envelope.get("value")
            if not isinstance(value, dict):
                raise ProtocolError("record needs a JSON object")
            return value
        return envelope.get("value")

    # -- persistence -----------------------------------------------------------------------------

    def _checkpoint_events(self, path):
        """Read bounded records from disk; never load archived mesh history into the live tree."""
        log_path = self.directory / "log.bin"
        if not log_path.exists():
            return
        # Fix the read boundary briefly under the writer's lock, then release it during disk I/O.
        # A page must not hold up the producer while scanning a large archive.
        with self.lock:
            end = log_path.stat().st_size
        for tag, payload in RunLog.read(log_path, end=end):
            if tag == _LOG_TEXT:
                header, arrays = decode_text(payload), []
            else:
                header, arrays = decode_binary(payload)
            if header.get("path") == path and (header.get("checkpoint") or header.get("t") == "delete"):
                yield header, arrays

    def checkpoints(self, path, *, before=None, limit=100):
        """Ascending page of checkpoint step numbers; before is an exclusive older-page cursor."""
        check_path(path)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ProtocolError("checkpoint page limit must be an integer from 1 to 100")
        if before is not None and (isinstance(before, bool) or not isinstance(before, int) or before < 0):
            raise ProtocolError("checkpoint before must be an integer >= 0")
        steps = set()
        for header, _ in self._checkpoint_events(path):
            if header.get("t") == "delete":
                steps.clear()
            elif before is None or header["step"] < before:
                steps.add(header["step"])
                if len(steps) > limit + 1:
                    steps.remove(min(steps))
        ordered = sorted(steps)
        return {"steps": ordered[-limit:], "has_more": len(ordered) > limit}

    def history(self, path, step=None):
        check_path(path)
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ProtocolError("history step must be an integer >= 0")
        with self.lock:
            try:
                return self.tree.raw(path, step)
            except KeyError:
                if step is None:
                    raise
        found = None
        for header, arrays in self._checkpoint_events(path):
            if header.get("t") == "delete":
                found = None
            elif header["step"] == step:
                found = (header["kind"], step, self._value(header["kind"], header, arrays))
        if found is None:
            raise KeyError("no retained frame or checkpoint at %s step %s" % (path, step))
        return found

    def record_text(self, envelope: dict[str, Any]) -> None:
        if self.log is not None:
            self.log.append_text(envelope)

    def record_binary(self, frame: bytes) -> None:
        if self.log is not None:
            self.log.append_binary(frame)

    def record_queue_tails(self) -> int:
        """Write each queue's retained entries as push events, so replay restores the tail."""
        if self.log is None:
            return 0
        written = 0
        with self.lock, self.disk_lock:
            # Check the entire tail before appending any of it. Never discard existing history.
            required = 0
            for path, kind, bound, entries in self.tree.queue_tails():
                for step, value in entries:
                    header = {"t": "push", "path": path, "kind": kind, "bound": bound, "step": step, "tail": True}
                    if kind in BINARY_KINDS:
                        meta, arrays = split_value(value); header.update(meta)
                        required += 5 + len(encode_binary(header, arrays))
                    else:
                        header["value"] = value
                        if kind == "text" and isinstance(value, dict):
                            header["value"], header["markdown"] = value["text"], value["markdown"]
                        required += 5 + len(encode_text(header).encode("utf-8"))
            try:
                self.log.check_capacity(required)
            except ProtocolError as error:
                self.recording_warning = str(error) + "; queue tails NOT saved"
                warnings.warn(self.recording_warning, RuntimeWarning)
                return 0
            for path, kind, bound, entries in self.tree.queue_tails():
                for step, value in entries:
                    envelope = {"t": "push", "path": path, "kind": kind, "bound": bound, "step": step, "tail": True}
                    if kind in BINARY_KINDS:
                        meta, arrays = split_value(value)
                        envelope.update(meta)
                        self.log.append_binary(encode_binary(envelope, arrays))
                    else:
                        envelope["value"] = value
                        if kind == "text" and isinstance(value, dict):
                            envelope["value"], envelope["markdown"] = value["text"], value["markdown"]
                        self.log.append_text(envelope)
                    written += 1
        return written

    def replay(self) -> int:
        count = 0
        for tag, payload in RunLog.read(self.directory / "log.bin"):
            if tag == _LOG_TEXT:
                self.apply(decode_text(payload))
            else:
                header, arrays = decode_binary(payload)
                self.apply(header, arrays)
            count += 1
        return count

    # -- layouts ---------------------------------------------------------------------------------

    def layout_save(self, spec: dict[str, Any]) -> str:
        name = spec.get("name") if isinstance(spec, dict) else None
        if not isinstance(name, str) or not name or len(name) > 128 or "/" in name or "\\" in name or name.startswith("."):
            raise ProtocolError("layout needs a plain name")
        if name in ("auto", "@proposed"):
            raise ProtocolError("layout name is reserved")
        spec = {**spec, "page_id": self.page_id(name)}
        self.layouts_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(spec, indent=1)
        path = self.layouts_dir / (name + ".json")
        if len(payload.encode("utf-8")) > 65536 or (not path.exists() and len(self.layout_list()) >= 16):
            raise ProtocolError("saved layout quota reached (16 layouts, 64 KiB each)")
        path.write_text(payload, encoding="utf-8")
        return name

    def layout_list(self) -> list[str]:
        if not self.layouts_dir.is_dir():
            return []
        return sorted(item.stem for item in self.layouts_dir.glob("*.json"))

    def layout_load(self, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or "/" in name or "\\" in name:
            raise ProtocolError("layout name is malformed")
        path = self.layouts_dir / (name + ".json")
        if not path.is_file():
            raise ProtocolError("no layout named %r" % (name,))
        return {**json.loads(path.read_text(encoding="utf-8")), "page_id": self.page_id(name)}

    # -- fan-out ---------------------------------------------------------------------------------

    def broadcast(self, update: dict[str, Any], arrays: list[tuple[str, ArrayView]] | None) -> None:
        with self.lock:
            pages = list(self.pages)
        for page in pages:
            page.enqueue(update, arrays)

    def forward_control(self, envelope: dict[str, Any]) -> None:
        with self.lock:
            clients = list(self.clients)
        text = encode_text({"t": "control_set", "path": envelope["path"], "value": envelope.get("value"), "step": envelope.get("step")})
        for client in clients:
            try:
                client.send_text(text)
            except ConnectionClosed:
                with self.lock:
                    self.clients.discard(client)


def _infer_control_kind(default: Any) -> str:
    if isinstance(default, bool):
        return "toggle"
    if isinstance(default, (int, float)):
        return "slider"
    if isinstance(default, str):
        return "select"
    return "button"


class _PageSession:
    def __init__(self, ws: WebSocket, run: Run) -> None:
        self.ws = ws
        self.run = run
        self.pending = OrderedDict()
        self.pending_bytes = 0
        self._condition = threading.Condition(threading.RLock())
        self.fps = 10.0
        self.bytes_per_second = 2 * 1024 * 1024
        self.paused = False
        self.mesh_deltas = False
        self.delta_cache = MeshDeltaCache(run.limits.page_bytes)
        self._next_send = 0.0
        self.sent_bytes = 0
        self.dropped = 0
        self._writer = threading.Thread(target=self._drain, name="gffx-dashboard-page-writer", daemon=True)
        self._alive = True

    def start(self) -> None:
        self._writer.start()

    def enqueue(self, update: dict[str, Any], arrays: list[tuple[str, ArrayView]] | None) -> None:
        size = value_bytes(update) + value_bytes(arrays or [])
        key = update.get("path", "@" + update.get("t", "event"))
        with self._condition:
            if self.paused or size > self.run.limits.page_bytes:
                self.dropped += 1
                return
            if key in self.pending:
                self.pending_bytes -= self.pending.pop(key)[2]
                self.dropped += 1
            while self.pending and (self.pending_bytes + size > self.run.limits.page_bytes or len(self.pending) >= self.run.limits.paths):
                _, old = self.pending.popitem(last=False)
                self.pending_bytes -= old[2]; self.dropped += 1
            self.pending[key] = (update, arrays, size)
            self.pending_bytes += size
            self._condition.notify_all()

    def configure(self, options):
        fps = options.get("fps", self.fps)
        rate = options.get("bytes_per_second", self.bytes_per_second)
        paused = options.get("paused", self.paused)
        deltas = options.get("mesh_deltas", self.mesh_deltas)
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0.2 <= fps <= 30:
            raise ProtocolError("view fps must be between 0.2 and 30")
        if isinstance(rate, bool) or not isinstance(rate, int) or not 16384 <= rate <= 2 * 1024 * 1024:
            raise ProtocolError("view bytes_per_second must be 16384..2097152")
        if not isinstance(paused, bool) or not isinstance(deltas, bool):
            raise ProtocolError("view paused and mesh_deltas must be booleans")
        with self._condition:
            self.fps, self.bytes_per_second = fps, rate
            self.paused, self.mesh_deltas = paused, deltas
            self.pending.clear(); self.pending_bytes = 0
            self.delta_cache.clear()
            self._condition.notify_all()

    def seed_latest(self):
        with self.run.lock:
            snapshot = self.run.tree.snapshot()
            for path, entry in snapshot["paths"].items():
                if not entry["steps"]:
                    continue
                kind, step, value = self.run.tree.raw(path)
                header = {"t": "update", "path": path, "kind": kind, "step": step,
                          "value": json_safe(value), **self.run.tree.retention(path)}
                arrays = None
                if kind in BINARY_KINDS:
                    _, arrays = split_value(value)
                self.enqueue(header, arrays)

    def _drain(self) -> None:
        while self._alive:
            with self._condition:
                delay = self._next_send - time.monotonic()
                if self.paused or not self.pending or delay > 0:
                    self._condition.wait(timeout=min(max(delay, 0.01), 0.5) if delay > 0 else 0.5)
                    continue
                _, (update, arrays, size) = self.pending.popitem(last=False)
                self.pending_bytes -= size
                if arrays:
                    if self.mesh_deltas:
                        update, arrays = self.delta_cache.encode(update, arrays)
                    payload = encode_binary(update, arrays)
                    count = len(payload)
                else:
                    payload = encode_text(update)
                    count = len(payload.encode("utf-8"))
            # Socket backpressure must never hold the sender's queue lock.
            try:
                if arrays:
                    self.ws.send_binary(payload)
                else:
                    self.ws.send_text(payload)
                with self._condition:
                    self.sent_bytes += count
                    self._next_send = time.monotonic() + max(1 / self.fps, count / self.bytes_per_second)
            except (ConnectionClosed, ProtocolError, OSError):
                self._alive = False

    def stop(self) -> None:
        with self._condition:
            self._alive = False
            self.pending.clear(); self.pending_bytes = 0; self.delta_cache.clear()
            self._condition.notify_all()


class _Handler(BaseHTTPRequestHandler):
    server_version = "gffx-dashboard/0.1"
    protocol_version = "HTTP/1.1"
    dashboard: "Server"  # set on the generated subclass

    def log_message(self, format: str, *args: Any) -> None:  # quiet by default; the server keeps its own count
        self.dashboard._requests += 1

    # -- HTTP ------------------------------------------------------------------------------------

    def do_GET(self) -> None:
        if self.dashboard.handle_http(self):
            return
        if self.path.split("?", 1)[0] == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            self._websocket()
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/runs":
            self._json({"runs": [self._run_summary(name) for name in self.dashboard.state.runs()]})
            return
        if path.startswith("/api/runs/"):
            parts = path[len("/api/runs/"):].split("/")
            if len(parts) == 2 and parts[1] == "snapshot" and self.dashboard.state.has(parts[0]):
                run = self.dashboard.state.run(parts[0])
                self._json({"run": run.name, **run.tree.snapshot(), **run.reference_metadata(), "controls": run.controls, "layouts": run.layout_list()})
                return
            self.send_error(404)
            return
        self._static("index.html" if path in ("", "/") else path.lstrip("/"))

    def do_POST(self) -> None:
        if not self.dashboard.handle_http(self):
            self.send_error(405)

    def _run_summary(self, name: str) -> dict[str, Any]:
        run = self.dashboard.state.run(name)
        return {"run": name, "run_id": run.run_id, "names": run.names, "paths": len(run.tree.snapshot()["paths"]), "current_step": run.tree.current_step,
                "resources": run.resources()}

    def _json(self, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, relative: str) -> None:
        root = STATIC_DIR
        target = (root / relative).resolve()
        if root not in target.parents and target != root or not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        suffix = target.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
            ".json": "application/json", ".wgsl": "text/plain; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
        }.get(suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # -- WebSocket sessions ----------------------------------------------------------------------

    def _websocket(self) -> None:
        try:
            ws = server_accept(self)
        except ProtocolError as error:
            self.send_error(400, str(error))
            return
        try:
            opcode, payload = ws.recv(timeout=30.0)
            first = decode_text(payload) if opcode == 0x1 else decode_binary(payload)[0]
            if first.get("t") == "hello":
                self._client_session(ws, first)
            elif first.get("t") == "open":
                self._page_session(ws, first)
            else:
                ws.send_text(encode_text({"t": "error", "id": first.get("id"), "message": "the first message must be hello or open"}))
                ws.close(1002, "no session opened")
        except (ConnectionClosed, socket.timeout):
            pass
        except ProtocolError as error:
            try:
                ws.send_text(encode_text({"t": "error", "id": None, "message": str(error)}))
                ws.close(1002, "protocol error")
            except ConnectionClosed:
                pass

    def _client_session(self, ws: WebSocket, hello: dict[str, Any]) -> None:
        if hello.get("protocol") != PROTOCOL:
            ws.send_text(encode_text({"t": "error", "id": hello.get("id"), "message": "protocol %r is not %d" % (hello.get("protocol"), PROTOCOL)}))
            ws.close(1002, "protocol mismatch")
            return
        run_name = hello.get("run")
        if not isinstance(run_name, str) or not run_name or "/" in run_name or "\\" in run_name or run_name.startswith("."):
            ws.send_text(encode_text({"t": "error", "id": hello.get("id"), "message": "hello needs a plain run name"}))
            ws.close(1002, "bad run")
            return
        with self.dashboard._run_lock:
            run = self.dashboard._open_run(run_name)
            with run.lock:
                if len(run.clients) >= 4:
                    raise ProtocolError("dashboard sender limit reached (4 per run)")
                run.clients.add(ws)
        try:
            with run.lock, run.disk_lock:
                provenance = check_provenance(hello.get("provenance"))
                if run.provenance is not None and provenance is not None and provenance != run.provenance:
                    raise ProtocolError("run provenance cannot change; use a new run")
                run.record_text(hello)
                run.apply(hello)
            ws.send_text(encode_text({"t": "ack", "id": hello.get("id")}))
            while True:
                opcode, payload = ws.recv()
                if opcode == 0x1:
                    envelope = decode_text(payload)
                    arrays: list[tuple[str, ArrayView]] = []
                    frame: bytes | None = None
                else:
                    envelope, arrays = decode_binary(payload)
                    frame = payload
                kind_of = envelope.get("t")
                if kind_of == "ping":
                    ws.send_text(encode_text({"t": "pong", "id": envelope.get("id")}))
                    continue
                try:
                    if kind_of not in ("set", "push", "delete", "control_def", "layout"):
                        raise ProtocolError("unknown client message type %r" % (kind_of,))
                    with run.lock, run.disk_lock:
                        if envelope.get("checkpoint") or envelope.get("tail"):
                            raise ProtocolError("checkpoint and tail are server-owned log fields")
                        checkpoint = False
                        if kind_of == "push" and "snapshot_every" in envelope:
                            from ._state import check_window_policy
                            check_window_policy(envelope.get("bound"), envelope["snapshot_every"])
                            step = envelope.get("step")
                            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                                raise ProtocolError("queue source step must be an integer >= 0")
                            checkpoint = bool(envelope["snapshot_every"] and step % envelope["snapshot_every"] == 0)
                        durable = kind_of != "push" or checkpoint
                        recorded = {**envelope, "checkpoint": True} if checkpoint else envelope
                        recorded_frame = encode_binary(recorded, arrays) if checkpoint and arrays else frame
                        if durable and run.log:
                            size = len(recorded_frame) if recorded_frame is not None else len(encode_text(recorded).encode("utf-8"))
                            if size > 8 * 1024 * 1024:
                                raise ProtocolError("checkpoint log frame exceeds 8 MiB")
                            run.log.check_capacity(size + 5)
                        update = run.apply(envelope, arrays)
                        # Modern queues persist one full selected frame, never a live-window dump.
                        # Legacy pushes retain their clean-shutdown-tail compatibility behavior.
                        if durable:
                            if recorded_frame is not None:
                                run.record_binary(recorded_frame)
                            else:
                                run.record_text(recorded)
                        if update is not None and kind_of == "push":
                            update["checkpoint"] = checkpoint
                except ProtocolError as error:
                    if "disk quota" in str(error):
                        run.recording_warning = str(error)
                    ws.send_text(encode_text({"t": "error", "id": envelope.get("id"), "message": str(error)}))
                    continue
                ack: dict[str, Any] = {"t": "ack", "id": envelope.get("id")}
                if kind_of == "push":
                    ack["step"] = envelope["step"]
                    ack["checkpoint"] = checkpoint
                ws.send_text(encode_text(ack))
                if update is not None:
                    if update.get("path") and not update.get("deleted"):
                        update.update(run.tree.retention(update["path"]))
                    run.broadcast(update, arrays if update.get("kind") in BINARY_KINDS else None)
        except (ConnectionClosed, socket.timeout):
            pass
        except ProtocolError as error:
            try:
                ws.send_text(encode_text({"t": "error", "id": None, "message": str(error)}))
            except ConnectionClosed:
                pass
        finally:
            with run.lock:
                run.clients.discard(ws)
            ws.close()

    def _page_session(self, ws: WebSocket, opened: dict[str, Any]) -> None:
        run_name = opened.get("run")
        if not isinstance(run_name, str) or not run_name or "/" in run_name or "\\" in run_name or run_name.startswith("."):
            ws.send_text(encode_text({"t": "error", "id": opened.get("id"), "message": "open needs a plain run name"}))
            ws.close(1002, "bad run")
            return
        with self.dashboard._run_lock:
            run = self.dashboard._open_run(run_name, default=False)
            session = _PageSession(ws, run)
            if isinstance(opened.get("view"), dict):
                session.configure(opened["view"])
            with run.lock:
                if len(run.pages) >= 4:
                    raise ProtocolError("dashboard page limit reached (4 per run)")
                snapshot = run.tree.snapshot()
                snapshot.update({"t": "snapshot", "id": opened.get("id"), "run": run.name, "controls": run.controls,
                                 "layouts": run.layout_list(), "proposed_layout": run.proposed_layout, "names": run.names})
                snapshot.update(run.reference_metadata())
                run.pages.add(session)
        try:
            ws.send_text(encode_text(snapshot))
            session.start()
            while True:
                opcode, payload = ws.recv()
                envelope = decode_text(payload) if opcode == 0x1 else decode_binary(payload)[0]
                kind_of = envelope.get("t")
                reply: dict[str, Any]
                reply_arrays: list[tuple[str, ArrayView]] | None = None
                try:
                    if kind_of == "view_config":
                        session.configure(envelope)
                        if not session.paused:
                            session.seed_latest()
                        reply = {"t": "ack", "id": envelope.get("id"), "fps": session.fps,
                                 "bytes_per_second": session.bytes_per_second, "paused": session.paused}
                    elif kind_of == "control_set":
                        with run.lock, run.disk_lock:
                            envelope["step"] = run.tree.current_step
                            if run.log:
                                run.log.check_capacity(5 + len(encode_text(envelope).encode("utf-8")))
                            update = run.apply(envelope)
                            run.record_text({"t": "control_set", "path": envelope["path"], "value": envelope.get("value"), "step": envelope["step"]})
                        run.forward_control(envelope)
                        run.broadcast(update, None)
                        reply = {"t": "ack", "id": envelope.get("id"), "step": envelope["step"]}
                    elif kind_of == "checkpoints":
                        path = check_path(envelope.get("path"))
                        reply = {"t": "checkpoints", "id": envelope.get("id"), "path": path,
                                 **run.checkpoints(path, before=envelope.get("before"), limit=envelope.get("limit", 100))}
                    elif kind_of == "history":
                        path = check_path(envelope.get("path"))
                        kind, step, value = run.history(path, envelope.get("step"))
                        reply = {"t": "history", "id": envelope.get("id"), "path": path, "kind": kind, "step": step}
                        if kind in BINARY_KINDS:
                            meta, reply_arrays = split_value(value)
                            reply.update(meta)
                        else:
                            reply["value"] = json_safe(value)
                    elif kind_of == "layout_save":
                        name = run.layout_save(envelope.get("spec"))
                        reply = {"t": "ack", "id": envelope.get("id"), "name": name, "page_id": run.page_id(name)}
                    elif kind_of == "layout_list":
                        reply = {"t": "layouts", "id": envelope.get("id"), "names": run.layout_list()}
                    elif kind_of == "layout_load":
                        reply = {"t": "layout", "id": envelope.get("id"), "spec": run.layout_load(envelope.get("name"))}
                    elif kind_of == "ping":
                        reply = {"t": "pong", "id": envelope.get("id")}
                    else:
                        raise ProtocolError("unknown page message type %r" % (kind_of,))
                except (ProtocolError, KeyError) as error:
                    if "disk quota" in str(error):
                        run.recording_warning = str(error)
                    reply = {"t": "error", "id": envelope.get("id"), "message": str(error)}
                if reply_arrays:
                    ws.send_binary(encode_binary(reply, reply_arrays))
                else:
                    ws.send_text(encode_text(reply))
        except (ConnectionClosed, socket.timeout, ProtocolError):
            pass
        finally:
            session.stop()
            with run.lock:
                run.pages.discard(session)
            ws.close()


class Server:
    """The dashboard server. ``start`` binds; ``stop`` releases; ``state`` reads every run.

    ``root`` is the directory runs live under, normally the device's ``outputs`` resource for gffx.
    ``host`` defaults to the Tailscale address; any other interface is refused, except loopback
    when ``allow_loopback_for_tests`` is set or additionally with ``also_loopback``.
    """

    def __init__(self, root: str | os.PathLike, *, host: str | None = None, port: int = DEFAULT_PORT,
                 allow_loopback_for_tests: bool = False, also_loopback: bool = False,
                 tls: tuple[str | os.PathLike, str | os.PathLike] | None = None,
                 limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits()
        self._disk_lock = threading.RLock()
        self._run_lock = threading.RLock()
        self.root = Path(root)
        self._requested_host = host
        self._port = port
        self._allow_loopback = allow_loopback_for_tests
        self._also_loopback = also_loopback
        # WebGPU exists only in a secure context: https, or http on loopback. Over the tailnet
        # that means TLS; `tls` is a (certificate, key) PEM pair, and `tailscale serve` is the
        # other route, fronting the loopback binding with a certificate the browser trusts.
        self._tls = None if tls is None else (Path(tls[0]), Path(tls[1]))
        self.scheme = "http" if tls is None else "https"
        self.state = Runs()
        self._httpd: ThreadingHTTPServer | None = None
        self._loopback_httpd: ThreadingHTTPServer | None = None
        self._threads: list[threading.Thread] = []
        self._requests = 0
        self.address: str | None = None
        self.host: str | None = None
        self.port: int | None = None

    # -- lifecycle -------------------------------------------------------------------------------

    def handle_http(self, request) -> bool:
        """Application extension seam. Return True only after handling the HTTP request.

        Subclasses can compose their own routes while retaining the standard viewer and
        WebSocket protocol. The base server installs no extra routes or dependencies.
        """
        return False

    def _resolve_host(self) -> str:
        tailnet = tailscale_address()
        host = self._requested_host
        if host is None:
            if self._allow_loopback:
                return "127.0.0.1"
            if tailnet is None:
                raise BindRefused(
                    "gffx.dashboard binds only the Tailscale interface and this machine has no address in "
                    "100.64.0.0/10; set GFFX_DASHBOARD_TAILSCALE, or pass allow_loopback_for_tests=True in a test")
            return tailnet
        if tailnet is not None and host == tailnet:
            return host
        if host in _LOOPBACK and self._allow_loopback:
            return host
        raise BindRefused(
            "gffx.dashboard refuses to bind %s: the contract allows only the Tailscale interface%s. "
            "Loopback is available with also_loopback=True beside the tailnet, or allow_loopback_for_tests=True in a test."
            % (host, "" if tailnet is None else " (%s here)" % tailnet))

    def start(self) -> "Server":
        if self._httpd is not None:
            return self
        host = self._resolve_host()
        handler = type("GffxDashboardHandler", (_Handler,), {"dashboard": self})
        self._httpd = ThreadingHTTPServer((host, self._port), handler)
        self._httpd.daemon_threads = True
        self._wrap_tls(self._httpd)
        self.host, self.port = self._httpd.server_address[0], self._httpd.server_address[1]
        self.address = "%s:%d" % (self.host, self.port)
        self._serve(self._httpd)
        if self._also_loopback and host not in _LOOPBACK:
            self._loopback_httpd = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
            self._loopback_httpd.daemon_threads = True
            self._wrap_tls(self._loopback_httpd)
            self._serve(self._loopback_httpd)
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def _wrap_tls(self, httpd: ThreadingHTTPServer) -> None:
        if self._tls is None:
            return
        import ssl
        certificate, key = self._tls
        if not certificate.is_file() or not key.is_file():
            raise FileNotFoundError("TLS needs both files: %s and %s" % (certificate, key))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(certificate), str(key))
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    def _serve(self, httpd: ThreadingHTTPServer) -> None:
        thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, name="gffx-dashboard", daemon=True)
        thread.start()
        self._threads.append(thread)

    def stop(self) -> None:
        for httpd in (self._httpd, self._loopback_httpd):
            if httpd is not None:
                httpd.shutdown()
                httpd.server_close()
        self._httpd = self._loopback_httpd = None
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()
        for name in self.state.runs():
            run = self.state.run(name)
            for page in list(run.pages):
                page.stop()
            if run.log is not None:
                run.record_queue_tails()
                run.log.close()
                run.log = None

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- runs ------------------------------------------------------------------------------------

    def run_dir(self, run: str) -> Path:
        return self.root / run

    def _open_run(self, name: str, *, default: bool = True) -> Run:
        with self._run_lock:
            if len(name) > 128:
                raise ProtocolError("run name exceeds 128 characters")
            return self._open_run_locked(name, default=default)

    def _open_run_locked(self, name: str, *, default: bool = True) -> Run:
        """The run named ``name``, created with its directory and log when it does not exist.

        A client's ``hello`` makes the run the default one reads resolve against; a page opening a
        run ahead of any process - to lay it out before the fit starts - does not.
        """
        if self.state.has(name):
            run = self.state.run(name)
            if run.log is None:
                run.open_log()
            self.state.register(name, run, default=default)
            return run
        if len(self.state.runs()) >= self.limits.runs:
            raise ProtocolError("dashboard active-run limit reached")
        run = Run(name, self.run_dir(name), limits=self.limits, disk_lock=self._disk_lock)
        if (run.directory / "log.bin").is_file():
            run.replay()
        run.open_log()
        self.state.register(name, run, default=default)
        return run

    def replay(self, run: str) -> int:
        """Load a run's log from ``root`` into the tree without a client; returns the event count."""
        directory = self.run_dir(run)
        if not (directory / "log.bin").is_file():
            raise FileNotFoundError(directory / "log.bin")
        if self.state.has(run):
            loaded = self.state.run(run)
        else:
            if len(self.state.runs()) >= self.limits.runs:
                raise ProtocolError("dashboard active-run limit reached")
            loaded = Run(run, directory, limits=self.limits, disk_lock=self._disk_lock)
            self.state.register(run, loaded)
        return loaded.replay()

    @property
    def requests(self) -> int:
        return self._requests
