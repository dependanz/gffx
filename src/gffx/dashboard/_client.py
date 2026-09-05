"""The Python client of DASHBOARD_CONTRACT_V0_1.md section 5.

Every send reduces its payload to bytes on the caller's thread, puts the frame on a bounded queue
and returns; a background thread writes frames and a second one reads acks, errors and forwarded
controls. A full queue drops its oldest frame and counts it. A dashboard that slows the process it
watches misreports the process, so nothing here blocks the caller unless it asked to wait for acks.
"""

from __future__ import annotations

import itertools
import os
import queue
import socket
import sys
import threading
import time
from typing import Any

from ._framing import ArrayView, ProtocolError, as_array, decode_text, encode_binary_parts, encode_text
from ._server import DEFAULT_PORT, PROTOCOL
from ._ws import ConnectionClosed, WebSocket, client_connect

_SENTINEL = object()


def _resolve_address(address: str | None) -> str | None:
    if address:
        return address
    env = os.environ.get("GFFX_DASHBOARD")
    if env:
        return env
    import gffx.dashboard as package  # looked up at call time so a test can substitute the lookup
    tailnet = package.tailscale_address()
    return None if tailnet is None else "%s:%d" % (tailnet, DEFAULT_PORT)


def _split(address: str) -> tuple[str, int]:
    from ._page import split_address
    return split_address(address)


def _default_run() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-%d" % os.getpid()


def _default_name() -> str:
    argv = sys.argv[:1] or ["python"]
    return os.path.basename(argv[0]) or "python"


class Control:
    """A live value a running process reads; the page changes it. Reads never block."""

    __slots__ = ("path", "value", "default", "kind", "updated_at")

    def __init__(self, path: str, default: Any, kind: str) -> None:
        self.path = path
        self.default = default
        self.value = default
        self.kind = kind
        self.updated_at: float | None = None

    def __repr__(self) -> str:
        return "Control(%s=%r)" % (self.path, self.value)


class Queue:
    """A bounded path the server assigns steps to; ``push`` is one send like any other."""

    __slots__ = ("_dashboard", "path", "kind", "bound")

    def __init__(self, dashboard: "Dashboard", path: str, kind: str, bound: int) -> None:
        self._dashboard = dashboard
        self.path = path
        self.kind = kind
        self.bound = bound

    def push(self, *args: Any, **kwargs: Any) -> None:
        fields, arrays = _build(self.kind, args, kwargs)
        envelope = {"t": "push", "path": self.path, "kind": self.kind, "bound": self.bound, **fields}
        self._dashboard._send(envelope, arrays)


def _build(kind: str, args: tuple, kwargs: dict) -> tuple[dict[str, Any], list[tuple[str, ArrayView]]]:
    """Envelope fields and wire arrays for one value of ``kind`` from a method's arguments."""
    if kind == "mesh":
        vertices, faces = args[0], args[1]
        arrays = [("vertices", as_array(vertices, dtype="float32")), ("faces", as_array(faces, dtype="uint32"))]
        for name in ("colors", "normals"):
            item = kwargs.get(name)
            if item is not None:
                view = as_array(item)
                if name == "colors" and view.dtype not in ("float32", "uint8"):
                    view = as_array(item, dtype="float32")
                if name == "normals" and view.dtype != "float32":
                    view = as_array(item, dtype="float32")
                arrays.append((name, view))
        return {}, arrays
    if kind == "points":
        arrays = [("positions", as_array(args[0], dtype="float32"))]
        colors = kwargs.get("colors")
        if colors is not None:
            view = as_array(colors)
            arrays.append(("colors", view if view.dtype in ("float32", "uint8") else as_array(colors, dtype="float32")))
        radii = kwargs.get("radii")
        if radii is not None:
            arrays.append(("radii", as_array(radii, dtype="float32")))
        return {}, arrays
    if kind == "image":
        view = as_array(args[0])
        if view.dtype not in ("uint8", "float32"):
            view = as_array(args[0], dtype="float32")
        shape = view.shape
        if len(shape) == 2:
            height, width, channels = shape[0], shape[1], 1
        elif len(shape) == 3 and shape[2] in (1, 3, 4):
            height, width, channels = shape
        elif len(shape) == 3 and shape[0] in (1, 3, 4):  # a framework's channels-first layout
            import numpy
            chw = view.to_numpy()
            view = as_array(numpy.ascontiguousarray(chw.transpose(1, 2, 0)))
            height, width, channels = view.shape
        else:
            raise ProtocolError("image needs [H,W], [H,W,C] or [C,H,W] with C in (1,3,4); received %r" % (shape,))
        return {"height": height, "width": width, "channels": channels, "dtype": view.dtype}, [("data", view)]
    if kind == "camera":
        matrix = as_array(args[0], dtype="float32")
        if matrix.shape != (4, 4):
            raise ProtocolError("camera needs a [4,4] world_to_clip; received %r" % (matrix.shape,))
        return {"value": {"world_to_clip": matrix.to_list(), "image_height": int(args[1]), "image_width": int(args[2])}}, []
    if kind == "scalar":
        value = args[0]
        if hasattr(value, "item"):
            value = value.item()
        return {"value": float(value)}, []
    if kind == "text":
        fields: dict[str, Any] = {"value": str(args[0])}
        if kwargs.get("markdown"):
            fields["markdown"] = True
        return fields, []
    if kind == "record":
        value = args[0]
        if not isinstance(value, dict):
            raise ProtocolError("record needs a dict")
        return {"value": value}, []
    raise ProtocolError("unknown kind %r" % (kind,))


class Dashboard:
    """One connection to a dashboard run. Construct through ``connect``."""

    def __init__(self, address: str | None, run: str, name: str, *, wait_acks: bool, queue_bound: int, timeout: float) -> None:
        self.address = address
        self.run = run
        self.name = name
        self.wait_acks = wait_acks
        self.timeout = timeout
        self.connected = False
        self.dropped = 0
        self.errors = 0
        self.last_error: str | None = None
        self._default_step = 0
        self._ids = itertools.count(1)
        self._controls: dict[str, Control] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_bound))
        self._ws: WebSocket | None = None
        self._sender: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._acks: dict[int, threading.Event] = {}
        self._ack_errors: dict[int, str] = {}
        self._acks_lock = threading.Lock()
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._inflight_zero = threading.Condition(self._inflight_lock)
        self._closing = False

    # -- lifecycle -------------------------------------------------------------------------------

    def _open(self) -> None:
        assert self.address is not None
        host, port = _split(self.address)
        try:
            self._ws = client_connect(host, port, timeout=self.timeout)
        except (OSError, ProtocolError) as error:
            raise ConnectionError("gffx.dashboard could not reach %s: %s" % (self.address, error)) from None
        hello_id = next(self._ids)
        self._ws.send_text(encode_text({"t": "hello", "id": hello_id, "run": self.run, "name": self.name, "protocol": PROTOCOL}))
        opcode, payload = self._ws.recv(timeout=self.timeout)
        reply = decode_text(payload)
        if reply.get("t") != "ack" or reply.get("id") != hello_id:
            self._ws.close()
            raise ConnectionError("gffx.dashboard hello refused by %s: %s" % (self.address, reply.get("message", reply)))
        self.connected = True
        self._reader = threading.Thread(target=self._read_loop, name="gffx-dashboard-reader", daemon=True)
        self._sender = threading.Thread(target=self._send_loop, name="gffx-dashboard-sender", daemon=True)
        self._reader.start()
        self._sender.start()

    def close(self, *, flush: bool = True) -> None:
        if not self.connected:
            return
        if flush:
            self.flush()
        self._closing = True
        self._queue.put(_SENTINEL)
        if self._sender is not None:
            self._sender.join(timeout=self.timeout)
        if self._ws is not None:
            self._ws.close()
        if self._reader is not None:
            self._reader.join(timeout=self.timeout)
        self.connected = False

    def __enter__(self) -> "Dashboard":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def flush(self, timeout: float | None = None) -> None:
        """Block until every queued frame has been written and, when acks are awaited, acknowledged."""
        if not self.connected:
            return
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        with self._inflight_zero:
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("%d frames still in flight after %.1fs" % (self._inflight, self.timeout if timeout is None else timeout))
                self._inflight_zero.wait(remaining)

    # -- the send path ---------------------------------------------------------------------------

    def _send(self, envelope: dict[str, Any], arrays: list[tuple[str, ArrayView]]) -> None:
        if not self.connected or self._closing:
            return
        message_id = next(self._ids)
        envelope["id"] = message_id
        frame: list[bytes] | str = encode_binary_parts(envelope, arrays) if arrays else encode_text(envelope)
        event: threading.Event | None = None
        if self.wait_acks:
            event = threading.Event()
            with self._acks_lock:
                self._acks[message_id] = event
        with self._inflight_lock:
            self._inflight += 1
        while True:
            try:
                self._queue.put_nowait((message_id, frame))
                break
            except queue.Full:
                try:
                    dropped_id, _ = self._queue.get_nowait()
                except queue.Empty:
                    continue
                self.dropped += 1
                self._finish(dropped_id)
        if event is not None:
            if not event.wait(self.timeout):
                raise TimeoutError("no ack for message %d within %.1fs" % (message_id, self.timeout))
            with self._acks_lock:
                error = self._ack_errors.pop(message_id, None)
            if error is not None:
                raise ProtocolError(error)

    def _finish(self, message_id: int) -> None:
        with self._inflight_zero:
            self._inflight -= 1
            if self._inflight <= 0:
                self._inflight = 0
                self._inflight_zero.notify_all()

    def _send_loop(self) -> None:
        assert self._ws is not None
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            message_id, frame = item
            try:
                if isinstance(frame, list):
                    self._ws.send_binary(frame)
                else:
                    self._ws.send_text(frame)
            except ConnectionClosed:
                self.connected = False
                self._finish(message_id)
                self._release_all()
                return
            if not self.wait_acks:
                self._finish(message_id)

    def _release_all(self) -> None:
        with self._acks_lock:
            for message_id, event in self._acks.items():
                self._ack_errors[message_id] = "connection closed before the ack"
                event.set()
            self._acks.clear()
        with self._inflight_zero:
            self._inflight = 0
            self._inflight_zero.notify_all()

    def _read_loop(self) -> None:
        assert self._ws is not None
        while True:
            try:
                opcode, payload = self._ws.recv()
            except (ConnectionClosed, socket.timeout, ProtocolError):
                self.connected = False
                self._release_all()
                return
            if opcode != 0x1:
                continue
            try:
                message = decode_text(payload)
            except ProtocolError:
                continue
            kind_of = message.get("t")
            if kind_of == "ack" or kind_of == "error":
                message_id = message.get("id")
                if kind_of == "error":
                    self.errors += 1
                    self.last_error = str(message.get("message"))
                if self.wait_acks and isinstance(message_id, int):
                    with self._acks_lock:
                        event = self._acks.pop(message_id, None)
                        if kind_of == "error" and event is not None:
                            self._ack_errors[message_id] = str(message.get("message"))
                    if event is not None:
                        self._finish(message_id)
                        event.set()
            elif kind_of == "control_set":
                control = self._controls.get(message.get("path"))
                if control is not None:
                    control.value = message.get("value")
                    control.updated_at = time.monotonic()

    # -- the surface -----------------------------------------------------------------------------

    def step(self, step: int) -> None:
        self._default_step = int(step)

    def _set(self, kind: str, path: str, args: tuple, kwargs: dict, step: int | None) -> None:
        if not self.connected:
            return
        fields, arrays = _build(kind, args, kwargs)
        envelope = {"t": "set", "path": path, "kind": kind, "step": self._default_step if step is None else int(step), **fields}
        self._send(envelope, arrays)

    def mesh(self, path: str, vertices: Any, faces: Any, *, colors: Any = None, normals: Any = None, step: int | None = None) -> None:
        self._set("mesh", path, (vertices, faces), {"colors": colors, "normals": normals}, step)

    def points(self, path: str, positions: Any, *, colors: Any = None, radii: Any = None, step: int | None = None) -> None:
        self._set("points", path, (positions,), {"colors": colors, "radii": radii}, step)

    def camera(self, path: str, world_to_clip: Any, image_height: int, image_width: int, *, step: int | None = None) -> None:
        self._set("camera", path, (world_to_clip, image_height, image_width), {}, step)

    def image(self, path: str, image: Any, *, step: int | None = None) -> None:
        self._set("image", path, (image,), {}, step)

    def scalar(self, path: str, value: Any, *, step: int | None = None) -> None:
        self._set("scalar", path, (value,), {}, step)

    def text(self, path: str, text: str, *, markdown: bool = False, step: int | None = None) -> None:
        self._set("text", path, (text,), {"markdown": markdown}, step)

    def record(self, path: str, record: dict[str, Any], *, step: int | None = None) -> None:
        """One JSON object at a step: a structured log entry, a test outcome, a configuration."""
        self._set("record", path, (record,), {}, step)

    def delete(self, path: str) -> None:
        if self.connected:
            self._send({"t": "delete", "path": path}, [])

    def queue(self, path: str, kind: str, *, bound: int = 256) -> Queue:
        return Queue(self, path, kind, int(bound))

    def control(self, path: str, default: Any, *, kind: str | None = None, range: Any = None, options: Any = None, label: str | None = None) -> Control:
        existing = self._controls.get(path)
        if existing is not None:
            return existing
        if kind is None:
            kind = "toggle" if isinstance(default, bool) else "slider" if isinstance(default, (int, float)) else "select" if options else "button"
        control = Control(path, default, kind)
        self._controls[path] = control
        if self.connected:
            envelope: dict[str, Any] = {"t": "control_def", "path": path, "kind": kind, "default": default}
            if range is not None:
                envelope["range"] = list(range)
            if options is not None:
                envelope["options"] = list(options)
            if label is not None:
                envelope["label"] = label
            self._send(envelope, [])
        return control

    def layout(self, spec: dict[str, Any]) -> None:
        if self.connected:
            self._send({"t": "layout", "spec": spec}, [])


def connect(address: str | None = None, run: str | None = None, name: str | None = None, *,
            wait_acks: bool = False, queue_bound: int = 64, timeout: float = 10.0) -> Dashboard:
    """Open a run on a dashboard server, or return a no-op dashboard when no address resolves.

    ``address`` defaults to ``GFFX_DASHBOARD``, then to this machine's Tailscale address on the
    default port. With neither, every method returns immediately and nothing is opened, so
    instrumented code runs unchanged without a server. An address that resolves but cannot be
    reached raises ``ConnectionError``: an explicit address is a claim the server is there.
    """
    resolved = _resolve_address(address)
    dashboard = Dashboard(resolved, run or _default_run(), name or _default_name(), wait_acks=wait_acks, queue_bound=queue_bound, timeout=timeout)
    if resolved is not None:
        dashboard._open()
    return dashboard
