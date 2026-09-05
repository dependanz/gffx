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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ._framing import (
    BINARY_KINDS, ArrayView, ProtocolError, decode_binary, decode_text, encode_binary, encode_text,
    json_safe, split_value, value_from_arrays,
)
from ._state import KINDS, Runs, StateTree, check_kind, check_path
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

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._file = open(path, "ab")

    def append_text(self, envelope: dict[str, Any]) -> None:
        self._append(_LOG_TEXT, encode_text(envelope).encode("utf-8"))

    def append_binary(self, frame: bytes) -> None:
        self._append(_LOG_BINARY, frame)

    def _append(self, tag: int, payload: bytes) -> None:
        with self._lock:
            self._file.write(struct.pack("<IB", len(payload) + 1, tag))
            self._file.write(payload)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()

    @staticmethod
    def read(path: Path):
        with open(path, "rb") as handle:
            while True:
                head = handle.read(5)
                if len(head) < 5:
                    return
                length, tag = struct.unpack("<IB", head)
                payload = handle.read(length - 1)
                if len(payload) < length - 1:
                    return  # a torn final record: the process died mid-write; everything before it stands
                yield tag, payload


class Run:
    def __init__(self, name: str, directory: Path) -> None:
        self.name = name
        self.directory = directory
        self.tree = StateTree()
        self.log: RunLog | None = None
        self.layouts_dir = directory / "layouts"
        self.controls: dict[str, dict[str, Any]] = {}
        self.proposed_layout: dict[str, Any] | None = None
        self.clients: set[WebSocket] = set()
        self.pages: set["_PageSession"] = set()
        self.lock = threading.RLock()
        self.names: list[str] = []

    def open_log(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.layouts_dir.mkdir(exist_ok=True)
        self.log = RunLog(self.directory / "log.bin")

    # -- the one mutation path -------------------------------------------------------------------

    def apply(self, envelope: dict[str, Any], arrays: list[tuple[str, ArrayView]] | None = None) -> dict[str, Any] | None:
        """Apply one accepted event to the tree and return the page-facing update, if any."""
        kind_of = envelope.get("t")
        if kind_of == "hello":
            name = envelope.get("name")
            if isinstance(name, str) and name not in self.names:
                self.names.append(name)
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
            self.controls[path] = definition
            if self.tree.kind(path) is None:
                self.tree.set(path, "control", self.tree.current_step, envelope.get("default"))
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

    def record_text(self, envelope: dict[str, Any]) -> None:
        if self.log is not None:
            self.log.append_text(envelope)

    def record_binary(self, frame: bytes) -> None:
        if self.log is not None:
            self.log.append_binary(frame)

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
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or name.startswith("."):
            raise ProtocolError("layout needs a plain name")
        self.layouts_dir.mkdir(parents=True, exist_ok=True)
        (self.layouts_dir / (name + ".json")).write_text(json.dumps(spec, indent=1), encoding="utf-8")
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
        return json.loads(path.read_text(encoding="utf-8"))

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
        self.outbound: queue.Queue = queue.Queue(maxsize=_PAGE_QUEUE)
        self.dropped = 0
        self._writer = threading.Thread(target=self._drain, name="gffx-dashboard-page-writer", daemon=True)
        self._alive = True

    def start(self) -> None:
        self._writer.start()

    def enqueue(self, update: dict[str, Any], arrays: list[tuple[str, ArrayView]] | None) -> None:
        item = (update, arrays)
        while True:
            try:
                self.outbound.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.outbound.get_nowait()  # a page that cannot keep up sees fewer frames, never a stalled sender
                    self.dropped += 1
                except queue.Empty:
                    pass

    def _drain(self) -> None:
        while self._alive:
            try:
                update, arrays = self.outbound.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if arrays:
                    self.ws.send_binary(encode_binary(update, arrays))
                else:
                    self.ws.send_text(encode_text(update))
            except (ConnectionClosed, ProtocolError):
                self._alive = False

    def stop(self) -> None:
        self._alive = False


class _Handler(BaseHTTPRequestHandler):
    server_version = "gffx-dashboard/0.1"
    protocol_version = "HTTP/1.1"
    dashboard: "Server"  # set on the generated subclass

    def log_message(self, format: str, *args: Any) -> None:  # quiet by default; the server keeps its own count
        self.dashboard._requests += 1

    # -- HTTP ------------------------------------------------------------------------------------

    def do_GET(self) -> None:
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
                self._json({"run": run.name, **run.tree.snapshot(), "controls": run.controls, "layouts": run.layout_list()})
                return
            self.send_error(404)
            return
        self._static("index.html" if path in ("", "/") else path.lstrip("/"))

    def _run_summary(self, name: str) -> dict[str, Any]:
        run = self.dashboard.state.run(name)
        return {"run": name, "names": run.names, "paths": len(run.tree.snapshot()["paths"]), "current_step": run.tree.current_step}

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
        run = self.dashboard._open_run(run_name)
        with run.lock:
            run.clients.add(ws)
        try:
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
                    with run.lock:
                        update = run.apply(envelope, arrays)
                        if frame is not None and kind_of == "push":
                            frame = encode_binary(envelope, arrays)  # carries the assigned step
                        if frame is not None:
                            run.record_binary(frame)
                        else:
                            run.record_text(envelope)
                except ProtocolError as error:
                    ws.send_text(encode_text({"t": "error", "id": envelope.get("id"), "message": str(error)}))
                    continue
                ack: dict[str, Any] = {"t": "ack", "id": envelope.get("id")}
                if kind_of == "push":
                    ack["step"] = envelope["step"]
                ws.send_text(encode_text(ack))
                if update is not None:
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
        run = self.dashboard._open_run(run_name, default=False)
        session = _PageSession(ws, run)
        with run.lock:
            snapshot = run.tree.snapshot()
            snapshot.update({"t": "snapshot", "id": opened.get("id"), "run": run.name, "controls": run.controls,
                             "layouts": run.layout_list(), "proposed_layout": run.proposed_layout, "names": run.names})
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
                    if kind_of == "control_set":
                        with run.lock:
                            envelope["step"] = run.tree.current_step
                            update = run.apply(envelope)
                            run.record_text({"t": "control_set", "path": envelope["path"], "value": envelope.get("value"), "step": envelope["step"]})
                        run.forward_control(envelope)
                        run.broadcast(update, None)
                        reply = {"t": "ack", "id": envelope.get("id"), "step": envelope["step"]}
                    elif kind_of == "history":
                        path = check_path(envelope.get("path"))
                        kind, step, value = run.tree.raw(path, envelope.get("step"))
                        reply = {"t": "history", "id": envelope.get("id"), "path": path, "kind": kind, "step": step}
                        if kind in BINARY_KINDS:
                            meta, reply_arrays = split_value(value)
                            reply.update(meta)
                        else:
                            reply["value"] = json_safe(value)
                    elif kind_of == "layout_save":
                        name = run.layout_save(envelope.get("spec"))
                        reply = {"t": "ack", "id": envelope.get("id"), "name": name}
                    elif kind_of == "layout_list":
                        reply = {"t": "layouts", "id": envelope.get("id"), "names": run.layout_list()}
                    elif kind_of == "layout_load":
                        reply = {"t": "layout", "id": envelope.get("id"), "spec": run.layout_load(envelope.get("name"))}
                    elif kind_of == "ping":
                        reply = {"t": "pong", "id": envelope.get("id")}
                    else:
                        raise ProtocolError("unknown page message type %r" % (kind_of,))
                except (ProtocolError, KeyError) as error:
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
                 allow_loopback_for_tests: bool = False, also_loopback: bool = False) -> None:
        self.root = Path(root)
        self._requested_host = host
        self._port = port
        self._allow_loopback = allow_loopback_for_tests
        self._also_loopback = also_loopback
        self.state = Runs()
        self._httpd: ThreadingHTTPServer | None = None
        self._loopback_httpd: ThreadingHTTPServer | None = None
        self._threads: list[threading.Thread] = []
        self._requests = 0
        self.address: str | None = None
        self.host: str | None = None
        self.port: int | None = None

    # -- lifecycle -------------------------------------------------------------------------------

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
        self.host, self.port = self._httpd.server_address[0], self._httpd.server_address[1]
        self.address = "%s:%d" % (self.host, self.port)
        self._serve(self._httpd)
        if self._also_loopback and host not in _LOOPBACK:
            self._loopback_httpd = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
            self._loopback_httpd.daemon_threads = True
            self._serve(self._loopback_httpd)
        self.root.mkdir(parents=True, exist_ok=True)
        return self

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
        run = Run(name, self.run_dir(name))
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
            loaded = Run(run, directory)
            self.state.register(run, loaded)
        return loaded.replay()

    @property
    def requests(self) -> int:
        return self._requests
