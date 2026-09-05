"""A page-side connection, as the browser page holds one, usable from Python for tests and tools."""

from __future__ import annotations

import itertools
import socket
import time
from typing import Any

from ._framing import ArrayView, ProtocolError, decode_binary, decode_text, encode_text
from ._ws import ConnectionClosed, WebSocket, client_connect


def split_address(address: str) -> tuple[str, int]:
    if address.startswith("ws://") or address.startswith("http://"):
        address = address.split("://", 1)[1]
    address = address.rstrip("/")
    if ":" not in address:
        from ._server import DEFAULT_PORT
        return address, DEFAULT_PORT
    host, port = address.rsplit(":", 1)
    return host, int(port)


class PageConnection:
    """One page session: ``open`` yields the snapshot; updates, replies and history follow."""

    def __init__(self, address: str, *, run: str, timeout: float = 10.0) -> None:
        self.address = address
        self.run = run
        self.timeout = timeout
        self._ws: WebSocket | None = None
        self._ids = itertools.count(1)
        self._pending: list[dict[str, Any]] = []
        self.snapshot: dict[str, Any] | None = None

    def open(self, layout: str | None = None) -> dict[str, Any]:
        host, port = split_address(self.address)
        self._ws = client_connect(host, port, timeout=self.timeout)
        envelope: dict[str, Any] = {"t": "open", "id": next(self._ids), "run": self.run}
        if layout is not None:
            envelope["layout"] = layout
        self._ws.send_text(encode_text(envelope))
        reply = self._wait(lambda message: message.get("t") in ("snapshot", "error"), self.timeout)
        if reply.get("t") == "error":
            raise ProtocolError(reply.get("message", "open refused"))
        self.snapshot = reply
        return reply

    # -- receiving -------------------------------------------------------------------------------

    def _recv_one(self, timeout: float | None) -> dict[str, Any]:
        assert self._ws is not None, "open() first"
        opcode, payload = self._ws.recv(timeout=timeout)
        if opcode == 0x1:
            return decode_text(payload)
        header, arrays = decode_binary(payload)
        header["arrays"] = dict(arrays)
        return header

    def _wait(self, predicate, timeout: float) -> dict[str, Any]:
        for index, message in enumerate(self._pending):
            if predicate(message):
                return self._pending.pop(index)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no matching message within %.1fs" % (timeout,))
            try:
                message = self._recv_one(remaining)
            except socket.timeout:
                raise TimeoutError("no matching message within %.1fs" % (timeout,)) from None
            if predicate(message):
                return message
            self._pending.append(message)

    def next_update(self, timeout: float = 5.0) -> dict[str, Any]:
        """The next ``update`` from the server, with binary arrays under ``arrays``."""
        return self._wait(lambda message: message.get("t") == "update", timeout)

    def next_message(self, timeout: float = 5.0) -> dict[str, Any]:
        return self._wait(lambda message: True, timeout)

    def _request(self, envelope: dict[str, Any], reply_types: tuple[str, ...]) -> dict[str, Any]:
        assert self._ws is not None, "open() first"
        envelope["id"] = next(self._ids)
        self._ws.send_text(encode_text(envelope))
        reply = self._wait(lambda message: message.get("id") == envelope["id"] and message.get("t") in reply_types + ("error",), self.timeout)
        if reply.get("t") == "error":
            raise ProtocolError(reply.get("message", "request refused"))
        return reply

    # -- page actions ----------------------------------------------------------------------------

    def control_set(self, path: str, value: Any) -> int:
        return int(self._request({"t": "control_set", "path": path, "value": value}, ("ack",))["step"])

    def history(self, path: str, step: int | None = None) -> dict[str, Any]:
        envelope: dict[str, Any] = {"t": "history", "path": path}
        if step is not None:
            envelope["step"] = step
        return self._request(envelope, ("history",))

    def layout_save(self, spec: dict[str, Any]) -> str:
        return str(self._request({"t": "layout_save", "spec": spec}, ("ack",))["name"])

    def layout_list(self) -> list[str]:
        return list(self._request({"t": "layout_list"}, ("layouts",))["names"])

    def layout_load(self, name: str) -> dict[str, Any]:
        return dict(self._request({"t": "layout_load", "name": name}, ("layout",))["spec"])

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except ConnectionClosed:
                pass
            self._ws = None
