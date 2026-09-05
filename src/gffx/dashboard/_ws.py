"""RFC 6455 WebSocket framing, written in-tree so the dashboard has no external dependency.

One class serves both roles. A client masks every frame it sends and a server never does, which
is the only asymmetry the protocol has. Text and binary messages, fragmentation, ping, pong and
close are handled; extensions and subprotocols are not negotiated.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import os
import socket
import struct
import threading

from ._framing import ProtocolError

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA
_MAX_MESSAGE = 1 << 31


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key.strip() + GUID).encode("ascii")).digest()).decode("ascii")


try:  # NumPy masks a megabyte in tens of microseconds and releases the GIL while it does
    import numpy as _numpy
except ImportError:  # the standard-library path: big-integer XOR, correct and slower
    _numpy = None


def _xor_mask(data: bytes | memoryview, mask: bytes) -> bytes:
    length = len(data)
    if length == 0:
        return b""
    if _numpy is not None and length >= 256:
        padded = length + (-length) % 4
        buffer = _numpy.frombuffer(bytes(data) + b"\0" * (padded - length), dtype=_numpy.uint32)
        (key,) = _numpy.frombuffer(mask, dtype=_numpy.uint32)
        return (buffer ^ key).tobytes()[:length]
    repeated = mask * (length // 4 + 1)
    return (int.from_bytes(data, "little") ^ int.from_bytes(repeated[:length], "little")).to_bytes(length, "little")


class ConnectionClosed(Exception):
    """The far side closed the connection, or the socket died beneath it."""


class WebSocket:
    def __init__(self, sock: socket.socket, *, mask: bool) -> None:
        self._sock = sock
        self._mask = mask
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        self._closed = False
        self._buffer = bytearray()

    @property
    def closed(self) -> bool:
        return self._closed

    # -- sending ---------------------------------------------------------------------------------

    def _send_frame(self, opcode: int, payload: bytes | list[bytes]) -> None:
        if isinstance(payload, list):  # parts assembled here, once, rather than on the caller's thread
            payload = b"".join(payload)
        length = len(payload)
        head = bytearray([0x80 | opcode])
        mask_bit = 0x80 if self._mask else 0
        if length < 126:
            head.append(mask_bit | length)
        elif length < 65536:
            head.append(mask_bit | 126)
            head += struct.pack(">H", length)
        else:
            head.append(mask_bit | 127)
            head += struct.pack(">Q", length)
        if self._mask:
            key = os.urandom(4)
            head += key
            payload = _xor_mask(payload, key)
        with self._send_lock:
            if self._closed:
                raise ConnectionClosed("send on a closed WebSocket")
            try:
                self._sock.sendall(bytes(head) + payload)
            except OSError as error:
                self._closed = True
                raise ConnectionClosed(str(error)) from None

    def send_text(self, text: str) -> None:
        self._send_frame(OP_TEXT, text.encode("utf-8"))

    def send_binary(self, payload: bytes | list[bytes]) -> None:
        self._send_frame(OP_BINARY, payload)

    def send_pong(self, payload: bytes = b"") -> None:
        self._send_frame(OP_PONG, payload)

    def send_ping(self, payload: bytes = b"") -> None:
        self._send_frame(OP_PING, payload)

    def close(self, code: int = 1000, reason: str = "") -> None:
        if self._closed:
            return
        try:
            self._send_frame(OP_CLOSE, struct.pack(">H", code) + reason.encode("utf-8"))
        except ConnectionClosed:
            pass
        self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    # -- receiving -------------------------------------------------------------------------------

    def _read_exact(self, count: int) -> bytes:
        while len(self._buffer) < count:
            try:
                chunk = self._sock.recv(max(65536, count - len(self._buffer)))
            except socket.timeout:
                raise
            except OSError as error:
                self._closed = True
                raise ConnectionClosed(str(error)) from None
            if not chunk:
                self._closed = True
                raise ConnectionClosed("connection closed by the far side")
            self._buffer += chunk
        out = bytes(self._buffer[:count])
        del self._buffer[:count]
        return out

    def _read_frame(self) -> tuple[bool, int, bytes]:
        b0, b1 = self._read_exact(2)
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        if length == 126:
            (length,) = struct.unpack(">H", self._read_exact(2))
        elif length == 127:
            (length,) = struct.unpack(">Q", self._read_exact(8))
        if length > _MAX_MESSAGE:
            raise ProtocolError("frame of %d bytes exceeds the message limit" % (length,))
        key = self._read_exact(4) if masked else None
        payload = self._read_exact(length)
        if key is not None:
            payload = _xor_mask(payload, key)
        return fin, opcode, payload

    def recv(self, timeout: float | None = None) -> tuple[int, bytes]:
        """The next complete text or binary message as ``(opcode, payload)``.

        Control frames are answered inline. ``socket.timeout`` propagates when ``timeout`` elapses
        with no complete message; ``ConnectionClosed`` when the far side is gone.
        """
        with self._recv_lock:
            if self._closed:
                raise ConnectionClosed("recv on a closed WebSocket")
            self._sock.settimeout(timeout)
            message_opcode: int | None = None
            parts: list[bytes] = []
            while True:
                fin, opcode, payload = self._read_frame()
                if opcode == OP_PING:
                    self.send_pong(payload)
                    continue
                if opcode == OP_PONG:
                    continue
                if opcode == OP_CLOSE:
                    self._closed = True
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    raise ConnectionClosed("close frame received")
                if opcode in (OP_TEXT, OP_BINARY):
                    if message_opcode is not None:
                        raise ProtocolError("new message began inside a fragmented one")
                    message_opcode = opcode
                elif opcode == OP_CONT:
                    if message_opcode is None:
                        raise ProtocolError("continuation frame without a message")
                else:
                    raise ProtocolError("unknown opcode %d" % (opcode,))
                parts.append(payload)
                if fin:
                    return message_opcode, b"".join(parts)


def client_connect(host: str, port: int, path: str = "/ws", *, timeout: float = 10.0) -> WebSocket:
    """Open a client WebSocket to ``host:port`` with the opening handshake of RFC 6455 section 4."""
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
    connection.putheader("Host", "%s:%d" % (host, port))
    connection.putheader("Upgrade", "websocket")
    connection.putheader("Connection", "Upgrade")
    connection.putheader("Sec-WebSocket-Key", key)
    connection.putheader("Sec-WebSocket-Version", "13")
    connection.endheaders()
    response = connection.getresponse()
    if response.status != 101:
        connection.close()
        raise ProtocolError("WebSocket handshake refused with HTTP %d" % (response.status,))
    if response.getheader("Sec-WebSocket-Accept", "") != accept_key(key):
        connection.close()
        raise ProtocolError("WebSocket handshake returned a wrong accept key")
    sock = connection.sock
    connection.sock = None  # the socket now belongs to the WebSocket
    sock.settimeout(None)
    return WebSocket(sock, mask=True)


def server_accept(handler) -> WebSocket:
    """Complete the server side of the handshake on a ``BaseHTTPRequestHandler`` and take its socket."""
    headers = handler.headers
    if headers.get("Upgrade", "").lower() != "websocket" or "upgrade" not in headers.get("Connection", "").lower():
        raise ProtocolError("not a WebSocket upgrade request")
    key = headers.get("Sec-WebSocket-Key")
    if not key or headers.get("Sec-WebSocket-Version") != "13":
        raise ProtocolError("WebSocket upgrade lacks a key or is not version 13")
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept_key(key))
    handler.end_headers()
    handler.wfile.flush()
    handler.close_connection = True
    return WebSocket(handler.connection, mask=False)
