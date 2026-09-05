"""gffx.dashboard: a configurable live dashboard for development and running processes.

The contract is DASHBOARD_CONTRACT_V0_1.md in the project memory. Nothing here is an operation:
the dashboard publishes no gradient and claims no conformance. It is a server (``Server``), a
Python client (``connect``, arriving with Phase 1 step 2), and a browser page (step 3), all in the
standard library, serving only the Tailscale interface.
"""

from __future__ import annotations

from ._client import Control, Dashboard, Queue, connect
from ._framing import ArrayView, ProtocolError, as_array
from ._page import PageConnection
from ._server import DEFAULT_PORT, PROTOCOL, BindRefused, Server, tailscale_address

__all__ = [
    "ArrayView",
    "BindRefused",
    "Control",
    "DEFAULT_PORT",
    "Dashboard",
    "PROTOCOL",
    "PageConnection",
    "ProtocolError",
    "Queue",
    "Server",
    "as_array",
    "connect",
    "tailscale_address",
]
