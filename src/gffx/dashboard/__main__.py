"""``python -m gffx.dashboard``: serve the dashboard on this machine's Tailscale interface."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

from ._server import DEFAULT_PORT, BindRefused, Server, tailscale_address


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gffx.dashboard", description="Serve the gffx dashboard.")
    parser.add_argument("--root", default=os.environ.get("GFFX_DASHBOARD_ROOT") or str(Path.home() / ".gffx" / "dashboard"),
                        help="directory runs are written under (default: GFFX_DASHBOARD_ROOT or ~/.gffx/dashboard)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=None, help="interface to bind; only the Tailscale address is accepted")
    parser.add_argument("--also-loopback", action="store_true", help="additionally serve 127.0.0.1 on the same port")
    parser.add_argument("--replay", action="append", default=[], metavar="RUN", help="load a run's log from --root at start")
    args = parser.parse_args(argv)

    server = Server(args.root, host=args.host, port=args.port, also_loopback=args.also_loopback)
    try:
        server.start()
    except BindRefused as error:
        print(error, file=sys.stderr)
        return 2
    except OSError as error:
        print("could not bind %s:%d: %s" % (args.host or tailscale_address(), args.port, error), file=sys.stderr)
        return 2
    for run in args.replay:
        try:
            count = server.replay(run)
            print("replayed %s: %d events" % (run, count))
        except FileNotFoundError:
            print("no log for run %r under %s" % (run, args.root), file=sys.stderr)
    print("gffx dashboard at http://%s/  (runs under %s)" % (server.address, args.root), flush=True)

    stop = threading.Event()

    def _stop(*_: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (AttributeError, ValueError):
        pass
    try:
        while not stop.wait(0.5):
            pass
    finally:
        server.stop()
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
