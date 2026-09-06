"""``python -m gffx.dashboard``: serve the dashboard on this machine's Tailscale interface."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

from ._server import DEFAULT_PORT, BindRefused, Server, tailscale_address
from ._limits import ResourceLimits


def _self_signed(root: Path, host: str | None) -> tuple[Path, Path] | None:
    """A self-signed certificate for ``host`` under ``root/tls``, made with openssl; reused when present."""
    import shutil
    import subprocess
    directory = root / "tls"
    certificate, key = directory / "dashboard.crt", directory / "dashboard.key"
    if certificate.is_file() and key.is_file():
        return certificate, key
    openssl = shutil.which("openssl")
    if openssl is None:
        print("--self-signed needs openssl on PATH; pass --tls CERT KEY with a certificate made elsewhere", file=sys.stderr)
        return None
    directory.mkdir(parents=True, exist_ok=True)
    names = ["DNS:localhost", "IP:127.0.0.1"]
    if host:
        names.append(("IP:" if host.replace(".", "").isdigit() else "DNS:") + host)
    command = [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650", "-subj", "/CN=gffx dashboard",
               "-addext", "subjectAltName=" + ",".join(names), "-keyout", str(key), "-out", str(certificate)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("openssl failed: %s" % (result.stderr.strip(),), file=sys.stderr)
        return None
    print("self-signed certificate written to %s (the browser will warn once per device)" % (directory,), flush=True)
    return certificate, key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gffx.dashboard", description="Serve the gffx dashboard.")
    parser.add_argument("--root", default=os.environ.get("GFFX_DASHBOARD_ROOT") or str(Path.home() / ".gffx" / "dashboard"),
                        help="directory runs are written under (default: GFFX_DASHBOARD_ROOT or ~/.gffx/dashboard)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-log-mib", type=int, default=64, help="per-run log cap; existing files are never deleted")
    parser.add_argument("--max-total-log-mib", type=int, default=256, help="log cap across the run root")
    parser.add_argument("--history-entries", type=int, default=2048, help="maximum retained entries per path; text defaults to 64")
    parser.add_argument("--host", default=None, help="interface to bind; only the Tailscale address is accepted")
    parser.add_argument("--also-loopback", action="store_true", help="additionally serve 127.0.0.1 on the same port")
    parser.add_argument("--replay", action="append", default=[], metavar="RUN", help="load a run's log from --root at start")
    parser.add_argument("--tls", nargs=2, metavar=("CERT.pem", "KEY.pem"), default=None,
                        help="serve https with this certificate and key; WebGPU needs a secure context, which plain http over the tailnet is not")
    parser.add_argument("--self-signed", action="store_true",
                        help="generate a self-signed certificate under --root with openssl (if on PATH) and serve https with it; "
                             "the browser warns once per device. `tailscale serve` fronting --also-loopback is the route with a trusted certificate.")
    args = parser.parse_args(argv)

    tls = tuple(args.tls) if args.tls else None
    if args.self_signed and tls is None:
        tls = _self_signed(Path(args.root), args.host or tailscale_address())
        if tls is None:
            return 2

    try:
        limits = ResourceLimits(log_bytes=args.max_log_mib * 1048576,
                                total_log_bytes=args.max_total_log_mib * 1048576,
                                history_entries=args.history_entries)
    except ValueError as error:
        parser.error(str(error))
    server = Server(args.root, host=args.host, port=args.port, also_loopback=args.also_loopback, tls=tls, limits=limits)
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
    print("gffx dashboard at %s://%s/  (runs under %s)" % (server.scheme, server.address, args.root), flush=True)
    if server.scheme == "http" and not args.also_loopback:
        print("note: WebGPU needs a secure context; over the tailnet use --tls or --self-signed, or `tailscale serve` with --also-loopback", flush=True)

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
