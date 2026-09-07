"""Separate developer HTTP application composed with the general dashboard server."""
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit, parse_qs
from gffx.dashboard import Server
from gffx.dashboard._framing import ProtocolError
from .model import Index

STATIC = Path(__file__).parent / "static"

class TestingServer(Server):
    def __init__(self, root, **kwargs):
        super().__init__(Path(root) / "viewer", **kwargs)
        self.index = Index(Path(root) / "testing.sqlite")
        self.presentation_root=Path(root)/'presentation'

    def _open_run_locked(self, name, *, default=True):
        # The parent reserves sender/page leases under _run_lock before returning.
        # Only this archival application retires trees; the general dashboard does not.
        if not self.state.has(name) and len(self.state.runs()) >= self.limits.runs:
            for candidate in self.state.runs():
                loaded = self.state.run(candidate)
                try:
                    finished = self.index.summary(candidate)['state'] == 'finished'
                except KeyError:
                    continue
                with loaded.lock:
                    if not finished or loaded.clients or loaded.pages or loaded.recording_warning:
                        continue
                    paths = loaded.tree.snapshot()['paths'].values()
                    # Never throw away an ephemeral queue tail or a legacy unsaved queue.
                    if any('bound' in p and (not p.get('snapshot_every') or
                           any(s % p['snapshot_every'] for s in p['steps'])) for p in paths):
                        continue
                    if loaded.log is not None:
                        loaded.log.close()
                        loaded.log = None
                    self.state.unregister(candidate)
                    break
        return super()._open_run_locked(name, default=default)

    def handle_http(self, request):
        url = urlsplit(request.path)
        if not url.path.startswith("/testing"): return False
        def reply(value, status=200, mime="application/json"):
            body = json.dumps(value).encode() if mime == "application/json" else value
            request.send_response(status)
            request.send_header("Content-Type", mime)
            request.send_header("Content-Length", str(len(body)))
            request.send_header("Cache-Control", "private, max-age=31536000, immutable" if mime=='image/png' else "no-store")
            request.send_header("X-Content-Type-Options", "nosniff")
            request.end_headers()
            request.wfile.write(body)
        try:
            query = {k:v[0] for k,v in parse_qs(url.query).items()}
            if request.command == "POST":
                origin = request.headers.get("Origin")
                if origin and urlsplit(origin).netloc != request.headers.get("Host"):
                    reply({"error":"cross-origin write refused"},403); return True
                if url.path != "/testing/api/events":
                    reply({"error":"unknown endpoint"},404); return True
                if request.headers.get("Content-Type","").split(";")[0] != "application/json":
                    reply({"error":"JSON required"},415); return True
                if request.headers.get("Transfer-Encoding"):
                    raise ValueError("chunked requests unsupported")
                size = int(request.headers.get("Content-Length","0"))
                if not 0 < size <= 1024*1024: raise ValueError("event limit is 1 MiB")
                request.connection.settimeout(3)
                body = request.rfile.read(size)
                if len(body) != size: raise ValueError("incomplete request")
                self.index.accept(json.loads(body))
                reply({"accepted":True})
            elif request.command=='GET' and url.path.startswith('/testing/media/'):
                from .presentation import read_asset
                try:reply(read_asset(self.presentation_root,url.path.removeprefix('/testing/media/')),mime='image/png')
                except (FileNotFoundError,OSError):reply({'error':'Presentation asset unavailable'},404)
            elif url.path == "/testing/api/pages": reply({"pages":self.index.pages()})
            elif url.path == '/testing/api/directory': reply(self.index.directory(offset=query.get('offset',0),limit=query.get('limit',50)))
            elif url.path == "/testing/api/page":
                reply(self.index.page(query['page'],run=query.get('run'),offset=query.get('offset',0),search=query.get('search',''),outcome=query.get('outcome','')))
            elif url.path == "/testing/api/latest-coverage": reply(self.index.latest_coverage(run=query.get('run'),offset=query.get('offset',0),layer=query.get('layer','python')))
            elif url.path == "/testing/api/runs": reply({"runs":self.index.runs()})
            elif url.path == "/testing/api/run": reply(self.index.summary(query["run"]))
            elif url.path == "/testing/api/tests":
                reply(self.index.tests(query["run"], offset=query.get("offset",0), search=query.get("search",""),outcome=query.get("outcome","")))
            elif url.path == "/testing/api/test": reply(self.index.test(query["run"],query["test"]))
            elif url.path == "/testing/api/coverage": reply(self.index.coverage(query["run"],offset=query.get("offset",0),layer=query.get('layer','python')))
            else:
                name = "index.html" if url.path in ("/testing","/testing/") else url.path.removeprefix("/testing/")
                if name not in ("index.html","app.js","app.css","model.js","playback.js","cinema.js"):
                    reply({"error":"not found"},404)
                else:
                    mime = {"html":"text/html; charset=utf-8","js":"text/javascript; charset=utf-8","css":"text/css; charset=utf-8"}[name.rsplit(".",1)[1]]
                    reply((STATIC/name).read_bytes(),mime=mime)
        except KeyError as error: reply({"error":str(error)},404)
        except (ValueError, TypeError, OverflowError, sqlite3.Error, ProtocolError) as error: reply({"error":str(error)},400)
        return True
