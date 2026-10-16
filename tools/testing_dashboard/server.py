"""Separate developer HTTP application composed with the general dashboard server."""
import json
import hashlib
import stat
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit, parse_qs
from gffx.dashboard import Server
from gffx.dashboard._framing import ProtocolError
from .model import Index

STATIC = Path(__file__).parent / "static"

class TestingServer(Server):
    def __init__(self, root, *, history_keep=100, max_active=32, metadata_bytes=64*1024*1024, **kwargs):
        super().__init__(Path(root) / "viewer", **kwargs)
        self.index = Index(Path(root) / "testing.sqlite",history_keep=history_keep,
                           max_active=max_active,metadata_bytes=metadata_bytes)
        self.presentation_root=Path(root)/'presentation'
        self.review_root=Path(root)

    def _open_run_locked(self, name, *, default=True):
        if self.index.is_retired(name):raise ProtocolError('execution evidence retired by retention policy')
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

    def replay(self, run):
        with self._run_lock:
            if self.index.is_retired(run):raise ProtocolError('execution evidence retired by retention policy')
            return super().replay(run)

    def _retention_files(self, name):
        """Allowlisted run artifacts only; refuse links, junctions and unknown content."""
        from .model import run_name
        directory=self.root/run_name(name)
        root=self.root.resolve()
        if self.root.is_symlink() or self.root.resolve()!=self.root.absolute():
            raise ValueError('linked viewer root')
        if not directory.exists() and not directory.is_symlink():return []
        files=[]
        pending=[directory]
        while pending:
            path=pending.pop()
            info=path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                raise ValueError('linked artifact path')
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError('unsupported artifact file type')
            if not path.resolve().is_relative_to(root) or path.resolve()==root:
                raise ValueError('artifact outside viewer root')
            rel=path.relative_to(directory).as_posix()
            if path.is_dir():
                if rel not in ('.','layouts'):raise ValueError('unknown artifact directory')
                pending.extend(path.iterdir())
            elif rel not in ('log.bin','identity.json') and not (path.parent==directory/'layouts' and path.suffix=='.json'):
                raise ValueError('unknown artifact file')
            files.append(dict(path=path.relative_to(self.root).as_posix(),bytes=info.st_size if path.is_file() else 0,
                              modified=info.st_mtime_ns,directory=path.is_dir()))
        return files

    def history_preview(self):
        with self._run_lock,self.index.lock:
            status=self.index.history_status();candidates=[];blocked={};files={}
            for name in status['eligible']:
                if self.state.has(name):
                    loaded=self.state.run(name)
                    with loaded.lock:
                        if loaded.clients or loaded.pages or loaded.recording_warning:
                            blocked[name]='live viewer/sender lease or recording warning';continue
                        if any('bound' in p and (not p.get('snapshot_every') or any(s%p['snapshot_every'] for s in p['steps']))
                               for p in loaded.tree.snapshot()['paths'].values()):
                            blocked[name]='unsaved live queue tail';continue
                try:files[name]=self._retention_files(name)
                except (ValueError,OSError) as error:blocked[name]=str(error);continue
                candidates.append(name)
            preview=dict(status,candidates=candidates,blocked=blocked,files=files,
                         policy='explicit preview/apply; protected evidence and page IDs are never removed',
                         log_limit_bytes=self.limits.total_log_bytes,presentation_policy='separate preserve-and-stop store; shared assets are not pruned')
            preview['token']=hashlib.sha256(json.dumps(preview,sort_keys=True).encode()).hexdigest()
            return preview

    def prune_history(self, token):
        with self._run_lock,self.index.lock:
            preview=self.history_preview()
            if not isinstance(token,str) or token!=preview['token']:
                raise ValueError('history changed; obtain a fresh preview before applying')
            names=preview['candidates']
            # Commit tombstones before touching files: an interrupted cleanup cannot let an
            # old URL or sender recreate retired evidence. Metadata deletion is transactional.
            self.index.retire(names)
            errors={}
            for name in names:
                try:
                    if self.state.has(name):
                        loaded=self.state.run(name)
                        if loaded.log is not None:loaded.log.close();loaded.log=None
                        self.state.unregister(name)
                    # Recheck paths immediately before each unlink; never traverse a link.
                    current=self._retention_files(name)
                    for item in sorted(current,key=lambda v:len(v['path']),reverse=True):
                        path=self.root/item['path']
                        if item['directory']:path.rmdir()
                        else:path.unlink()
                except (OSError,ValueError) as error:errors[name]=str(error)
            return dict(removed=names,cleanup_errors=errors,
                        recovery='retired evidence is not recoverable without an external backup; shared presentation assets unchanged')

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
                if url.path not in ("/testing/api/events","/testing/api/pin","/testing/api/history/prune"):
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
                event=json.loads(body)
                if not isinstance(event,dict):raise ValueError('JSON object required')
                with self._run_lock,self.index.lock:
                    if url.path=='/testing/api/pin':self.index.pin(event.get('run'),event.get('pinned'))
                    elif url.path=='/testing/api/history/prune':
                        reply(self.prune_history(event.get('token')));return True
                    else:self.index.accept(event)
                reply({"accepted":True})
            elif request.command=='GET' and url.path.startswith('/testing/media/'):
                from .presentation import read_asset
                try:reply(read_asset(self.presentation_root,url.path.removeprefix('/testing/media/')),mime='image/png')
                except (FileNotFoundError,OSError):reply({'error':'Presentation asset unavailable'},404)
            elif url.path == '/testing/api/portrait-review':
                from .portrait_review import load_review
                try:reply(load_review(self.review_root))
                except FileNotFoundError:reply({'error':'Portrait review not installed'},404)
            elif url.path == "/testing/api/pages": reply({"pages":self.index.pages(), "checkpoint":json.loads((STATIC/"checkpoint.json").read_text(encoding="utf-8"))})
            elif url.path == '/testing/api/directory': reply(self.index.directory(offset=query.get('offset',0),limit=query.get('limit',50)))
            elif url.path == "/testing/api/page":
                reply(self.index.page(query['page'],run=query.get('run'),offset=query.get('offset',0),search=query.get('search',''),outcome=query.get('outcome','')))
            elif url.path == "/testing/api/latest-coverage": reply(self.index.latest_coverage(run=query.get('run'),offset=query.get('offset',0),layer=query.get('layer','python')))
            elif url.path == "/testing/api/history": reply(self.history_preview())
            elif url.path == "/testing/api/runs": reply({"runs":self.index.runs(offset=query.get('offset',0),limit=query.get('limit',100))})
            elif url.path == "/testing/api/run": reply(self.index.summary(query["run"]))
            elif url.path == "/testing/api/tests":
                reply(self.index.tests(query["run"], offset=query.get("offset",0), search=query.get("search",""),outcome=query.get("outcome","")))
            elif url.path == "/testing/api/test": reply(self.index.test(query["run"],query["test"]))
            elif url.path == "/testing/api/coverage": reply(self.index.coverage(query["run"],offset=query.get("offset",0),layer=query.get('layer','python')))
            else:
                name = "index.html" if url.path in ("/testing","/testing/") else url.path.removeprefix("/testing/")
                if name not in ("index.html","app.js","app.css","model.js","playback.js","cinema.js","portrait-review.js"):
                    reply({"error":"not found"},404)
                else:
                    mime = {"html":"text/html; charset=utf-8","js":"text/javascript; charset=utf-8","css":"text/css; charset=utf-8"}[name.rsplit(".",1)[1]]
                    reply((STATIC/name).read_bytes(),mime=mime)
        except KeyError as error: reply({"error":str(error)},404)
        except (ValueError, TypeError, OverflowError, sqlite3.Error, ProtocolError) as error: reply({"error":str(error)},400)
        return True
