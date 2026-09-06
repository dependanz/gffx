"""Opt-in serial pytest bridge for the separate testing application."""
import os
import subprocess
import threading
import uuid
import time
from urllib.parse import urlsplit
import pytest
from .model import test_id
from .reporter import Reporter

class Session:
    def __init__(self, config):
        self.reporter = None; self.viewer = None; self.config = config
        self.stop = threading.Event(); self.thread = None
        address = os.environ.get("GFFX_TEST_DASHBOARD")
        if not address: return
        run = os.environ.get("GFFX_TEST_RUN") or "pytest-" + uuid.uuid4().hex
        self.reporter = Reporter(address, run)
        provenance = {}
        try:
            revision = subprocess.check_output(["git","rev-parse","HEAD"], cwd=config.rootpath, timeout=2, stderr=subprocess.DEVNULL, text=True).strip()
            dirty = bool(subprocess.check_output(["git","status","--porcelain"], cwd=config.rootpath, timeout=2, stderr=subprocess.DEVNULL))
            provenance = {"revision":revision,"dirty":dirty}
        except (OSError, subprocess.SubprocessError): pass
        if not self.reporter.send("start", kind="tests", provenance=provenance): return
        if hasattr(config, "workerinput") or getattr(config.option, "numprocesses", None):
            self.reporter.send("diagnostic", message="Parallel pytest reporting is unsupported; use a serial run.")
            self.reporter.send("finish", exit_code=3)
            self.reporter.failed = True
            return
        self.thread = threading.Thread(target=self.heartbeat, daemon=True, name="gffx-test-heartbeat")
        self.thread.start()

    def heartbeat(self):
        while not self.stop.wait(5):
            if self.reporter.failed: break
            self.reporter.send("heartbeat")

    def send(self, event, **values):
        return self.reporter is not None and self.reporter.send(event, **values)

    def close(self):
        self.stop.set()
        if self.thread: self.thread.join(timeout=2)
        if self.viewer:
            try: self.viewer.close()
            except Exception: pass
            self.viewer = None

    def evidence(self, nodeid, kind, name, *args, check=None, **kwargs):
        if self.reporter is None or self.reporter.failed: return
        try:
            from gffx.dashboard import connect
            if self.viewer is None:
                u = urlsplit(self.reporter.address)
                port = u.port or (443 if u.scheme == "https" else 80)
                self.viewer = connect("%s://%s:%d" % (u.scheme, u.hostname, port),
                    run=self.reporter.run, name="pytest evidence", wait_acks=True)
            path = "tests/" + test_id(nodeid) + ("/checks/"+check if check else "") + "/" + test_id(str(name))
            getattr(self.viewer, kind)(path, *args, **kwargs)
            panel = {"scalar":"plot","points":"mesh"}.get(kind,kind)
            self.send("artifact", nodeid=nodeid, artifact={"kind":panel,"path":path,"label":str(name)[:200],**({'check':check} if check else {})})
        except Exception as error:
            self.send("diagnostic",message="Evidence unavailable: " + str(error))

_session = None

def pytest_configure(config):
    global _session
    _session = Session(config)

def pytest_collection_finish(session):
    names = [item.nodeid for item in session.items]
    for i in range(0,len(names),100):
        _session.send("collect",nodeids=names[i:i+100])

def pytest_collectreport(report):
    if _session is not None and report.failed:
        _session.send("diagnostic", message="Collection failure: " + str(report.longrepr)[-3500:])

def pytest_runtest_logreport(report):
    if _session is None: return
    outcome = report.outcome
    if hasattr(report,"wasxfail"): outcome = "xfailed" if report.skipped else "xpassed"
    # Strict XPASS is a failed report without wasxfail; it still correctly fails the run.
    _session.send("report",nodeid=report.nodeid,when=report.when,outcome=outcome,
                  duration=float(report.duration),message=str(report.longrepr or "")[-4000:])

def pytest_sessionfinish(session, exitstatus):
    _session.send("finish",exit_code=int(exitstatus))
    _session.close()

def pytest_unconfigure(config):
    if _session is not None: _session.close()

class Evidence:
    def __init__(self,nodeid,check_id=None): self.nodeid = nodeid;self.check_id=check_id
    def _send(self,kind,name,*args,**kwargs):
        if _session: _session.evidence(self.nodeid,kind,name,*args,check=self.check_id,**kwargs)
    def check(self,name):
        if self.check_id: raise ValueError('Nested checks are not supported; use sibling named checks')
        return NamedCheck(self.nodeid,name)
    def scalar(self,name,value,**kwargs): self._send("scalar",name,value,**kwargs)
    def text(self,name,value,**kwargs): self._send("text",name,value,**kwargs)
    def record(self,name,value,**kwargs): self._send("record",name,value,**kwargs)
    def image(self,name,value,**kwargs): self._send("image",name,value,**kwargs)
    def mesh(self,name,vertices,faces,**kwargs): self._send("mesh",name,vertices,faces,**kwargs)
    def points(self,name,positions,**kwargs): self._send("points",name,positions,**kwargs)

class NamedCheck(Evidence):
    """Explicit assertion scope. Reporting never suppresses a test's exception."""
    def __init__(self,nodeid,name):
        from .model import plain
        self.name=plain(name,200);super().__init__(nodeid,test_id(self.name))
    def __enter__(self):
        self.started=time.perf_counter()
        if _session:_session.send('check_start',nodeid=self.nodeid,name=self.name)
        return self
    def __exit__(self,kind,error,traceback):
        outcome=('passed' if kind is None else 'skipped' if isinstance(error,pytest.skip.Exception)
                 else 'interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed')
        if _session:_session.send('check_finish',nodeid=self.nodeid,name=self.name,outcome=outcome,
                                 duration=time.perf_counter()-self.started,message=str(error or '')[:4000])
        return False

@pytest.fixture
def dashboard(request):
    return Evidence(request.node.nodeid)
