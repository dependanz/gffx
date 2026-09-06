"""Repository CLI: serve the app, run serial pytest, or import explicit coverage."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import urllib.parse
import uuid
from .reporter import Reporter
from .model import coverage_report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--root",required=True)
    serve.add_argument("--port",type=int,default=8790)
    serve.add_argument("--also-loopback",action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--address",required=True)
    run.add_argument("--run",default=None)
    run.add_argument("--coverage",action="store_true")
    run.add_argument("--output",default="build/testing-runs")
    run.add_argument("pytest_args",nargs=argparse.REMAINDER)
    cov = sub.add_parser("coverage")
    cov.add_argument("--address",required=True); cov.add_argument("--run",required=True)
    cov.add_argument("report")
    demo = sub.add_parser("demo")
    demo.add_argument("--address", required=True)
    demo.add_argument("--run", default=None)
    args = parser.parse_args(argv)
    if args.command == "serve":
        from .server import TestingServer
        stop = threading.Event()
        for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
        with TestingServer(args.root,port=args.port,also_loopback=args.also_loopback) as server:
            print("gffx testing: http://%s/testing/" % server.address,flush=True)
            while not stop.wait(.5): pass
        return 0
    if args.command == "demo":
        from .demo import publish
        publish(args.address, args.run or "demo-" + uuid.uuid4().hex)
        return 0
    if args.command == "coverage":
        raw = Path(args.report)
        if raw.stat().st_size > 8*1024*1024: parser.error("coverage input exceeds 8 MiB")
        result = coverage_report(json.loads(raw.read_text(encoding="utf-8")))
        result["artifact_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        return 0 if Reporter(args.address,args.run).send("coverage",coverage=result) else 1
    runid = args.run or "pytest-" + uuid.uuid4().hex
    from .model import run_name
    run_name(runid)
    # Never attach a new execution's coverage to a pre-existing run after its start is refused.
    try:
        with urllib.request.urlopen(args.address.rstrip("/")+"/testing/api/run?"+urllib.parse.urlencode({"run":runid}),timeout=1):
            parser.error("run already exists; choose a fresh --run")
    except urllib.error.HTTPError as error:
        if error.code != 404: parser.error("cannot validate run identity: " + str(error))
    except (urllib.error.URLError,OSError):
        pass  # Offline reporting must not prevent the requested tests from running.
    env = {**os.environ,"GFFX_TEST_DASHBOARD":args.address,"GFFX_TEST_RUN":runid}
    env.pop("GFFX_DASHBOARD",None)  # the legacy reporter is not this application's metadata source
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]: pytest_args = pytest_args[1:]
    command = [sys.executable,"-m","pytest","-p","tools.testing_dashboard.pytest_plugin",*pytest_args]
    output = Path(args.output).resolve() / runid
    coverage = args.coverage and importlib.util.find_spec("coverage") is not None
    if args.coverage and not coverage:
        print("coverage.py is not installed: coverage will be NOT MEASURED",file=sys.stderr)
    if coverage:
        output.mkdir(parents=True,exist_ok=False)
        command = [sys.executable,"-m","coverage","run","--branch","--source=gffx",
                   "--data-file",str(output/".coverage"),"-m","pytest",
                   "-p","tools.testing_dashboard.pytest_plugin",*pytest_args]
    print("Testing run:",runid,flush=True)
    result = subprocess.run(command,env=env).returncode
    if coverage:
        report = output/"coverage.json"
        export = subprocess.run([sys.executable,"-m","coverage","json","--data-file",str(output/".coverage"),"-o",str(report)]).returncode
        if export == 0:
            try:
                if report.stat().st_size > 8*1024*1024: raise ValueError("coverage input exceeds 8 MiB")
                value = coverage_report(json.loads(report.read_text()))
                value["artifact_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
                Reporter(args.address,runid).send("coverage",coverage=value)
            except Exception as error:
                print("Coverage unavailable:",error,file=sys.stderr)
        else: print("Coverage export failed; tests retain their original exit status",file=sys.stderr)
    return result

if __name__ == "__main__": raise SystemExit(main())
