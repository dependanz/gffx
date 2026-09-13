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
    serve.add_argument('--history-keep',type=int,default=100,help='Older unprotected executions to retain; not a lifetime execution cap')
    serve.add_argument('--max-active',type=int,default=32)
    serve.add_argument('--metadata-mib',type=int,default=64)
    history = sub.add_parser('history',help='Preview safe history cleanup; apply only an exact preview token')
    history.add_argument('--address',required=True)
    history.add_argument('--apply-token')
    pin = sub.add_parser('pin',help='Protect an execution from history cleanup')
    pin.add_argument('--address',required=True);pin.add_argument('--run',required=True)
    pin.add_argument('--unpin',action='store_true')
    run = sub.add_parser("run")
    run.add_argument("--address",required=True)
    run.add_argument("--run",default=None)
    run.add_argument("--coverage",action="store_true")
    run.add_argument('--coverage-source',action='append',help='Python source package/directory (repeatable); defaults to gffx and repository tooling')
    run.add_argument("--output",default="build/testing-runs")
    run.add_argument("pytest_args",nargs=argparse.REMAINDER)
    cov = sub.add_parser("coverage")
    cov.add_argument("--address",required=True); cov.add_argument("--run",required=True)
    cov.add_argument('--format',choices=['python','lcov'],default='python')
    cov.add_argument('--layer',choices=['python','native','javascript','cuda','shader','other'],default='python')
    cov.add_argument('--inventory-root',help='Attach tracked/nonignored source paths from this Git repository')
    cov.add_argument('--path-map',nargs=2,action='append',default=[],metavar=('FROM','TO'),help='Explicit report-path prefix mapping for installed artifacts')
    cov.add_argument("report")
    demo = sub.add_parser("demo")
    demo.add_argument("--address", required=True)
    demo.add_argument("--run", default=None)
    args = parser.parse_args(argv)
    if args.command in ('history','pin'):
        route='/testing/api/history';data=None
        if args.command=='pin':route='/testing/api/pin';data=dict(run=args.run,pinned=not args.unpin)
        elif args.apply_token:route+='/prune';data=dict(token=args.apply_token)
        request=urllib.request.Request(args.address.rstrip('/')+route,
            data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=30) as response:print(json.dumps(json.load(response),indent=2))
        return 0
    if args.command == "serve":
        from .server import TestingServer
        stop = threading.Event()
        for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
        with TestingServer(args.root,port=args.port,also_loopback=args.also_loopback,
                           history_keep=args.history_keep,max_active=args.max_active,
                           metadata_bytes=args.metadata_mib*1024*1024) as server:
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
        if args.format=='lcov':
            from .coverage_formats import lcov_report
            result=lcov_report(raw.read_text(encoding='utf-8'),layer=args.layer)
        else:
            if args.layer!='python':parser.error('Python JSON only measures the Python layer')
            result = coverage_report(json.loads(raw.read_text(encoding="utf-8")))
        result["artifact_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        for file in result['files']:
            path=file['path'].replace('\\','/')
            for old,new in args.path_map:
                old=old.replace('\\','/').rstrip('/');new=new.replace('\\','/').rstrip('/')
                if path.startswith(old+'/'):path=new+path[len(old):];break
            file['path']=path
        result['path_mappings']=args.path_map
        if args.inventory_root:
            from .coverage_formats import repository_inventory
            result['inventory']=repository_inventory(args.inventory_root)
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
        command = [sys.executable,"-m","coverage","run","--branch","--source="+','.join(args.coverage_source or ['gffx','tools']),
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
                from .coverage_formats import repository_inventory
                value['inventory']=repository_inventory(Path.cwd())
                # Attribute installed-wheel package paths to their repository source paths.
                import gffx
                package=Path(gffx.__file__).resolve().parent
                for file in value['files']:
                    path=Path(file['path']).resolve()
                    if path.is_relative_to(package):file['path']='src/gffx/'+path.relative_to(package).as_posix()
                    elif path.is_relative_to(Path.cwd()):file['path']=path.relative_to(Path.cwd()).as_posix()
                Reporter(args.address,runid).send("coverage",coverage=value)
            except Exception as error:
                print("Coverage unavailable:",error,file=sys.stderr)
        else: print("Coverage export failed; tests retain their original exit status",file=sys.stderr)
    return result

if __name__ == "__main__": raise SystemExit(main())
