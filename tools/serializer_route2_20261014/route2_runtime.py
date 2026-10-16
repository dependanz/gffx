"""Bounded actual-worker supervisor. Numerical subjects require launch authority."""
import argparse,importlib,os,subprocess,sys,time,traceback
from pathlib import Path
from route2_common import *
from route2_environment import win,snapshot,StorageObserver
from report_journal142 import final_bytes
def worker(args):
    from tools import material_loss_cost as cost
    spec=read(args.spec);p=packet();limits=p["limits"];started=time.perf_counter()
    cost.PROFILE.update(cooperative_seconds=spec["seconds"],watchdog_seconds=spec["watchdog_seconds"],sampled_working_set_limit_bytes=spec["host_bytes"])
    deadline=float(os.environ["ROUTE2_DEADLINE"]);old=cost.check_deadline
    def bounded(value):
        if(LEDGER/"STOP").exists():raise InterruptedError("shared campaign STOP")
        old(min(value,started+spec["seconds"],deadline))
        from route2_phases import current
        if current()is not None:current().check()
    cost.check_deadline=bounded
    ready=Path(str(args.output)+".worker-ready.json");go=Path(str(args.output)+".GO.json")
    seal(ready,{"worker_pid":os.getpid(),"runtime_sha256":sha(__file__)})
    while not go.exists():bounded(deadline);time.sleep(.01)
    try:
        sys.argv=[spec["module"],"--worker","--output",str(args.output),*spec["args"]]
        return importlib.import_module(spec["module"]).main()or 0
    except BaseException:
        traceback.print_exc();raise
def supervise(args):
    import psutil
    from tools import material_loss_cost as cost
    p=packet();spec=read(args.spec)
    if not os.environ.get("ROUTE2_AUTHORITY"):raise PermissionError("explicit launch authority required")
    if args.output.exists():raise FileExistsError("new subject only")
    from route2_checkpoint import verify_authority
    verify_authority(Path(os.environ["ROUTE2_AUTHORITY"]),p)
    from route2_metadata import task_for_index
    index=int(spec["args"][-1]);task=task_for_index(index-p["base_launch_index"],p)
    if task["spec"]!=spec or args.output!=Path(p["stage"])/task["output"]:raise ValueError("runtime exact task/output binding")
    limits=p["limits"];process=None;job=None;failure=None;peak=private=commit=0;samples=0;matched=False
    start=time.perf_counter();deadline=min(float(os.environ["ROUTE2_DEADLINE"]),start+spec["watchdog_seconds"])
    reports=[Path(p["stage"])/t["output"]for t in p["tasks"]]
    disk=StorageObserver(ROOT,limit=limits["combined_bytes"],report_paths=reports)
    cmd=cost.worker_command(ROOT,args.output,worker_module="route2_runtime",worker_args=["--spec",str(args.spec)])
    try:
        process=subprocess.Popen(cmd,env=os.environ.copy())
        win.HOST=spec["host_bytes"];job=win.assign_memory_job(process);monitored=psutil.Process(process.pid)
        last=time.perf_counter()
        while process.poll()is None:
            now=time.perf_counter()
            if now>deadline:raise TimeoutError("subject/checkpoint deadline")
            if now-last>2:raise RuntimeError("resource observer coverage gap")
            last=now
            try:m=monitored.memory_info()
            except psutil.NoSuchProcess:
                if process.poll()is not None:break
                raise
            peak=max(peak,m.rss);private=max(private,getattr(m,"private",0));commit=max(commit,getattr(m,"peak_pagefile",0));samples+=1
            if max(peak,private,commit)>spec["host_bytes"]:raise MemoryError("bounded process memory")
            disk.bytes_used();s=snapshot(ROOT)
            if s["available_RAM_bytes"]<limits["RAM_reserve_bytes"]or s["commit_headroom_bytes"]<limits["commit_reserve_bytes"]or s["free_disk_bytes"]<limits["disk_reserve_bytes"]:raise OSError("live RAM/commit/disk reserve")
            ready=Path(str(args.output)+".worker-ready.json")
            if not matched and ready.exists():
                try:r=read(ready)
                except json.JSONDecodeError:r=None
                if r is not None:
                    if r["worker_pid"]!=process.pid or r["runtime_sha256"]!=sha(__file__):raise ValueError("actual worker identity")
                    seal(str(args.output)+".GO.json",{"worker_pid":process.pid});matched=True
            time.sleep(.05)
        process.wait()
    except BaseException as error:
        failure=repr(error)
        if process is not None and process.poll()is None:process.kill();process.wait()
    finally:
        job_peak=0
        if job:
            import ctypes
            q=win.ExtendedLimits()
            if not win.kernel().QueryInformationJobObject(job,9,ctypes.byref(q),ctypes.sizeof(q),None):failure=failure or"OS peak unavailable"
            else:job_peak=q.PeakProcessMemoryUsed
            win.kernel().CloseHandle(job)
        if job_peak>spec["host_bytes"]:failure=failure or"OS process commit limit exceeded"
        raw=args.output.read_bytes()if args.output.exists()else None
        if raw is not None:
            with Path(str(args.output)+".worker-final.json").open("xb")as out:out.write(raw)
        try:data=json.loads(raw)if raw is not None else None
        except (json.JSONDecodeError,UnicodeDecodeError):data=None
        if data is None:
            failure=failure or"missing/incomplete worker terminal report"
            data={"status":"stopped","phase":"abrupt-worker-stop","error":failure}
        if not failure and(data.get("worker_pid")!=getattr(process,"pid",None)or not matched):failure="actual worker identity mismatch"
        if failure or process is None or process.returncode:data["status"]="stopped"
        data["supervisor"]={"elapsed_seconds":time.perf_counter()-start,"monitored_pid":process.pid if process else None,"sampled_peak_working_set_bytes":peak,"sampled_peak_private_bytes":private,"OS_peak_commit_bytes":commit,"kernel_job_peak_process_commit_bytes":job_peak,"sample_interval_seconds":.05,"memory_samples":samples,"actual_worker_identity_matches":matched,"includes":"whole worker,retained input and runtime;not isolated serializer overhead","exit_code":process.returncode if process else-1,"failure":failure}
        # Preserve the immutable child final report before the supervisor adds observations.
        if not Path(str(args.output)+".live.jsonl").exists():
            with Path(str(args.output)+".live.jsonl").open("x")as out:out.write(json.dumps({"sequence":0,"status":"working","phase":"worker-not-entered","worker_pid":getattr(process,"pid",None)})+"\n")
        with args.output.open("wb")as out:out.write(final_bytes(data))
        return 0 if data["status"]=="passed"and failure is None else 1
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--worker",action="store_true");parser.add_argument("--fixtures");parser.add_argument("--output",type=Path,required=True);parser.add_argument("--spec",type=Path,required=True)
    args=parser.parse_args();return worker(args)if args.worker else supervise(args)
if __name__=="__main__":raise SystemExit(main())
