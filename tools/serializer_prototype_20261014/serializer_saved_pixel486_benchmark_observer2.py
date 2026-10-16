"""ONE saved-pixel486 serializer engineering subject. No retry or science."""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from owned_storage_observer import bytes_used

GIB = 1024**3
HOST = 4*GIB
DISK = 2*GIB
SECONDS = 600
QUALIFICATION = Path("D:/devspace/gffit/qualification-20261006")
HERE = Path(__file__).resolve().parent
INPUT = QUALIFICATION/"boundary-sh28/gradient136-independent-whole-image16/chunks/chunk-30/pixel-0486.json"
FROZEN = QUALIFICATION/"boundary-sh28/gradient136-source20-checkpoint2"
LEDGER = QUALIFICATION/"boundary-sh27/ledger"
EXPECTED_HASH = "9f4073b6462d164d27d46342fbc3cfb28da751c1aaa80aed3223d5762fe16150"
EXPECTED_BYTES = 184772285
EXPECTED = {"validated_unique_objects":4480683, "validated_container_edges":5412893,
            "serialized_fraction_occurrences":107364}
EXPECTED_SOURCES = {
    str(HERE/"owned_storage_observer.py"):"6c4229846b0f506d84109fafabe3f78b4423196876becedfca24da6e807cde92",
    str(HERE/"serializer_saved_pixel486_benchmark.py"):"b490f384cad18c12486266f91cb7a5e99f80d00aea15d6a20fe696808bc80c8f",
    str(HERE/"disk_exact_json.py"):"56184c56fed001b9f5e7585da10a5bfbc324ceb7540662b930d7bddad023d391",
    str(FROZEN/"streamed_exact_json312.py"):"837bbd17fad7c7382fa6d02fbf5f0148bfe7a6771f10b5132810c5dae72c1a1e",
    str(FROZEN/"duffy_integral271.py"):"dcbb94dfcbf6a133b30f5378c08cb3689c763b8f885036116410d79887c07039",
    str(FROZEN/"pipeline-packet1.json"):"1197ab081060c34eca964d8c28502c9924dfc13289ba5a46ca2afe6db3225398",
    str(FROZEN/"checkpoint-terminal-reconciliation2078.json"):"acaac8e54e84573d8443df003b0b1ada0cdb652682d717217ef353d9cb72501a",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda:source.read(1024**2),b""):
            h.update(block)
    return h.hexdigest()


def seal(path, value):
    with Path(path).open("x",encoding="utf-8",newline="") as out:
        json.dump(value,out,indent=2)


def ledger_snapshot():
    launches = sorted(LEDGER.glob("launch-*.json"),key=lambda p:int(p.stem.split("-")[1]))
    finishes = list(LEDGER.glob("finish-*.json"))
    if len(launches) != len(finishes):
        raise ValueError("retained ledger has an unfinished subject")
    digest = hashlib.sha256()
    worker = supervised = 0.0
    counts = Counter()
    for index,path in enumerate(launches):
        if int(path.stem.split("-")[1]) != index:
            raise ValueError("complete contiguous retained launch inventory required")
        raw = path.read_bytes()
        launch = json.loads(raw)
        finish_path = path.with_name(path.name.replace("launch-","finish-",1))
        finish_raw = finish_path.read_bytes()
        finish = json.loads(finish_raw)
        if launch["index"] != index or finish["launch_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("retained launch/finish identity mismatch")
        for name in ("worker_seconds","supervised_seconds"):
            amount = finish[name]
            if type(amount) not in (int,float) or not math.isfinite(amount) or amount < 0:
                raise ValueError("retained finite nonnegative charge required")
        worker += finish["worker_seconds"]
        supervised += finish["supervised_seconds"]
        counts[launch["category"]] += 1
        for file,content in ((path,raw),(finish_path,finish_raw)):
            digest.update(file.name.encode())
            digest.update(hashlib.sha256(content).digest())
    terminal = json.loads((FROZEN/"checkpoint-terminal-reconciliation2078.json").read_bytes())
    expected = terminal["used"]
    if len(launches) != expected["count"] or dict(counts) != {k:v for k,v in expected["counts"].items() if v}:
        raise ValueError("retained ledger count/category changes require a new scope")
    if not math.isclose(worker,expected["worker_seconds"],rel_tol=0,abs_tol=1e-8):
        raise ValueError("retained worker charges differ from closed checkpoint")
    if not math.isclose(supervised,expected["supervised_seconds"],rel_tol=0,abs_tol=1e-8):
        raise ValueError("retained supervision charges differ from closed checkpoint")
    return {"files":len(launches)+len(finishes),"jobs":len(launches),
            "worker_seconds":worker,"supervised_seconds":supervised,
            "counts":dict(counts),"all_launch_finish_sha256":digest.hexdigest(),
            "last_receipt_sha256":expected["previous"]}


class MemoryStatus(ctypes.Structure):
    _fields_=[("dwLength",wintypes.DWORD),("dwMemoryLoad",wintypes.DWORD),
              ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
              ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
              ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
              ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]


class BasicLimits(ctypes.Structure):
    _fields_=[("PerProcessUserTimeLimit",ctypes.c_longlong),("PerJobUserTimeLimit",ctypes.c_longlong),
              ("LimitFlags",wintypes.DWORD),("MinimumWorkingSetSize",ctypes.c_size_t),
              ("MaximumWorkingSetSize",ctypes.c_size_t),("ActiveProcessLimit",wintypes.DWORD),
              ("Affinity",ctypes.c_size_t),("PriorityClass",wintypes.DWORD),
              ("SchedulingClass",wintypes.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_=[(n,ctypes.c_ulonglong) for n in
              ("ReadOperationCount","WriteOperationCount","OtherOperationCount",
               "ReadTransferCount","WriteTransferCount","OtherTransferCount")]


class ExtendedLimits(ctypes.Structure):
    _fields_=[("BasicLimitInformation",BasicLimits),("IoInfo",IOCounters),
              ("ProcessMemoryLimit",ctypes.c_size_t),("JobMemoryLimit",ctypes.c_size_t),
              ("PeakProcessMemoryUsed",ctypes.c_size_t),("PeakJobMemoryUsed",ctypes.c_size_t)]


_WIN_KERNEL = None

def kernel():
    global _WIN_KERNEL
    if _WIN_KERNEL is None:
        _WIN_KERNEL = ctypes.WinDLL("kernel32",use_last_error=True)
        _WIN_KERNEL.CloseHandle.argtypes=[wintypes.HANDLE]
        _WIN_KERNEL.CloseHandle.restype=wintypes.BOOL
        _WIN_KERNEL.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MemoryStatus)]
        _WIN_KERNEL.GlobalMemoryStatusEx.restype=wintypes.BOOL
    return _WIN_KERNEL


def physical(root):
    import psutil
    memory=MemoryStatus()
    memory.dwLength=ctypes.sizeof(memory)
    if not kernel().GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"total_RAM_bytes":memory.ullTotalPhys,"available_RAM_bytes":memory.ullAvailPhys,
            "available_commit_bytes":memory.ullAvailPageFile,
            "free_disk_bytes":psutil.disk_usage(str(root)).free}


def quiet_processes():
    import psutil
    matches=[]
    denied=[]
    ancestors={parent.pid for parent in psutil.Process(os.getpid()).parents()}
    excluded_launchers=[]
    markers=("run_whole136","whole_pixel_integrals","whole_pixel_group_audit",
             "whole_image_assembly","g4-source20","g4-whole","qualify_source20_checkpoint",
             "benchmark_source20_350","gffx\\tools\\opaque_","gffx/tools/opaque_",
             "serializer_saved_pixel486_benchmark.py",
             "serializer_saved_pixel486_benchmark_observer2.py")
    for process in psutil.process_iter(["pid","name","cmdline"]):
        if process.pid==os.getpid():
            continue
        try:
            name=(process.info["name"] or "").lower()
            command=" ".join(process.info["cmdline"] or [])
            if process.pid in ancestors and "serializer_saved_pixel486_benchmark_observer2.py" in command.lower():
                excluded_launchers.append(process.pid)
                continue
            if name.startswith("python") and process.info["cmdline"] is None:
                denied.append(process.pid)
                continue
            if name.startswith("python") and any(word in command.lower() for word in markers):
                matches.append({"pid":process.pid,"name":name,"command":command})
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            if (process.info["name"] or "").lower().startswith("python"):
                denied.append(process.pid)
    if matches or denied:
        raise ValueError("quiet-process admission refused: "+repr({"matches":matches,"unreadable_python":denied}))
    return {"matching_heavy_commands":[],"unreadable_python":[],"excluded_exact_controller_launcher_pids":excluded_launchers}


def lock_files(paths):
    api=kernel()
    api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
                             wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    api.CreateFileW.restype=wintypes.HANDLE
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handles=[]
    try:
        for path in paths:
            handle=api.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
            if handle==ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
        return handles
    except BaseException:
        for handle in handles:api.CloseHandle(handle)
        raise


def assign_memory_job(process):
    api=kernel()
    api.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
    api.CreateJobObjectW.restype=wintypes.HANDLE
    api.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    api.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    api.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
    handle=api.CreateJobObjectW(None,None)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    limits=ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags=0x100|0x2000
    limits.ProcessMemoryLimit=HOST
    try:
        if not api.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not api.AssignProcessToJobObject(handle,wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
        return handle
    except BaseException:
        api.CloseHandle(handle)
        raise


class Recorder:
    def __init__(self,root):
        self.root=root;self.rows={};self.sequence=0;self.start=time.perf_counter()
    def event(self,kind,label):
        event={"sequence":self.sequence,"event":kind,"phase":label,
               "seconds":time.perf_counter()-self.start,"worker_pid":os.getpid()}
        self.sequence+=1
        with (self.root/"phases.jsonl").open("a",encoding="utf-8") as out:
            out.write(json.dumps(event)+"\n")
    @contextmanager
    def phase(self,label):
        self.event("enter",label);wall=time.perf_counter();cpu=time.process_time()
        try:yield
        finally:
            self.rows[label]={"wall_seconds":time.perf_counter()-wall,
                              "process_CPU_seconds":time.process_time()-cpu}
            self.event("exit",label)


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def worker(root):
    started=time.perf_counter()
    packet=json.loads((root/"packet.json").read_bytes())
    recorder=Recorder(root)
    report={"status":"working","worker_pid":os.getpid(),"graph_decodes":0,
            "implementations":{},"science_acceptance":False,"whole_history_capacity_qualified":False}
    seal(root/"worker-ready.json",{"worker_pid":os.getpid(),"harness_sha256":sha(__file__)})
    while not (root/"GO.json").exists():
        if time.perf_counter()-started>SECONDS:
            raise TimeoutError("one benchmark admission timeout")
        time.sleep(0.01)
    output_total=0
    max_combined=0
    try:
        for path,digest in packet["sources"].items():
            if sha(path)!=digest:raise ValueError("source binding mismatch: "+path)
        with recorder.phase("loading.read_and_json_parse"):
            with INPUT.open("r",encoding="utf-8") as source:
                raw=json.load(source)
        tree=ast.parse((FROZEN/"duffy_integral271.py").read_text(encoding="utf-8"))
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="decode")
        namespace={"Q":Fraction}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(FROZEN/"duffy_integral271.py"),"exec"),namespace)
        with recorder.phase("loading.exact_decode"):
            graph=namespace["decode"](raw)
        del raw,tree
        gc.collect()
        report["graph_decodes"]=1
        report["same_immutable_graph_identity"]=id(graph)
        legacy=import_file("saved486_legacy",FROZEN/"streamed_exact_json312.py")
        prototype=import_file("saved486_prototype",HERE/"disk_exact_json.py")
        limits=prototype.IndexLimits(max_database_bytes=GIB,max_scratch_bytes=1280*1024**2)
        def check_combined():
            nonlocal max_combined
            used=bytes_used(root)
            max_combined=max(max_combined,used)
            if used>DISK:raise OSError("combined output/scratch limit exceeded")
        original_scratch=prototype.DiskHeldIndex._check_scratch
        def checked_scratch(index):
            original_scratch(index)
            check_combined()
        prototype.DiskHeldIndex._check_scratch=checked_scratch
        try:
            for label,module in (("legacy312",legacy),("prototype",prototype)):
                gc.collect()
                row={"graph_identity":id(graph)}
                report["implementations"][label]=row
                old_validate=module.validate_exact_graph
                old_dump=json.dump
                telemetry={}
                def timed_validate(value,**kwargs):
                    if value is not graph:raise ValueError("same held graph required")
                    with recorder.phase(label+".validation"):
                        stats=old_validate(value,**kwargs)
                    row["validation_stats"]=stats
                    return stats
                def timed_dump(*args,**kwargs):
                    with recorder.phase(label+".encoding_write"):
                        return old_dump(*args,**kwargs)
                module.validate_exact_graph=timed_validate
                json.dump=timed_dump
                path=root/(label+".json")
                local_bytes=0
                class BudgetWriter:
                    def __init__(self,out):self.out=out
                    def write(self,text):
                        nonlocal local_bytes
                        count=len(text.encode("utf-8"))
                        # Scratch is gone after preflight; reserve64MiB for every
                        # metadata/observer file as well as both emitted streams.
                        if output_total+local_bytes+count+64*1024**2>DISK:
                            raise OSError("combined output forecast refusal")
                        local_bytes+=count
                        return self.out.write(text)
                try:
                    with path.open("x",encoding="utf-8",newline="") as out:
                        destination=BudgetWriter(out)
                        options={} if label=="legacy312" else {
                            "scratch_root":str(root/"scratch"),"limits":limits,"telemetry":telemetry}
                        stats=module.dump_exact(graph,destination,**options)
                        with recorder.phase(label+".flush_fsync"):
                            out.flush();os.fsync(out.fileno())
                    output_total+=local_bytes
                finally:
                    module.validate_exact_graph=old_validate
                    json.dump=old_dump
                row["serialized_stats"]=stats;row["index_observations"]=telemetry
                if stats!=EXPECTED:raise ValueError("exact original traversal/Fraction counts differ")
                if row["validation_stats"]!={k:EXPECTED[k] for k in row["validation_stats"]}:
                    raise ValueError("validation counters differ")
                check_combined()
                with recorder.phase(label+".readback_hash"):
                    row["readback_bytes"]=path.stat().st_size
                    row["readback_sha256"]=sha(path)
                if row["readback_bytes"]!=EXPECTED_BYTES or row["readback_sha256"]!=EXPECTED_HASH:
                    raise ValueError("original exact output byte count/hash mismatch")
        finally:
            prototype.DiskHeldIndex._check_scratch=original_scratch
        with recorder.phase("readback.exact_three_way_byte_comparison"):
            with INPUT.open("rb") as original,(root/"legacy312.json").open("rb") as a,(root/"prototype.json").open("rb") as b:
                compared=0
                while True:
                    first=original.read(1024**2);left=a.read(1024**2);right=b.read(1024**2)
                    if first!=left or first!=right:raise ValueError("exact output bytes differ")
                    compared+=len(first)
                    if not first:break
            if compared!=EXPECTED_BYTES:raise ValueError("complete exact readback length")
        report.update(status="passed",all_bytes_hashes_and_counts_equal=True,
                      exact_bytes_compared=compared)
    except BaseException as error:
        report.update(status="stopped",error=repr(error),traceback=traceback.format_exc())
    finally:
        report["phases"]=recorder.rows
        report["worker_elapsed_seconds"]=time.perf_counter()-started
        report["peak_combined_bytes_at_worker_boundaries"]=max_combined
        seal(root/"worker-result.json",report)
    return 0 if report["status"]=="passed" else 1


def parent(root):
    import psutil
    started=time.perf_counter()
    if root.exists():raise FileExistsError("one new benchmark directory; no retry")
    root.resolve().relative_to(QUALIFICATION.resolve())
    root.mkdir()
    (root/"scratch").mkdir()
    process=None;job=None;handles=[];refusal=None;samples=0;peaks={};phase_samples={}
    result={"status":"admission","science_acceptance":False,"whole_history_capacity_qualified":False,
            "limits":{"process_bytes":HOST,"seconds":SECONDS,"combined_output_scratch_bytes":DISK},
            "reserves":{"available_RAM_bytes":4*GIB,"free_disk_bytes":GIB},
            "input":{"path":str(INPUT),"bytes":EXPECTED_BYTES,"sha256":EXPECTED_HASH}}
    seal(root/"launch.json",{"scope":"ONE saved-pixel486 serializer engineering benchmark; no retry",
                             "created_at_UTC":datetime.now(timezone.utc).isoformat(),
                             "parent_pid":os.getpid(),"limits":result["limits"]})
    try:
        paths=[Path(path) for path in EXPECTED_SOURCES]+[INPUT,Path(__file__)]
        paths+=sorted(LEDGER.glob("launch-*.json"))+sorted(LEDGER.glob("finish-*.json"))
        handles=lock_files(paths)
        for path,digest in EXPECTED_SOURCES.items():
            if sha(path)!=digest:raise ValueError("frozen source/evidence binding differs: "+path)
        if INPUT.stat().st_size!=EXPECTED_BYTES or sha(INPUT)!=EXPECTED_HASH:
            raise ValueError("preserved input identity differs")
        ledger=ledger_snapshot()
        snapshot=physical(root)
        quiet=quiet_processes()
        if snapshot["available_RAM_bytes"]<HOST+4*GIB:
            raise MemoryError("fresh available RAM plus4GiB reserve refusal")
        if snapshot["available_commit_bytes"]<HOST+4*GIB:
            raise MemoryError("fresh process commit headroom plus4GiB reserve refusal")
        if snapshot["free_disk_bytes"]<=DISK+GIB:
            raise OSError("fresh2GiB combined forecast plus1GiB disk reserve refusal")
        sources=dict(EXPECTED_SOURCES);sources[str(Path(__file__))]=sha(__file__)
        seal(root/"packet.json",{"sources":sources,"input":result["input"],"expected_counts":EXPECTED,
                                 "limits":result["limits"],"immutable_graph":"decode exactly once; retain for both serializers",
                                 "index_limits":{"database_bytes":GIB,"scratch_bytes":1280*1024**2}})
        seal(root/"admission.json",{"snapshot":snapshot,"quiet":quiet,"ledger":ledger,
                                   "packet_sha256":sha(root/"packet.json"),
                                   "read_only_write_delete_excluding_file_locks":len(handles)})
        result["fresh_admission"]={"snapshot":snapshot,"quiet":quiet,"ledger":ledger}
        env=os.environ.copy()
        env["TMP"]=env["TEMP"]=env["TMPDIR"]=str(root/"scratch")
        env["PYTHONDONTWRITEBYTECODE"]="1"
        with (root/"console.log").open("x",encoding="utf-8") as console:
            process=subprocess.Popen([getattr(sys,"_base_executable",sys.executable),"-B",
                                      str(Path(__file__)),"--worker","--root",str(root)],
                                      env=env,cwd=HERE,stdout=console,stderr=subprocess.STDOUT)
            job=assign_memory_job(process)
            monitored=psutil.Process(process.pid)
            result["monitored_pid"]=process.pid
            admitted=False
            observed_storage=0
            while process.poll() is None:
                elapsed=time.perf_counter()-started
                if elapsed>SECONDS:raise TimeoutError("single600-second benchmark timeout")
                try:
                    memory=monitored.memory_info()
                except psutil.NoSuchProcess:
                    if process.poll() is not None:break
                    raise
                values={"working_set_bytes":memory.rss,"private_bytes":getattr(memory,"private",0),
                        "OS_peak_working_set_bytes":getattr(memory,"peak_wset",0),
                        "OS_peak_commit_bytes":getattr(memory,"peak_pagefile",0)}
                samples+=1
                for key,value in values.items():peaks[key]=max(peaks.get(key,0),value)
                if max(values.values())>HOST:raise MemoryError("4GiB process-memory refusal")
                observed_storage=max(observed_storage,bytes_used(root))
                if observed_storage>DISK:raise OSError("2GiB combined output/scratch refusal")
                current=physical(root)
                if current["available_RAM_bytes"]<4*GIB or current["free_disk_bytes"]<GIB:
                    raise OSError("live OS or disk reserve refusal")
                ready=root/"worker-ready.json"
                if not admitted and ready.exists():
                    try:identity=json.loads(ready.read_text())
                    except json.JSONDecodeError:identity=None
                    if identity is not None:
                        if identity["worker_pid"]!=process.pid or identity["harness_sha256"]!=sources[str(Path(__file__))]:
                            raise ValueError("actual Python worker identity mismatch")
                        result["actual_worker_identity_matches"]=True
                        seal(root/"GO.json",{"worker_pid":process.pid,"hard_job_process_commit_bytes":HOST})
                        admitted=True
                journal=root/"phases.jsonl"
                if journal.exists():
                    lines=journal.read_bytes().splitlines(keepends=True)
                    completed=[json.loads(line) for line in lines if line.endswith(b"\n")]
                    if completed and completed[-1]["event"]=="enter":
                        label=completed[-1]["phase"]
                        observation=phase_samples.setdefault(label,{"samples":0,"sampled_peak_working_set_bytes":0,
                                                                    "sampled_peak_private_bytes":0})
                        observation["samples"]+=1
                        observation["sampled_peak_working_set_bytes"]=max(observation["sampled_peak_working_set_bytes"],memory.rss)
                        observation["sampled_peak_private_bytes"]=max(observation["sampled_peak_private_bytes"],getattr(memory,"private",0))
                time.sleep(.025)
            exit_code=process.wait()
        result["exit_code"]=exit_code
        worker_path=root/"worker-result.json"
        if not worker_path.exists():raise ValueError("worker ended without its terminal report")
        child=json.loads(worker_path.read_bytes())
        result["worker"]=child
        if child["worker_pid"]!=process.pid or not result.get("actual_worker_identity_matches"):
            raise ValueError("terminal actual-worker identity mismatch")
        if exit_code!=0 or child["status"]!="passed":
            raise ValueError("first worker refusal: "+str(child.get("error")))
        ledger_after=ledger_snapshot()
        if ledger_after!=ledger:raise ValueError("retained ledger changed during benchmark")
        for path,digest in sources.items():
            if sha(path)!=digest:raise ValueError("source changed during held benchmark")
        if sha(INPUT)!=EXPECTED_HASH:raise ValueError("immutable input changed")
        if time.perf_counter()-started>SECONDS:
            raise TimeoutError("single600-second terminal verification timeout")
        result["retained_ledger_after"]=ledger_after
        result["status"]="passed"
        result["sampled_peak_combined_output_scratch_bytes"]=observed_storage
    except BaseException as error:
        refusal=repr(error)
        result.update(status="stopped",error=refusal,traceback=traceback.format_exc())
        if process is not None and process.poll() is None:
            process.kill();process.wait()
        if (root/"worker-result.json").exists() and "worker" not in result:
            result["worker"]=json.loads((root/"worker-result.json").read_bytes())
    finally:
        if job:
            limits=ExtendedLimits()
            api=kernel()
            if api.QueryInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits),None):
                result["kernel_job_peak_process_commit_bytes"]=limits.PeakProcessMemoryUsed
                if limits.PeakProcessMemoryUsed>HOST:result.update(status="stopped",error="kernel process memory exceeds limit")
            api.CloseHandle(job)
        for handle in handles:kernel().CloseHandle(handle)
        result["observed_memory"]=peaks
        result["phase_memory_samples"]=phase_samples
        result["memory_samples"]=samples
        result["sample_interval_seconds"]=.025
        result["supervised_seconds"]=time.perf_counter()-started
        result["conservative_charged_worker_seconds"]=result["supervised_seconds"]
        result["new_output_scratch_bytes_before_report"]=bytes_used(root)
        result["historical_campaign_ledger_modified"]=False
        result["no_automatic_retry"]=True
        seal(root/"result.json",result)
        seal(root/"engineering-receipt.json",{"status":result["status"],"report_sha256":sha(root/"result.json"),
                                             "conservative_worker_seconds":result["conservative_charged_worker_seconds"],
                                             "supervised_seconds":result["supervised_seconds"],
                                             "historical_campaign_jobs_retained":5288,
                                             "separate_scoped_engineering_charge":True,"retry_permitted":False})
    print(json.dumps({"status":result["status"],"error":result.get("error"),"report":str(root/"result.json"),
                      "supervised_seconds":result["supervised_seconds"],"observed_memory":peaks}))
    return 0 if result["status"]=="passed" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(worker(args.root) if args.worker else parent(args.root))
