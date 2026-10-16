"""Finite preparation tests; actual checkpoint modules are never called."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from route2_common import *
from route2_environment import snapshot,win,StorageObserver
class FocusedStorageObserver(StorageObserver):
    """Only this child's disposable fixture tree is temporary; reports stay strict."""
    def __init__(self,root,*,limit):
        super().__init__(root,limit=limit)
        self.fixture_root=self.root/"case-run/cases"
    def _temporary(self,path):
        return path==self.fixture_root or path.is_relative_to(self.fixture_root)or super()._temporary(path)
def main(root):
    import psutil,shutil
    started=time.perf_counter();root=Path(root).resolve();root.relative_to(QUAL.resolve());root.mkdir()
    (root/"scratch").mkdir();(root/"sources").mkdir()
    sources={str(p):sha(p)for p in ROOT.glob("*.py")}
    for p in ROOT.iterdir():
        if p.is_file():shutil.copyfile(p,root/"sources"/p.name)
    fresh=snapshot(root)
    if fresh["available_RAM_bytes"]<5*GIB or fresh["commit_headroom_bytes"]<5*GIB or fresh["free_disk_bytes"]<GIB+64*1024**2:raise OSError("focused1GiB/120s/64MiB admission with existing reserves")
    seal(root/"launch.json",{"scope":"focused bindings only","snapshot":fresh,"sources":sources,"limits":{"process_bytes":GIB,"seconds":120,"combined_bytes":64*1024**2}})
    env=dict(os.environ,TEMP=str(root/"scratch"),TMP=str(root/"scratch"),TMPDIR=str(root/"scratch"),PYTHONDONTWRITEBYTECODE="1")
    boot="import sys;sys.path="+repr(sys.path)+";sys.executable="+repr(sys.executable)+";sys.prefix="+repr(sys.prefix)+";sys.exec_prefix="+repr(sys.exec_prefix)+";from route2_binding_tests import main;raise SystemExit(main(sys.argv[1]))"
    peak=private=samples=0;failure=None;job=None;process=None
    storage=FocusedStorageObserver(root,limit=64*1024**2)
    try:
        with(root/"console.log").open("x")as log:
            process=subprocess.Popen([getattr(sys,"_base_executable",sys.executable),"-B","-c",boot,str(root/"case-run")],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            win.HOST=GIB;job=win.assign_memory_job(process);observed=psutil.Process(process.pid)
            while process.poll()is None:
                if time.perf_counter()-started>120:raise TimeoutError("focused deadline")
                try:m=observed.memory_info()
                except psutil.NoSuchProcess:
                    if process.poll()is not None:break
                    raise
                peak=max(peak,m.rss);private=max(private,getattr(m,"private",0));samples+=1
                if max(peak,private)>GIB:raise MemoryError("focused memory")
                storage.bytes_used()
                time.sleep(.01)
            process.wait()
    except BaseException as error:
        failure=repr(error)
        if process is not None and process.poll()is None:process.kill();process.wait()
    finally:
        import ctypes
        job_peak=None
        if job:
            q=win.ExtendedLimits()
            if win.kernel().QueryInformationJobObject(job,9,ctypes.byref(q),ctypes.sizeof(q),None):job_peak=q.PeakProcessMemoryUsed
            else:failure=failure or"OS peak unavailable"
            win.kernel().CloseHandle(job)
        child=read(root/"case-run/result.json")if(root/"case-run/result.json").exists()else None
        matched=bool(child and process and child["worker_pid"]==process.pid)
        status="passed"if child and child["status"]=="passed"and matched and not failure and process.returncode==0 else"failed"
        elapsed=time.perf_counter()-started
        result={"status":status,"error":failure,"child":child,"actual_worker_identity_matches":matched,"observed_pid":process.pid if process else None,"peak_working_set_bytes":peak,"peak_private_bytes":private,"kernel_job_peak_process_commit_bytes":job_peak,"samples":samples,"source_bindings_unchanged":all(sha(p)==h for p,h in sources.items()),"conservative_worker_seconds":elapsed,"supervised_seconds":elapsed,"no_checkpoint_or_numerical_main_run":True}
        seal(root/"focused-results.json",result);seal(root/"engineering-receipt.json",{"status":status,"result_sha256":sha(root/"focused-results.json"),"conservative_worker_seconds":elapsed,"supervised_seconds":elapsed,"historical_campaign_ledger_modified":False})
        print(json.dumps({k:result[k]for k in("status","error","actual_worker_identity_matches","conservative_worker_seconds","peak_working_set_bytes")}))
    return 0 if status=="passed"else 1
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True);raise SystemExit(main(parser.parse_args().root))
