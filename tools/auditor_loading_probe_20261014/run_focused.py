"""Bounded small development tests; does not create or consume the actual probe."""
import argparse,ctypes,os,subprocess,sys,time
from probe_core import *
def main(root):
    import psutil
    root=Path(root);root.relative_to(QUAL);root.mkdir();(root/'scratch/fixtures').mkdir(parents=True);(root/'sources').mkdir()
    start=time.perf_counter();sources={str(p):sha(p)for p in CODE.iterdir()if p.is_file()}
    for p in CODE.iterdir():
        if p.is_file():shutil.copyfile(p,root/'sources'/p.name)
    fresh=physical(root);admit_snapshot(fresh,limit=GIB);seal(root/'admission.json',{'fresh':fresh,'limits':{'process_bytes':GIB,'seconds':120,'report_bytes':64*MIB}})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMP=str(root/'scratch'),TEMP=str(root/'scratch'))
    boot='import sys;sys.path='+repr(sys.path)+';sys.executable='+repr(sys.executable)+';sys.prefix='+repr(sys.prefix)+';sys.exec_prefix='+repr(sys.exec_prefix)+';from probe_tests import main;raise SystemExit(main(sys.argv[1]))'
    process=job=None;failure=None;peak=private=0
    storage=Storage(root,limit=64*MIB)
    try:
        with(root/'console.log').open('xb')as log:
            process=subprocess.Popen([getattr(sys,'_base_executable',sys.executable),'-B','-c',boot,str(root/'scratch/fixtures/case-run')],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
            win.HOST=GIB;job=win.assign_memory_job(process);observed=psutil.Process(process.pid)
            while process.poll()is None:
                if time.perf_counter()-start>120:raise TimeoutError('focused120s ceiling')
                try:m=observed.memory_info()
                except psutil.NoSuchProcess:
                    if process.poll()is not None:break
                    raise
                peak=max(peak,m.rss);private=max(private,m.private);storage.bytes_used()
                if max(peak,private)>GIB:raise MemoryError('focused1GiB ceiling')
                time.sleep(.025)
            process.wait()
    except BaseException as error:
        failure=repr(error)
        if process is not None and process.poll()is None:process.kill();process.wait()
    finally:
        job_peak=None
        if job:
            info=win.ExtendedLimits()
            if win.kernel().QueryInformationJobObject(job,9,ctypes.byref(info),ctypes.sizeof(info),None):job_peak=info.PeakProcessMemoryUsed
            win.kernel().CloseHandle(job)
        child=read(root/'scratch/fixtures/case-run/test-result.json')if(root/'scratch/fixtures/case-run/test-result.json').exists()else None
        status='passed'if child and child['status']=='passed'and child['worker_pid']==process.pid and process.returncode==0 and not failure else'failed'
        elapsed=time.perf_counter()-start;result={'status':status,'error':failure,'child':child,'sampled_peak_RSS_bytes':peak,'sampled_peak_private_commit_bytes':private,'OSJob_peak_commit_bytes':job_peak,'source_bindings_unchanged':all(sha(p)==h for p,h in sources.items()),'conservative_worker_seconds':elapsed,'supervised_seconds':elapsed,'no_actual_probe':True}
        seal(root/'result.json',result);seal(root/'engineering-receipt.json',{'status':status,'result_sha256':sha(root/'result.json'),'worker_seconds':elapsed,'supervised_seconds':elapsed,'historical_campaign_ledger_modified':False});print(json.dumps({k:result[k]for k in('status','error','conservative_worker_seconds')}))
    return 0 if status=='passed'else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);raise SystemExit(main(parser.parse_args().root))
