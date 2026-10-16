"""One isolated saved-pixel491 subject; no checkpoint or mathematical audit."""
import argparse,ctypes,json,os,subprocess,sys,time
from pathlib import Path
from probe_core import *
def files(root):
    rows=[]
    def fail(error):raise error
    for directory,dirs,names in os.walk(root,onerror=fail):
        for name in names:
            path=Path(directory)/name
            if path.is_symlink():raise OSError('source/evidence link refused')
            rows.append(path)
    return rows
def sources():
    paths=files(FROZEN)+files(CHECKPOINT)+files(PROTO)
    paths+=[Path('C:/dev/gffx/tools/opaque_shading_intervals.py'),Path('C:/dev/gffx/tools/material_loss_cost.py')]
    paths+=[p for p in CODE.iterdir()if p.is_file()]
    return {str(p):sha(p)for p in paths}
class PhaseObserver:
    """Cross-process samples remain available while the parser holds the GIL."""
    def __init__(self,root):self.path=Path(root)/'phase-events.jsonl';self.sequence=0;self.stack=[];self.rows={}
    def poll(self):
        if not self.path.exists():return
        raw=self.path.read_bytes()
        if len(raw)>MIB:raise OSError('phase journal scalar bound')
        lines=raw.split(b'\n')[:-1]  # Only complete append-only rows.
        for line in lines[self.sequence:]:
            event=json.loads(line)
            if event['sequence']!=self.sequence:raise ValueError('phase event sequence')
            self.sequence+=1;label=event['phase']
            if event['event']=='enter':self.stack.append(label)
            elif event['event']=='exit':
                if not self.stack or self.stack.pop()!=label:raise ValueError('phase event nesting')
    def sample(self,m):
        for label in self.stack:
            row=self.rows.setdefault(label,{'samples':0,'sampled_peak_RSS_bytes':0,'sampled_peak_private_commit_bytes':0,'observed_OS_lifetime_peak_RSS_bytes':0,'observed_OS_lifetime_peak_commit_bytes':0})
            row['samples']+=1;row['sampled_peak_RSS_bytes']=max(row['sampled_peak_RSS_bytes'],m.rss);row['sampled_peak_private_commit_bytes']=max(row['sampled_peak_private_commit_bytes'],m.private)
            row['observed_OS_lifetime_peak_RSS_bytes']=max(row['observed_OS_lifetime_peak_RSS_bytes'],m.peak_wset);row['observed_OS_lifetime_peak_commit_bytes']=max(row['observed_OS_lifetime_peak_commit_bytes'],m.peak_pagefile)
def supervise(spec):
    import psutil
    from tools import material_loss_cost as cost
    root=Path(spec['root']);started=time.perf_counter();deadline=started+spec['seconds']
    seal(root/'worker-spec.json',spec)
    storage=Storage(root,limit=spec['report_limit_bytes']);phase_observer=PhaseObserver(root)
    process=job=None;failure=None;matched=False;final_seen=False;peak=private=ospeak=0;samples=0;child=None
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TEMP=str(root/'scratch/owned-worker'),TMP=str(root/'scratch/owned-worker'),TMPDIR=str(root/'scratch/owned-worker'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    (root/'scratch/owned-worker').mkdir(parents=True)
    command=cost.worker_command(CODE,root/'unused-output.json',worker_module='probe_worker',worker_args=['--spec',str(root/'worker-spec.json')])
    try:
        with(root/'console.log').open('xb')as out:
            process=subprocess.Popen(command,env=env,cwd=CODE,stdout=out,stderr=subprocess.STDOUT)
            win.HOST=spec['process_bytes'];job=win.assign_memory_job(process);observed=psutil.Process(process.pid);last=time.perf_counter()
            while process.poll()is None:
                now=time.perf_counter()
                if now>deadline:raise TimeoutError('900-second isolated probe deadline')
                if now-last>2:raise RuntimeError('resource observer coverage gap')
                last=now
                phase_observer.poll()
                try:m=observed.memory_info()
                except psutil.NoSuchProcess:
                    if process.poll()is not None:break
                    raise
                phase_observer.sample(m)
                peak=max(peak,m.rss);private=max(private,m.private);ospeak=max(ospeak,m.peak_pagefile);samples+=1
                if max(peak,private,ospeak)>spec['process_bytes']:raise MemoryError('20GiB process ceiling')
                storage.bytes_used();fresh=physical(root)
                if fresh['available_RAM_bytes']<4*GIB or fresh['commit_headroom_bytes']<4*GIB or fresh['D_free_bytes']<GIB:raise OSError('live RAM/commit/disk reserve')
                ready=root/'worker-ready.json'
                if not matched and ready.exists():
                    try:r=read(ready)
                    except json.JSONDecodeError:r=None
                    if r is not None:
                        if r['worker_pid']!=process.pid or r['worker_sha256']!=sha(CODE/'probe_worker.py'):raise ValueError('actual worker identity')
                        seal(root/'GO.json',{'worker_pid':process.pid});matched=True
                result=root/'worker-result.json'
                if matched and not final_seen and result.exists():
                    try:child=read(result)
                    except json.JSONDecodeError:child=None
                    if child is not None:
                        if child['worker_pid']!=process.pid:raise ValueError('terminal worker identity')
                        if child['status']=='passed':
                            if not child['retained_record_and_owner_alive']or child['consumed_input_sha256']!=spec['input_sha256']or child['mathematical_audit_operations']!=0 or child['serialization_calls']!=0:raise ValueError('exact isolated retained-state result')
                            # One last actual process measurement with graph and owner still live.
                            live=observed.memory_info()
                            seal(root/'measurement-complete.json',{'worker_pid':process.pid,'worker_result_sha256':sha(result),'observed_live_RSS_bytes':live.rss,'observed_live_private_commit_bytes':live.private,'OS_lifetime_peak_RSS_bytes':live.peak_wset,'OS_lifetime_peak_commit_bytes':live.peak_pagefile})
                        final_seen=True
                time.sleep(.025)
            process.wait()
    except BaseException as error:
        failure=repr(error)
        if process is not None and process.poll()is None:process.kill();process.wait()
    finally:
        job_peak=None
        if job:
            limits=win.ExtendedLimits()
            if not win.kernel().QueryInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits),None):failure=failure or 'OS Job peak unavailable'
            else:job_peak=limits.PeakProcessMemoryUsed
            win.kernel().CloseHandle(job)
        try:child=read(root/'worker-result.json')
        except (FileNotFoundError,json.JSONDecodeError):child=None
        if job_peak is not None and job_peak>spec['process_bytes']:failure=failure or 'OS process ceiling exceeded'
        elapsed=time.perf_counter()-started
        passed=bool(child and child['status']=='passed'and matched and final_seen and failure is None and process.returncode==0)
        return {'status':'passed'if passed else'failed','error':failure,'child':child,'external_phase_memory_samples':phase_observer.rows,'external_phase_sample_attribution':'latest complete enter/exit events;transition uncertainty up to one25ms sample interval','actual_worker_identity_matches':matched,'exit_code':process.returncode if process else None,'sampled_peak_RSS_bytes':peak,'sampled_peak_private_commit_bytes':private,'OS_peak_commit_bytes':ospeak,'kernel_job_peak_process_commit_bytes':job_peak,'samples':samples,'sample_interval_seconds':.025,'worker_supervision_seconds':elapsed,'conservative_worker_seconds':max(elapsed,child['elapsed_seconds']if child else 0),'supervised_seconds':elapsed,'result_path':str(root/'worker-result.json'),'measurement_acknowledged_with_state_alive':bool(child and(root/'measurement-complete.json').exists())}
def actual():
    started=time.perf_counter();RUN.mkdir();status='refused';error=None;handles=[];result=None;baseline=None;campaign=None;launched=False
    try:
        seal(RUN/'attempt-start.json',{'scope':'ONE saved pixel491 exact production loading/decode/OwnedWork.hold probe','timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limits':{'process_bytes':20*GIB,'seconds':900,'report_log_scratch_bytes':64*MIB},'checkpoint_launch_authorization':False})
        quiet();admit_snapshot(physical(RUN))
        baseline=sources();handles=win.lock_files([Path(p)for p in baseline]+[INPUT])
        if INPUT.stat().st_size!=1663807861:raise ValueError('saved Source20 pixel491 byte length')
        input_sha=sha(INPUT)
        campaign=win.ledger_snapshot()
        fresh=physical(RUN);admit_snapshot(fresh);quiet()
        if time.perf_counter()-started>=840:raise TimeoutError('admission consumes active deadline;60s reconciliation reserve')
        seal(RUN/'preservation-before.json',{'files':baseline,'input':str(INPUT),'input_bytes':1663807861,'input_sha256':input_sha,'campaign':campaign,'input_hash_is_newly_pinned_before_probe':True})
        seal(RUN/'admission.json',{'fresh':fresh,'required_available_RAM_bytes':24*GIB,'required_commit_headroom_bytes':24*GIB,'required_D_free_strictly_greater_than_bytes':GIB+64*MIB,'quiet':True})
        for p in CODE.iterdir():
            if p.is_file():shutil.copyfile(p,RUN/p.name)
        spec={'root':str(RUN),'input':str(INPUT),'input_sha256':input_sha,'pixel':491,'process_bytes':20*GIB,'seconds':840-(time.perf_counter()-started),'report_limit_bytes':64*MIB,'source_bindings':baseline,'actual_subject':True}
        launched=True;result=supervise(spec);status=result['status']
    except BaseException as exc:error=repr(exc);status='refused'if not launched else'failed'
    finally:
        preserved=False;after_campaign=None
        try:
            if baseline:
                preserved=all(sha(p)==h for p,h in baseline.items())
                if not preserved:raise ValueError('protected source/checkpoint evidence changed')
            if campaign:
                after_campaign=win.ledger_snapshot()
                if after_campaign!=campaign:raise ValueError('historical campaign/charges changed')
        except BaseException as exc:error=error or repr(exc);status='failed'
        for handle in handles:win.kernel().CloseHandle(handle)
        elapsed=time.perf_counter()-started
        if elapsed>900:status='failed';error=error or '900-second total deadline exceeded'
        final={'status':status,'error':error,'launched_actual_probe':launched,'probe':result,'protected_sources_checkpoint_and_serializers_unchanged':preserved,'historical_campaign_unchanged':after_campaign==campaign if campaign else None,'campaign_after':after_campaign,'total_supervised_seconds':elapsed,'conservative_worker_seconds':result['conservative_worker_seconds']if result else 0,'additional_nonoverlapping_admission_reconciliation_seconds':max(0,elapsed-(result['worker_supervision_seconds']if result else 0)),'whole_history_capacity_qualified':False,'scientific_acceptance':False,'accepted_pixels':480,'remaining_pixels':544,'no_automatic_retry':True}
        seal(RUN/'result.json',final);seal(RUN/'engineering-receipt.json',{'status':status,'result_sha256':sha(RUN/'result.json'),'actual_worker_charge_seconds':final['conservative_worker_seconds'],'total_supervised_seconds':elapsed,'additional_nonoverlapping_admission_reconciliation_seconds':final['additional_nonoverlapping_admission_reconciliation_seconds'],'historical_campaign_ledger_modified':False})
    print(json.dumps({'status':status,'error':error,'launched':launched,'root':str(RUN),'elapsed_seconds':elapsed}));return 0 if status=='passed'else 1
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--actual',action='store_true');args=parser.parse_args()
    if not args.actual:raise PermissionError('one explicitly authorized actual subject only')
    return actual()
if __name__=='__main__':raise SystemExit(main())
