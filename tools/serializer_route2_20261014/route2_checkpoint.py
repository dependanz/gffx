"""Exactly gate,one480..495 producer,conditional audit,terminal;never auto retry."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from route2_common import *
from route2_environment import snapshot,quiet,storage_inventory,StorageObserver,win
from route2_admission import before_arrays
from route2_metadata import metadata
from bruteforce_envelope228 import inventory
from capacity_launch322 import admit
from route2_phases import Recorder
def verify_authority(path,p):
    a=read(path)
    if not a.get("explicit_launch_authorization")or a.get("packet_sha256")!=sha(ROOT/"pipeline-packet1.json")or a.get("controller_sha256")!=sha(__file__)or a.get("scope")!="gate+one480-495+conditional-audit+terminal":raise PermissionError("exact separate launch approval required")
    if set(a.get("memory_record_hashes",{}))!={"gffx","gffit"}or any(set(records)!={"PLAN.md","DECISIONS.md","NUMBERS.md","MANIFEST.json"}for records in a["memory_record_hashes"].values()):raise ValueError("both complete authoritative record sets required")
    for project,records in a["memory_record_hashes"].items():
        for name,h in records.items():
            if sha(Path("C:/dev/devbrain/projects")/project/name)!=h:raise ValueError("approved memory changed")
    return a
def run(approval):
    p=packet();verify_authority(approval,p)
    if(ROOT/"pipeline-start1.json").exists()or(ROOT/"pipeline-stopped1.json").exists()or Path(p["stage"]).exists():raise FileExistsError("one fresh checkpoint,no retry")
    started=time.perf_counter();deadline=started+p["limits"]["checkpoint_seconds"]
    active_deadline=deadline-p["limits"]["terminal_reserve_seconds"]
    completed=[];attempted=[];error=None;handles=[];child_envelopes=0.
    def check():
        if time.perf_counter()>active_deadline:raise TimeoutError("checkpoint active deadline;terminal reserve")
        if(LEDGER/"STOP").exists():raise InterruptedError("shared STOP")
    try:
        quiet(psutil_ancestors());used=inventory(LEDGER)
        if used!=read(ROOT/"pipeline-ledger-before1.json")or used["count"]!=5288:raise ValueError("full retained ledger changed")
        initial=snapshot(ROOT);resolve_task(p["tasks"][0],initial,p["limits"])
        seal(ROOT/"initial-resource-admission1.json",{"snapshot":initial,"no_arrays":True,"limits":p["limits"]})
        verify_sources(p,check=check)
        handles=win.lock_files([Path(x)for x in p["sources"]]+[ROOT/"pipeline-packet1.json",Path(approval)]+[Path("C:/dev/devbrain/projects")/pr/n for pr,v in read(approval)["memory_record_hashes"].items()for n in v])
        with Recorder(ROOT/"controller-metrics")as phases:
            with phases.measure("controller_prerequisite_verification"):before_arrays(0,check)
            Path(p["stage"]).mkdir();(ROOT/"scratch").mkdir(exist_ok=True)
            seal(ROOT/"pipeline-start1.json",{"controller_pid":os.getpid(),"packet_sha256":sha(ROOT/"pipeline-packet1.json"),"exactly3_tasks":True,"checkpoint_deadline_counter":deadline})
            for i,contract in enumerate(p["tasks"]):
                check()
                if i:
                    with phases.measure("controller_prerequisite_verification"):before_arrays(i,check)
                quiet(psutil_ancestors());check();fresh=snapshot(ROOT)
                task=resolve_task(contract,fresh,p["limits"]);index=p["base_launch_index"]+i
                full=storage_inventory(QUAL);check()
                evidence=ROOT/f"job-{index}-preflight1.json"
                seal(evidence,{"snapshot":fresh,"used":inventory(LEDGER),"matching_other_heavy_commands":[],"independent_psutil_quiet_excluding_exact_admitted_controller":True,"fresh_CIM_and_psutil":True,"full_permitted_storage":full,"task_contract_sha256":digest(contract)})
                resolution=ROOT/f"job-{index}-resolved-task1.json"
                seal(resolution,{"schema":"route2-fresh-resolution/1","packet_sha256":sha(ROOT/"pipeline-packet1.json"),"task_contract_sha256":digest(contract),"snapshot":fresh,"task":task,"physical_evidence_path":str(evidence),"physical_evidence_sha256":sha(evidence)})
                out=Path(p["stage"])/task["output"];out.parent.mkdir(parents=True,exist_ok=True);seal(str(out)+".spec.json",task["spec"])
                temp=ROOT/"scratch"/("worker-"+str(index));temp.mkdir()
                launch=admit(LEDGER,task["category"],out,ROOT/"pipeline-packet1.json",QUAL,task["spec"],resolution)
                attempted.append(index);tick=time.perf_counter()
                env=dict(os.environ,ROUTE2_AUTHORITY=str(approval),ROUTE2_DEADLINE=str(active_deadline),ROUTE2_CONTROLLER_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE="1",TEMP=str(temp),TMP=str(temp),TMPDIR=str(temp),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONPATH=str(ROOT)+os.pathsep+str(H)+os.pathsep+str(H/"runtime2")+os.pathsep+str(FROZEN))
                rc=None
                try:
                    with Path(str(out)+".out.log").open("xb")as so,Path(str(out)+".err.log").open("xb")as se:
                        rc=subprocess.run([sys.executable,"-B","-X","utf8",str(ROOT/"route2_runtime.py"),"--spec",str(out)+".spec.json","--output",str(out)],env=env,cwd=ROOT,stdout=so,stderr=se,check=False,timeout=max(1,active_deadline-time.perf_counter()+60))
                finally:
                    elapsed=time.perf_counter()-tick;child_envelopes+=elapsed
                    try:data=read(out)
                    except (FileNotFoundError,json.JSONDecodeError,UnicodeDecodeError):
                        from report_journal142 import final_bytes
                        if out.exists():
                            with Path(str(out)+".incomplete-supervisor-report").open("xb")as stream:stream.write(out.read_bytes())
                        data={"status":"stopped","phase":"supervisor-report-refusal","error":"missing/incomplete supervisor report","supervisor":{"elapsed_seconds":elapsed,"exit_code":rc.returncode if rc else-1,"failure":"outer supervisor report missing","actual_worker_identity_matches":False}}
                        with out.open("wb")as stream:stream.write(final_bytes(data))
                        if not Path(str(out)+".live.jsonl").exists():
                            with Path(str(out)+".live.jsonl").open("x")as stream:stream.write(json.dumps({"sequence":0,"status":"working","phase":"supervisor-report-refusal"})+"\n")
                    su=data["supervisor"];elapsed=max(elapsed,su["elapsed_seconds"]);worker=data.get("seconds",su["elapsed_seconds"])
                    if data["status"]!="passed":worker=max(worker,su["elapsed_seconds"])
                    rec={"launch_sha256":sha(launch),"report_sha256":sha(out),"worker_seconds":worker,"supervised_seconds":elapsed,"status":data["status"]}
                    if"seconds"not in data:rec.update(report_worker_seconds_unreported=True,worker_charge_basis="supervisor/outer elapsed,conservative")
                    seal(LEDGER/f"finish-{index:04}.json",rec);seal(str(out)+".receipt.json",rec)
                result=metadata(out,task,index,ROOT/"pipeline-packet1.json",read(launch)["previous"])
                if rc is None or rc.returncode!=0:raise ValueError("first worker refusal")
                check_prior_status(i,result["data"]);completed.append(index)
                seal(ROOT/f"job-{index}-complete1.json",{"status":"passed","report":str(out),"receipt_sha256":result["receipt_sha256"]})
            check();seal(ROOT/"pipeline-complete1.json",{"status":"passed","completed_launches":completed,"used":inventory(LEDGER),"photo_readiness":False})
    except BaseException as exc:
        error=repr(exc)
        seal(ROOT/"pipeline-stopped1.json",{"status":"stopped","error":error,"completed_launches":completed,"attempted_launches":attempted,"failed_launch":attempted[-1]if len(attempted)>len(completed)else None,"used":inventory(LEDGER)})
    terminal_error=None;result=None
    try:
        from route2_terminal import reconcile
        result=reconcile(allow=psutil_ancestors(),deadline=deadline)
    except BaseException as exc:
        terminal_error=repr(exc)
        seal(ROOT/"terminal-reconciliation-refusal.json",{"status":"stopped","error":terminal_error,"accepted_pixels":480,"numerical_acceptance_forbidden":True,"worker_pipeline_error":error,"manual_reconciliation_required":True})
    finally:
        for handle in handles:win.kernel().CloseHandle(handle)
        total=time.perf_counter()-started
        seal(ROOT/"controller-terminal-engineering-receipt.json",{"status":"passed"if error is None and terminal_error is None else"stopped","terminal_error":terminal_error,"total_checkpoint_elapsed_seconds":total,"worker_supervision_envelopes_seconds":child_envelopes,"additional_nonoverlapping_controller_terminal_seconds":max(0,total-child_envelopes),"scientific_worker_charges_remain_in_original_ledger":True,"no_charge_reset":True})
    if terminal_error is not None:
        print(json.dumps({"status":"TERMINAL_REFUSAL_UNACCEPTED","accepted_pixels":480,"error":terminal_error}));return 1
    print(json.dumps({"status":result["status"],"accepted_pixels":result["accepted_pixels"],"error":error}))
    return 0 if error is None else 1

def psutil_ancestors():
    import psutil
    return [p.pid for p in psutil.Process().parents()]
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--approval",type=Path,required=True)
    return run(parser.parse_args().approval)
if __name__=="__main__":raise SystemExit(main())
