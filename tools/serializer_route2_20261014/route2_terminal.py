"""Independent terminal reconciliation; no arrays,fixtures or numerical replay."""
from route2_common import *
from route2_environment import quiet,storage_inventory
from route2_metadata import metadata,task_for_index
from route2_admission import before_arrays
from route2_phases import Recorder
from bruteforce_envelope228 import inventory
def reconcile(allow=(),deadline=None):
    def check():
        if deadline is not None and __import__("time").perf_counter()>deadline:raise TimeoutError("terminal reserve exhausted")
    check();quiet(allow);p=packet();verify_sources(p,check=check);accepted=480
    complete=ROOT/"pipeline-complete1.json";stop=ROOT/"pipeline-stopped1.json"
    if complete.exists()==stop.exists():raise ValueError("exactly one terminal state")
    terminal=read(complete if complete.exists()else stop)
    attempted=terminal.get("attempted_launches",terminal["completed_launches"])
    previous=read(ROOT/"pipeline-ledger-before1.json")["previous"];actual=[]
    with Recorder(ROOT/"terminal-metrics")as phases:
        with phases.measure("terminal_historical_prerequisites"):before_arrays(0,check)
        with phases.measure("terminal_new_typed_chain_and_artifacts"):
            for index in attempted:
                i=index-p["base_launch_index"];task=task_for_index(i,p);out=Path(p["stage"])/task["output"]
                try:a=metadata(out,task,index,ROOT/"pipeline-packet1.json",previous)
                except BaseException as error:
                    if index in terminal["completed_launches"]:raise
                    from route2_metadata import failure_cost_metadata
                    a=failure_cost_metadata(out,task,index,ROOT/"pipeline-packet1.json",previous,error)
                previous=a["receipt_sha256"];actual.append(a)
                if index in terminal["completed_launches"]:check_prior_status(i,a["data"])
                elif a["data"]["status"]!="stopped":raise ValueError("unaccepted incomplete state")
            if len(actual)==3 and all(a["clean_supervision"]for a in actual):
                check_prior_status(1,actual[1]["data"]);check_prior_status(2,actual[2]["data"]);accepted=496
        with phases.measure("terminal_full_ledger_storage"):
            used=inventory(LEDGER)
            if used!=terminal["used"]or used["count"]!=5288+len(attempted)or used["previous"]!=previous:raise ValueError("complete retained ledger")
            full=storage_inventory(QUAL);owned=storage_inventory(ROOT);check()
        result={"status":"ROUTE2_CHECKPOINT_VERIFIED"if complete.exists()and accepted==496 else"ROUTE2_FIRST_REFUSAL_RECONCILED","actual":actual,"used":used,"accepted_pixels":accepted,"remaining_pixels":1024-accepted,"full_storage":full,"new_owned_storage":owned,"packet_sha256":sha(ROOT/"pipeline-packet1.json"),"terminal_source_sha256":sha(__file__),"terminal_phase_metrics":phases.finish(),"whole_image_complete":False,"photo_readiness":False,"no_campaign_or_retry":True,"original_requirements_preserved":True}
    seal(ROOT/"checkpoint-terminal-reconciliation.json",result);return result
if __name__=="__main__":print(json.dumps(reconcile()))
