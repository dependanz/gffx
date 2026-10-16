"""Strict bounded task, CRLF/journal, PID, artifact and receipt linkage."""
from route2_common import *
from report_journal142 import validate_event,MAX_BYTES,MAX_RECORD_BYTES,MAX_RECORDS
from bruteforce_gpu_profile228 import valid_report
def task_for_index(i,p=None):
    p=p or packet()
    if type(i)is not int or not 0<=i<3:raise ValueError("three subject sequence")
    index=p["base_launch_index"]+i;r=read(ROOT/f"job-{index}-resolved-task1.json")
    if set(r)!={"schema","packet_sha256","task_contract_sha256","snapshot","task","physical_evidence_path","physical_evidence_sha256"}or r["schema"]!="route2-fresh-resolution/1":raise ValueError("closed resolution")
    if r["packet_sha256"]!=sha(ROOT/"pipeline-packet1.json")or r["task_contract_sha256"]!=digest(p["tasks"][i])or r["task"]!=resolve_task(p["tasks"][i],r["snapshot"],p["limits"]):raise ValueError("packet/physical/task binding")
    e=read(r["physical_evidence_path"])
    if sha(r["physical_evidence_path"])!=r["physical_evidence_sha256"]or e["snapshot"]!=r["snapshot"]or e["used"]["count"]!=index or e["matching_other_heavy_commands"]!=[]or not e["independent_psutil_quiet_excluding_exact_admitted_controller"]:raise ValueError("fresh physical evidence")
    return r["task"]
def metadata(out,task,index,packet_path,previous=None):
    p=read(packet_path);i=index-p["base_launch_index"]
    if task!=task_for_index(i,p):raise ValueError("resolved task")
    lp=LEDGER/f"launch-{index:04}.json";launch=read(lp);finish=read(LEDGER/f"finish-{index:04}.json");data=read(out)
    if launch["index"]!=index or launch["category"]!=task["category"]or Path(launch["output"])!=Path(out)or launch["packet_sha256"]!=sha(packet_path)or launch["profile_sha256"]!=sha(ROOT/"profile2.json")or launch["resolved_task_sha256"]!=sha(ROOT/f"job-{index}-resolved-task1.json"):raise ValueError("launch binding")
    if launch["reserved_worker_seconds"]!=task["spec"]["seconds"]or launch["reserved_supervised_seconds"]!=task["spec"]["watchdog_seconds"]or read(str(out)+".spec.json")!=task["spec"]:raise ValueError("spec/reservation")
    if previous is not None and launch["previous"]!=previous:raise ValueError("receipt chain")
    if finish!=read(str(out)+".receipt.json")or finish["launch_sha256"]!=sha(lp)or finish["report_sha256"]!=sha(out)or finish["status"]!=data["status"]:raise ValueError("finish binding")
    if not all(finite(finish[k])for k in("worker_seconds","supervised_seconds")):raise ValueError("retained finite charge")
    su=data["supervisor"]
    if not finite(su["elapsed_seconds"])or type(su["exit_code"])is not int:raise ValueError("actual supervisor")
    clean=data["status"]=="passed"and su["exit_code"]==0 and su["failure"]is None
    if clean:
        valid_report(data,task["spec"])
        if not su["actual_worker_identity_matches"]or su["kernel_job_peak_process_commit_bytes"]>p["limits"]["process_bytes"]:raise ValueError("PID/OS cap")
    elif data["status"]!="stopped"or(su["exit_code"]==0 and not su["failure"]):raise ValueError("strict STOP")
    if "seconds"not in data and(clean or not finish.get("report_worker_seconds_unreported")or finish["worker_seconds"]<su["elapsed_seconds"]):raise ValueError("conservative abrupt charge")
    raw=Path(str(out)+".live.jsonl").read_bytes();lines=raw.splitlines()
    if not raw.endswith(b"\n")or len(raw)>MAX_BYTES or len(lines)>MAX_RECORDS or any(len(x)+1>MAX_RECORD_BYTES for x in lines):raise ValueError("complete journal bounds")
    events=[validate_event(json.loads(line))for line in lines]
    if not events or[x["sequence"]for x in events]!=list(range(len(events))):raise ValueError("journal sequence")
    times=[x["seconds"]for x in events if x.get("seconds")is not None]
    if times!=sorted(times)or(clean and events[-1]["status"]!="passed")or(not clean and events[-1]["status"]not in("working","stopped")):raise ValueError("journal terminal/monotonicity")
    raw=Path(out).read_bytes()
    if b"\r\n"not in raw or b"\n"in raw.replace(b"\r\n",b""):raise ValueError("original CRLF")
    if "phase_metrics"in data and sha(data["phase_metrics"]["path"])!=data["phase_metrics"]["sha256"]:raise ValueError("phase binding")
    if clean:
        a=data.get("serializer_adoption",{})
        if not a.get("verified"):raise ValueError("adoption verified")
        if i and(not a["calls"]or sha(a["journal"])!=a["journal_sha256"]):raise ValueError("complete adoption journal")
        if i:
            raw=Path(a["journal"]).read_bytes();rows=[json.loads(line)for line in raw.splitlines()]
            if not raw.endswith(b"\n")or len(rows)!=a["calls"]or a["calls"]!=(17 if i==1 else 2):raise ValueError("exact full serialization call inventory")
            for row in rows:
                if row["status"]!="passed"or not row["graph_still_same"]or sha(row["path"])!=row["readback"]["sha256"]or Path(row["path"]).stat().st_size!=row["readback"]["bytes"]:raise ValueError("complete independent serialization readback")
                if any(type(row["stats"][k])is not int or row["stats"][k]<0 for k in ("validated_unique_objects","validated_container_edges","serialized_fraction_occurrences")):raise ValueError("exact recorded counters")
        check_prior_status(i,data)
        refs=data["certificates"]+[data["chunk_history"]]if i==1 else[data["certificate_index"],data["audit_history"]]if i==2 else[]
        for ref in refs:
            if sha(ref["path"])!=ref["sha256"]:raise ValueError("complete artifact hash")
    return {"report":str(out),"task":task,"launch":launch,"finish":finish,"report_sha256":sha(out),"receipt_sha256":sha(str(out)+".receipt.json"),"data":data,"clean_supervision":clean}


def failure_cost_metadata(out,task,index,packet_path,previous,cause):
    # Cost reconciliation only. This NEVER supplies clean numerical acceptance.
    p=read(packet_path)
    if task!=task_for_index(index-p["base_launch_index"],p):raise ValueError("failed exact resolution")
    lp=LEDGER/f"launch-{index:04}.json";launch=read(lp);finish=read(LEDGER/f"finish-{index:04}.json");data=read(out)
    if launch["index"]!=index or launch["previous"]!=previous or launch["packet_sha256"]!=sha(packet_path)or launch["profile_sha256"]!=sha(ROOT/"profile2.json")or launch["resolved_task_sha256"]!=sha(ROOT/f"job-{index}-resolved-task1.json")or Path(launch["output"])!=Path(out)or read(str(out)+".spec.json")!=task["spec"]:raise ValueError("failed launch/source/spec chain")
    if finish!=read(str(out)+".receipt.json")or finish["launch_sha256"]!=sha(lp)or finish["report_sha256"]!=sha(out)or finish["status"]!="stopped"or data["status"]!="stopped"or not all(finite(finish[k])for k in("worker_seconds","supervised_seconds")):raise ValueError("failed exact retained charges")
    if finish["worker_seconds"]<data["supervisor"]["elapsed_seconds"]and"seconds"not in data:raise ValueError("conservative incomplete report charge")
    return {"report":str(out),"task":task,"launch":launch,"finish":finish,"receipt_sha256":sha(str(out)+".receipt.json"),"data":data,"clean_supervision":False,"original_journal_or_metadata_refusal":repr(cause),"numerical_acceptance_forbidden":True}
