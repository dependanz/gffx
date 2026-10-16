"""Preserve original historical checks and append new exact conditional chain."""
from route2_common import *
from route2_metadata import metadata,task_for_index
from report_journal142 import validate_event
def before_arrays(i=0,check=lambda:None):
    if type(i)is not int or not 0<=i<3:raise ValueError("three subject sequence")
    from whole_image_admission292 import before_arrays as original
    prior=original(0,check=check)
    terminal=read(FROZEN/"checkpoint-terminal-reconciliation2078.json")
    if sha(FROZEN/"checkpoint-terminal-reconciliation2078.json")!="acaac8e54e84573d8443df003b0b1ada0cdb652682d717217ef353d9cb72501a":raise ValueError("Source20 closure binding")
    from capacity_metadata335 import metadata as old_metadata
    previous=read(FROZEN/"pipeline-ledger-before1.json")["previous"]
    for j,saved in enumerate(terminal["actual"]):
        check();old=old_metadata(saved["report"],saved["task"],5285+j,FROZEN/"pipeline-packet1.json",previous)
        if old!=saved:raise ValueError("full retained Source20 typed chain")
        previous=old["receipt_sha256"]
        raw=Path(str(saved["report"])+".live.jsonl").read_bytes();events=[validate_event(json.loads(line))for line in raw.splitlines()]
        if not raw.endswith(b"\n")or[e["sequence"]for e in events]!=list(range(len(events)))or events[-1]["status"]!=old["data"]["status"]:raise ValueError("retained original journals")
    if terminal["accepted_pixels"]!=480 or terminal["remaining_pixels"]!=544 or terminal["used"]["count"]!=5288 or terminal["actual"][-1]["data"]["error"]!="MemoryError()":raise ValueError("closed unaccepted stop")
    p=packet();verify_sources(p,check=check)
    if p["base_launch_index"]!=5288 or len(p["tasks"])!=3:raise ValueError("no remaining campaign")
    previous=read(ROOT/"pipeline-ledger-before1.json")["previous"];chunks=list(prior["complete_pixel_chunks"])
    if len(chunks)!=30:raise ValueError("complete480 prefix")
    for j in range(i):
        check();task=task_for_index(j,p);out=Path(p["stage"])/task["output"]
        saved=metadata(out,task,p["base_launch_index"]+j,ROOT/"pipeline-packet1.json",previous)
        check_prior_status(j,saved["data"]);previous=saved["receipt_sha256"]
        if j==1:chunks.append(saved)
    prior["complete_pixel_chunks"]=chunks;prior["complete_pixel_audits"]=[];prior["retained_source20_closed_stop"]=terminal["actual"][-1]
    return prior
