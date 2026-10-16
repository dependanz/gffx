"""Shared source-bound route2 contracts; importing this module launches nothing."""
import copy,hashlib,importlib.util,json,math,os,sys
from pathlib import Path
GIB=1024**3
QUAL=Path("D:/devspace/gffit/qualification-20261006")
FROZEN=QUAL/"boundary-sh28/gradient136-source20-checkpoint2"
H=QUAL/"boundary-sh28/gradient136-independent-visibility-bruteforce-preparation1"
LEDGER=QUAL/"boundary-sh27/ledger"
ROOT=Path(__file__).resolve().parent
for item in (str(H/"runtime2"),str(H),str(FROZEN),"C:/dev/gffx","C:/dev/gffx/tools/serializer_prototype_20261014"):
    if item not in sys.path:sys.path.append(item)
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb")as stream:
        for block in iter(lambda:stream.read(1024**2),b""):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def seal(path,value):
    with Path(path).open("x",encoding="utf-8",newline="")as stream:json.dump(value,stream,indent=2,allow_nan=False)
def finite(value):return type(value)in(int,float)and math.isfinite(value)and value>=0
def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def packet():
    p=read(ROOT/"pipeline-packet1.json");validate_packet(p);return p
def validate_packet(p):
    if p.get("schema")!="route2-prepared-checkpoint/1"or p.get("status")!="PREPARED_NOT_AUTHORIZED"or p.get("base_launch_index")!=5288 or Path(p["stage"])!=ROOT/"results":raise ValueError("prepared one-checkpoint scope")
    expected=(("route2_gate","control_repair","controls.json",3600,["--launch-index","5288"]),("route2_producer","geometry","chunks/chunk-30/result.json",18000,["--launch-index","5289"]),("route2_auditor","geometry","audit-30/result.json",10800,["--launch-index","5290"]))
    if len(p["tasks"])!=3:raise ValueError("exactly gate,one group,conditional audit")
    for task,(module,category,output,seconds,args)in zip(p["tasks"],expected):
        validate_task_shape(task);s=task["spec"]
        if(task["category"],task["output"],s["module"],Path(s["module_root"]),s["args"],s["seconds"])!=(category,output,module,ROOT,args,seconds):raise ValueError("exact subject binding")
    limits={"process_bytes":40*GIB,"RAM_reserve_bytes":4*GIB,"commit_reserve_bytes":4*GIB,"combined_bytes":128*GIB,"disk_reserve_bytes":GIB,"checkpoint_seconds":43200,"terminal_reserve_seconds":3600}
    if p["limits"]!=limits:raise ValueError("exact reviewed ceilings and reserves")
def verify_sources(p,check=lambda:None):
    for path,expected in p["sources"].items():
        check()
        if sha(path)!=expected:raise ValueError("source binding differs: "+path)
def validate_task_shape(contract):
    if type(contract)is not dict or set(contract)not in ({"category","output","output_forecast_bytes","spec"},{"category","output","output_forecast_bytes","spec","pixels"}):raise TypeError("closed task")
    s=contract["spec"]
    if type(s)is not dict or set(s)!={"module","module_root","args","host_bytes","seconds","watchdog_seconds"}:raise TypeError("closed spec")
    if s["host_bytes"]!="FRESH_AVAILABLE_RAM_BYTES_MINUS_4GIB":raise ValueError("fresh host token")
    if type(s["seconds"])is not int or not 0<s["seconds"]<=18000 or type(s["watchdog_seconds"])is not int or s["watchdog_seconds"]!=s["seconds"]+60:raise ValueError("bounded subject time")
    if type(s["module"])is not str or not s["module"]or type(s["module_root"])is not str or not s["module_root"]or type(contract["output"])is not str or not contract["output"]:raise TypeError("ordinary module/output")
    if contract["category"]not in("control_repair","geometry")or type(contract["output_forecast_bytes"])is not int or contract["output_forecast_bytes"]<0:raise TypeError("category/forecast")
    if Path(contract["output"]).is_absolute()or ".."in Path(contract["output"]).parts:raise ValueError("owned relative output")
    if type(s["args"])is not list or any(type(x)is not str for x in s["args"]):raise TypeError("closed arguments")
    if "pixels"in contract and contract["pixels"]!=[480,496]:raise ValueError("one original16 pixel group")
def resolve_task(contract,snapshot,limits):
    validate_task_shape(contract)
    keys={"total_RAM_bytes","available_RAM_bytes","commit_limit_bytes","commit_used_bytes","commit_headroom_bytes","free_disk_bytes"}
    if type(snapshot)is not dict or set(snapshot)!=keys or any(type(snapshot[k])is not int or snapshot[k]<0 for k in keys):raise TypeError("ordinary physical/commit snapshot")
    if snapshot["available_RAM_bytes"]>snapshot["total_RAM_bytes"]or snapshot["commit_used_bytes"]>snapshot["commit_limit_bytes"]or snapshot["commit_headroom_bytes"]!=snapshot["commit_limit_bytes"]-snapshot["commit_used_bytes"]:raise ValueError("physical/commit arithmetic")
    if snapshot["available_RAM_bytes"]<limits["process_bytes"]+limits["RAM_reserve_bytes"]:raise MemoryError("RAM admission")
    if snapshot["commit_headroom_bytes"]<limits["process_bytes"]+limits["commit_reserve_bytes"]:raise MemoryError("commit admission")
    if snapshot["free_disk_bytes"]<=limits["combined_bytes"]+limits["disk_reserve_bytes"]:raise OSError("disk admission")
    task=copy.deepcopy(contract);task["spec"]["host_bytes"]=min(limits["process_bytes"],snapshot["available_RAM_bytes"]-limits["RAM_reserve_bytes"],snapshot["commit_headroom_bytes"]-limits["commit_reserve_bytes"])
    return task
def check_prior_status(position,data):
    if data.get("status")!="passed"or not data.get("supervisor")or data["supervisor"].get("failure")is not None or data["supervisor"].get("exit_code")!=0:raise ValueError("clean conditional predecessor required")
    if position==0:
        if data.get("tests")!=1639 or data.get("failed")!=0 or data.get("skipped")!=0 or not data.get("binding_tests_passed"):raise ValueError("required qualification gate")
    elif position==1:
        if data.get("pixels_complete")!=16 or data.get("pixel_range")!=[480,496]or len(data.get("certificates",[]))!=16 or not data.get("all_histories_saved_losslessly"):raise ValueError("complete producer")
    elif position==2:
        if data.get("pixels_audited")!=16 or data.get("pixel_range")!=[480,496]or not data.get("pixel_group_audit_complete")or not data.get("native_group_colour_and_mask_agreement")or data.get("whole_image_complete")is not False:raise ValueError("complete independent selected audit")
    else:raise ValueError("exactly3 admitted subjects")
    if not data.get("serializer_adoption",{}).get("verified"):raise ValueError("verified source-bound adoption")
