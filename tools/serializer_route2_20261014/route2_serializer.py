"""Source-bound adoption; exact serializer source is unchanged."""
import io,json,os,sys,time
from fractions import Fraction
from pathlib import Path
from contextlib import contextmanager
from route2_common import GIB,QUAL,load_file,sha
PROTO=Path("C:/dev/gffx/tools/serializer_prototype_20261014/disk_exact_json.py")
HASH="56184c56fed001b9f5e7585da10a5bfbc324ceb7540662b930d7bddad023d391"
_OBS=Path("C:/dev/gffx/tools/serializer_prototype_20261014/owned_storage_observer3.py")
_OBS_HASH="32d51b4abf6051882701cf543d7bb36f6a9ec17f6c662ee4f9ffe37005097b7b"
if sha(PROTO)!=HASH or sha(_OBS)!=_OBS_HASH:raise ValueError("tested source binding")
prototype=load_file("route2_pinned_prototype",PROTO)
storage=load_file("route2_pinned_observer",_OBS)
_context=None
class Context:
    def __init__(self,stage,subject,checkpoint=lambda:None,limits=None):
        self.stage=Path(stage).resolve();self.stage.relative_to(QUAL.resolve())
        self.scratch=self.stage/"scratch"
        self.scratch.mkdir(exist_ok=True)
        self.journal=self.stage/("serializer-"+str(subject)+".jsonl")
        if self.journal.exists():raise FileExistsError("fresh adoption journal")
        self.checkpoint=checkpoint;self.rows=[]
        self.limits=limits or prototype.IndexLimits(max_database_bytes=8*GIB,max_scratch_bytes=16*GIB)
        from route2_environment import StorageObserver
        self.observer=StorageObserver(self.stage,limit=128*GIB,report_paths=[self.stage/"results"/x for x in ("controls.json","chunks/chunk-30/result.json","audit-30/result.json")])
    def check(self):
        self.checkpoint()
        return self.observer.bytes_used()
    def record(self,row):
        self.rows.append(row)
        text=json.dumps(row,allow_nan=False)+"\n"
        if len(text.encode())>8192 or len(self.rows)>50000:raise ValueError("adoption journal bound")
        with self.journal.open("a",encoding="utf-8")as stream:stream.write(text)
@contextmanager
def context(stage,subject,checkpoint=lambda:None,limits=None):
    global _context
    if _context is not None:raise ValueError("single adoption context")
    installed=Context(stage,subject,checkpoint,limits);_context=installed
    try:yield installed
    finally:_context=None
def configure(stage,subject,checkpoint=lambda:None):
    global _context
    if _context is not None:raise ValueError("single installed context")
    _context=Context(stage,subject,checkpoint)
def finish_context():
    global _context
    c=_context
    if c is None:return {"verified":False,"calls":0,"not_installed":True,"prototype_sha256":HASH}
    _context=None
    return {"verified":all(row.get("status")=="passed"for row in c.rows),
            "calls":len(c.rows),"journal":str(c.journal),
            "journal_sha256":sha(c.journal)if c.journal.exists()else None,
            "prototype_sha256":HASH,"full_graph_retained":True}
def baseline():
    try:
        import psutil
        m=psutil.Process().memory_info()
        return {"working_set_bytes":m.rss,"private_bytes":getattr(m,"private",0)}
    except ImportError:return {"not_measured":"psutil unavailable"}
class ComparingWriter:
    def __init__(self,source,checkpoint):
        import hashlib
        self.source=source;self.checkpoint=checkpoint;self.bytes=0;self.hash=hashlib.sha256()
    def write(self,text):
        data=text.encode("utf-8")
        for start in range(0,len(data),1024**2):
            self.checkpoint();block=data[start:start+1024**2]
            if self.source.read(len(block))!=block:raise ValueError("complete reference bytes differ")
            self.hash.update(block);self.bytes+=len(block)
        return len(text)
def _dump_exact_bound(value,out):
    c=_context
    if c is None:raise ValueError("source-bound context required")
    if type(out)is not io.TextIOWrapper or out.tell()!=0:raise TypeError("fresh ordinary file destination")
    path=Path(out.name).resolve();path.relative_to(c.stage)
    if path.is_relative_to(c.scratch):raise ValueError("permanent output outside scratch")
    if sha(PROTO)!=HASH:raise ValueError("prototype changed")
    import hashlib
    row={"path":str(path),"status":"working","graph_identity":id(value),"before_validation":baseline(),
         "before_validation_is_process_baseline_not_isolated_graph":True,"phases":{}}
    old_validate=prototype.validate_exact_graph;old_dump=json.dump
    original_scratch=prototype.DiskHeldIndex._check_scratch
    def phase(label,call):
        wall=time.perf_counter();cpu=time.process_time()
        try:
            from route2_phases import current
            recorder=current()
            if recorder is None:return call()
            name={"validation":"serializer_graph_validation","encoding_write":"serializer_json_encode_write"}.get(label,"serializer_"+label)
            with recorder.measure(name):return call()
        finally:row["phases"][label]={"wall_seconds":time.perf_counter()-wall,"CPU_seconds":time.process_time()-cpu}
    def timed_validate(root,**kw):
        if root is not value:raise ValueError("same exact held graph")
        result=phase("validation",lambda:old_validate(root,**kw))
        row["validation_counts"]=dict(result);return result
    def timed_dump(*args,**kw):return phase("encoding_write",lambda:old_dump(*args,**kw))
    def checked_scratch(index):original_scratch(index);c.check()
    class BudgetWriter:
        def __init__(self):self.bytes=0;self.other_bytes=0;self.hash=hashlib.sha256()
        def write(self,text):
            c.checkpoint();data=text.encode("utf-8");next_bytes=self.bytes+len(data)
            if self.other_bytes+next_bytes+16*GIB+64*1024**2>128*GIB:raise OSError("output forecast exceeds combined allowance")
            count=out.write(text);self.bytes=next_bytes;self.hash.update(data);return count
    writer=BudgetWriter();telemetry={}
    try:
        writer.other_bytes=c.check()
        prototype.validate_exact_graph=timed_validate;json.dump=timed_dump
        prototype.DiskHeldIndex._check_scratch=checked_scratch
        try:stats=prototype.dump_exact(value,writer,scratch_root=str(c.scratch),limits=c.limits,telemetry=telemetry)
        finally:
            prototype.validate_exact_graph=old_validate;json.dump=old_dump
            prototype.DiskHeldIndex._check_scratch=original_scratch
        phase("flush_fsync",lambda:(out.flush(),os.fsync(out.fileno())))
        c.check()
        fractions=0
        def default(item):
            nonlocal fractions
            if type(item)is not Fraction:raise TypeError("exact Fraction oracle")
            fractions+=1
            return {"rational_hex":(format(item.numerator,"x"),format(item.denominator,"x"))}
        def compare():
            with path.open("rb")as source:
                oracle=ComparingWriter(source,c.checkpoint)
                old_dump(value,oracle,default=default,separators=(",",":"),ensure_ascii=True,allow_nan=False)
                if source.read(1):raise ValueError("complete output EOF")
                if oracle.bytes!=writer.bytes or oracle.hash.hexdigest()!=writer.hash.hexdigest():raise ValueError("complete readback count/hash")
                return {"bytes":oracle.bytes,"sha256":oracle.hash.hexdigest()}
        row["readback"]=phase("independent_complete_readback",compare)
        if fractions!=stats["serialized_fraction_occurrences"]or any(stats[k]!=v for k,v in row["validation_counts"].items()):raise ValueError("exact traversal/Fraction counts")
        row.update(status="passed",stats=stats,index=telemetry,graph_still_same=id(value)==row["graph_identity"],after_validation=baseline())
        c.record(row);return stats
    except BaseException as error:
        row.update(status="stopped",error=repr(error),index=telemetry,emitted_bytes=writer.bytes)
        c.record(row);raise


def dump_exact(value,out):
    from route2_phases import current
    recorder=current()
    if recorder is None:return _dump_exact_bound(value,out)
    with recorder.measure("serializer_total"):return _dump_exact_bound(value,out)
