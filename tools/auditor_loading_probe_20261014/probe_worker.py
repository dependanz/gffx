"""One bytes-loader → original decoder → original OwnedWork.hold measurement."""
import argparse,contextlib,json,os,sys,threading,time,traceback
from pathlib import Path
from probe_core import *
class Recorder:
    def __init__(self,root,deadline,limit):
        self.root=root;self.deadline=deadline;self.limit=limit;self.owner=None;self.graph=None
        self.rows={};self.stack=[];self.lock=threading.RLock();self.stop=threading.Event();self.fault=None;self.sequence=0
        self.storage=Storage(root,limit=64*MIB);self.started=time.perf_counter()
    def check(self):
        if self.fault is not None:raise RuntimeError('measurement observer failed:'+repr(self.fault))
        if time.perf_counter()>self.deadline:raise TimeoutError('single-record900s deadline')
    def event(self,event,label):
        row={'sequence':self.sequence,'event':event,'phase':label,'seconds':time.perf_counter()-self.started,'memory':memory(),'ownership':ownership(self.owner)if self.owner else None}
        data=(json.dumps(row,allow_nan=False)+'\n').encode()
        if len(data)>8192 or self.storage.bytes_used()+len(data)>64*MIB:raise OSError('64MiB report/log allowance')
        with(self.root/'phase-events.jsonl').open('ab')as out:out.write(data)
        self.sequence+=1
    def sample(self):
        m=memory()
        if m['RSS_bytes']>self.limit or m['private_commit_bytes']>self.limit:raise MemoryError('process memory ceiling')
        with self.lock:
            for name in self.stack:
                row=self.rows[name];row['samples']+=1
                row['sampled_peak_RSS_bytes']=max(row['sampled_peak_RSS_bytes'],m['RSS_bytes'])
                row['sampled_peak_private_commit_bytes']=max(row['sampled_peak_private_commit_bytes'],m['private_commit_bytes'])
                if self.owner:
                    row['sampled_peak_seen_count']=max(row['sampled_peak_seen_count'],len(self.owner.storage.seen))
                    row['sampled_peak_seen_set_shallow_bytes']=max(row['sampled_peak_seen_set_shallow_bytes'],sys.getsizeof(self.owner.storage.seen))
        return m
    def loop(self):
        try:
            while not self.stop.wait(.01):self.sample()
        except BaseException as error:self.fault=error
    def start(self):self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def close(self):self.stop.set();self.thread.join()
    @contextlib.contextmanager
    def phase(self,name):
        self.check();entry=memory();start=time.perf_counter();cpu=time.process_time()
        if name in self.rows:raise ValueError('one invocation per phase')
        with self.lock:
            self.rows[name]={'entry':entry,'samples':0,'sampled_peak_RSS_bytes':entry['RSS_bytes'],'sampled_peak_private_commit_bytes':entry['private_commit_bytes'],'sampled_peak_seen_count':0,'sampled_peak_seen_set_shallow_bytes':0,'ownership_before':ownership(self.owner)if self.owner else None};self.stack.append(name)
        self.event('enter',name)
        try:yield
        finally:
            self.sample();row=self.rows[name]
            row.update(exit=memory(),wall_seconds=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu,ownership_after=ownership(self.owner)if self.owner else None)
            with self.lock:
                if self.stack.pop()!=name:raise ValueError('phase nesting')
            self.event('exit',name)
    def call(self,name,fn,*args):
        with self.phase(name):return fn(*args)
@contextlib.contextmanager
def loader_observation(recorder,input_path,expected):
    original_read=Path.read_bytes;original_loads=json.loads
    def observed_read(path):
        if Path(path)!=input_path:return original_read(path)
        with recorder.phase('bytes_read'):
            data=original_read(path)
            recorder.consumed_bytes=len(data);recorder.consumed_sha256=hashlib.sha256(data).hexdigest()
            if recorder.consumed_sha256!=expected:raise ValueError('actual consumed input SHA256')
            return data
    def observed_loads(*args,**kwargs):
        with recorder.phase('JSON_parse'):return original_loads(*args,**kwargs)
    Path.read_bytes=observed_read;json.loads=observed_loads
    try:yield
    finally:Path.read_bytes=original_read;json.loads=original_loads
def run(spec):
    started=time.perf_counter();root=Path(spec['root']);deadline=started+spec['seconds']
    recorder=Recorder(root,deadline,spec['process_bytes']);recorder.start();status='failed';error=None;graph=owner=None
    seal(root/'worker-ready.json',{'worker_pid':os.getpid(),'worker_sha256':sha(__file__)})
    try:
        while not(root/'GO.json').exists():recorder.check();time.sleep(.01)
        with recorder.phase('production_imports'):
            for path,h in spec['source_bindings'].items():
                recorder.check()
                if sha(path)!=h:raise ValueError('unchanged source binding:'+path)
            load_record,decode,OwnedWork,optional_none=production()
            owner=OwnedWork(9223372036854775807,9223372036854775807,recorder.check)
            optional_none(owner,None);recorder.owner=owner
        with loader_observation(recorder,Path(spec['input']),spec['input_sha256']):
            # Root parsed JSON has expression lifetime, as in r=g.decode(read(path)).
            # No raw copy or parsed-tree local is retained after the decoder returns.
            graph=recorder.call('exact_decode',decode,recorder.call('bytes_loader_total',load_record,Path(spec['input'])))
        recorder.graph=graph
        if type(graph)is not dict or type(graph.get('pixel'))is not int or graph['pixel']!=spec['pixel']:raise ValueError('selected record identity')
        recorder.call('OwnedWork.hold',owner.hold,graph)
        with recorder.phase('retained_accounting_inspection'):
            complete=ownership(owner,deep=True)
            if owner.history[-1]is not graph or id(graph)not in owner.storage.seen or owner.failed:raise ValueError('exact record/ownership retention')
            if owner.work.count!=1:raise ValueError('only original optional-None ownership setup unit')
        recorder.check();recorder.event('complete_with_record_and_owner_alive','retained_state');status='passed'
    except BaseException as exception:error=repr(exception);traceback.print_exc()
    finally:
        recorder.close()
        if recorder.fault is not None:status='failed';error=error or repr(recorder.fault)
        retained=bool(graph is not None and owner is not None and owner.history and owner.history[-1]is graph)
        result={'status':status,'worker_pid':os.getpid(),'error':error,'phases':recorder.rows,'sample_interval_seconds':.01,'OS_peaks_are_cumulative_lifetime_at_phase_entry_exit':True,'OS_peaks_never_reset':True,'same_production_functions':True,'consumed_input_bytes':getattr(recorder,'consumed_bytes',None),'consumed_input_sha256':getattr(recorder,'consumed_sha256',None),'ownership_final':complete if status=='passed'else ownership(owner)if owner else None,'retained_record_and_owner_alive':retained,'retained_record_identity':id(graph)if graph else None,'held_record_identity':id(owner.history[-1])if retained else None,'mathematical_audit_operations':0,'serialization_calls':0,'integration_calls':0,'scientific_acceptance':False,'whole_history_capacity_qualified':False,'elapsed_seconds':time.perf_counter()-started,'final_memory':memory(),'measurement_overhead':'scalar reports/sampling/input SHA and final identity-size inspection;no gc.collect or graph cleanup'}
        seal(root/'worker-result.json',result)
        if status=='passed':
            # Supervisor observes final live state before releasing this one process.
            while not(root/'measurement-complete.json').exists():
                if time.perf_counter()>deadline:raise TimeoutError('final measurement acknowledgment deadline')
                time.sleep(.01)
            if owner.history[-1]is not graph:raise ValueError('retention until supervisor completion')
    return 0 if status=='passed'else 1
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--spec',required=True);parser.add_argument('--output');parser.add_argument('--fixtures')
    args=parser.parse_args()
    if not args.worker:raise PermissionError('isolated supervisor only')
    return run(read(args.spec))
if __name__=='__main__':raise SystemExit(main())
