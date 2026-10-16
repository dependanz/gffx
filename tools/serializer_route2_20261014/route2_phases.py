"""Separate inclusive timings with append-only phase events and visible faults."""
import contextlib,hashlib,json,os,threading,time
from pathlib import Path
import psutil
_current=None
def current():return _current
class Recorder:
    def __init__(self,output):
        self.output=Path(output);self.partial=Path(str(output)+".phases-current.json");self.final=Path(str(output)+".phases-final.json")
        self.events=Path(str(output)+".phase-events.jsonl");self.rows={};self.stack=[];self.lock=threading.RLock();self.stop=threading.Event()
        self.proc=psutil.Process();self.started=time.perf_counter();self.done=None;self.fault=None;self.sequence=0
    def _sample(self):
        rss=self.proc.memory_info().rss
        with self.lock:
            for label in self.stack:self.rows[label]["sampled_peak_host_bytes"]=max(self.rows[label]["sampled_peak_host_bytes"],rss)
        return rss
    def check(self):
        if self.fault is not None:raise RuntimeError("phase observer failed: "+repr(self.fault))
    def _loop(self):
        try:
            while not self.stop.wait(.05):self._sample()
        except BaseException as error:self.fault=error
    def __enter__(self):
        global _current
        self.previous_current=_current;_current=self
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start();return self
    def event(self,label,kind):
        with self.events.open("a",encoding="utf-8")as stream:
            stream.write(json.dumps({"sequence":self.sequence,"phase":label,"event":kind,"seconds":time.perf_counter()-self.started})+"\n")
        self.sequence+=1
    @contextlib.contextmanager
    def measure(self,label):
        self.check();rss=self._sample();start=time.perf_counter();cpu=time.process_time();self.event(label,"enter")
        with self.lock:
            row=self.rows.setdefault(label,{"calls":0,"wall_seconds":0.,"process_cpu_seconds":0.,"sampled_peak_host_bytes":0,"minimum_entry_host_bytes":rss})
            row["minimum_entry_host_bytes"]=min(row["minimum_entry_host_bytes"],rss);self.stack.append(label)
        try:yield
        finally:
            self._sample()
            with self.lock:
                if self.stack.pop()!=label:raise ValueError("phase nesting")
                row["calls"]+=1;row["wall_seconds"]+=time.perf_counter()-start;row["process_cpu_seconds"]+=time.process_time()-cpu
            self.event(label,"exit");self.check()
    def wrap(self,fn,label):
        def call(*args,**kwargs):
            with self.measure(label):return fn(*args,**kwargs)
        return call
    def data(self):
        with self.lock:return {"schema":"route2-inclusive-phase-measurements/1","elapsed_wall_seconds":time.perf_counter()-self.started,"sample_interval_seconds":.05,"phases":{k:dict(v)for k,v in self.rows.items()},"inclusive_nested_phases_do_not_sum":True,"sampled_process_memory_includes_live_input_graph_and_python":True,"observer_fault":repr(self.fault)if self.fault else None,"no_numerical_operations_or_checks_removed":True}
    def finish(self):
        if self.done is not None:return self.done
        self.stop.set();self.thread.join();self.check();data=self.data()
        with self.final.open("x",encoding="utf-8")as stream:json.dump(data,stream,indent=2)
        self.done={"path":str(self.final),"sha256":hashlib.sha256(self.final.read_bytes()).hexdigest()};return self.done
    def __exit__(self,*exc):
        global _current
        try:self.finish()
        finally:_current=self.previous_current
