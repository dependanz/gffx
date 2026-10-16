"""Isolated measurement support; importing this module performs no probe."""
import ctypes,datetime,hashlib,importlib.util,json,os,shutil,sys,time
from ctypes import wintypes
from pathlib import Path
GIB=1024**3;MIB=1024**2
QUAL=Path('D:/devspace/gffit/qualification-20261006')
FROZEN=QUAL/'boundary-sh28/gradient136-source20-checkpoint2'
CHECKPOINT=QUAL/'boundary-sh28/gradient136-disk-serializer-checkpoint2'
INPUT=QUAL/'boundary-sh28/gradient136-source20-checkpoint-results2/chunks/chunk-30/pixel-0491.json'
RUN=QUAL/'boundary-sh28/auditor-loading-pixel491-20261014-probe1'
CODE=Path(__file__).resolve().parent
PROTO=Path('C:/dev/gffx/tools/serializer_prototype_20261014')
for path in (str(CHECKPOINT),str(FROZEN),'C:/dev/gffx',str(PROTO)):
    if path not in sys.path:sys.path.insert(0,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(MIB),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def seal(path,value):
    text=json.dumps(value,indent=2,allow_nan=False).encode()
    if len(text)>MIB:raise OSError('bounded scalar report')
    with Path(path).open('xb')as stream:stream.write(text)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
HELPER=PROTO/'serializer_saved_pixel486_benchmark_observer3.py'
if sha(HELPER)!='5a4096394f6d2268a83951b653eb5078230473e5f0834dcfc15fb2cba211408b':raise ValueError('qualified OS helper binding')
win=load('_auditor_probe_os',HELPER)
observer=load('_auditor_probe_storage',PROTO/'owned_storage_observer3.py')
if sha(PROTO/'owned_storage_observer3.py')!='32d51b4abf6051882701cf543d7bb36f6a9ec17f6c662ee4f9ffe37005097b7b':raise ValueError('qualified storage binding')
class Storage(observer.StorageObserver):
    def _temporary(self,path):
        roots=(self.root/'scratch/fixtures',self.root/'scratch/owned-worker')
        return any(path==root or path.is_relative_to(root)for root in roots)or super()._temporary(path)
class PI(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(n,wintypes.DWORD)for n in('HandleCount','ProcessCount','ThreadCount')]
def physical(root):
    api=ctypes.WinDLL('psapi',use_last_error=True);api.GetPerformanceInfo.argtypes=[ctypes.POINTER(PI),wintypes.DWORD];api.GetPerformanceInfo.restype=wintypes.BOOL
    p=PI();p.cb=ctypes.sizeof(p)
    if not api.GetPerformanceInfo(ctypes.byref(p),p.cb):raise ctypes.WinError(ctypes.get_last_error())
    return {'timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'total_RAM_bytes':p.PhysicalTotal*p.PageSize,'available_RAM_bytes':p.PhysicalAvailable*p.PageSize,'commit_limit_bytes':p.CommitLimit*p.PageSize,'commit_usage_bytes':p.CommitTotal*p.PageSize,'commit_headroom_bytes':(p.CommitLimit-p.CommitTotal)*p.PageSize,'D_free_bytes':shutil.disk_usage(root).free}
def admit_snapshot(s,limit=20*GIB,report_limit=64*MIB):
    if s['available_RAM_bytes']<limit+4*GIB:raise MemoryError('RAM admission:process allowance plus4GiB')
    if s['commit_headroom_bytes']<limit+4*GIB:raise MemoryError('Windows commit admission:process allowance plus4GiB')
    if s['D_free_bytes']<=GIB+report_limit:raise OSError('D: disk admission:1GiB reserve plus report allowance')
def quiet():
    from route2_environment import quiet as qualified
    import psutil,subprocess
    allowed={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    result=qualified(allowed)
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid in allowed:continue
        if 'python'not in(process.info['name']or'').lower():continue
        if process.info['cmdline']is None:raise PermissionError('unreadable Python process')
        command=' '.join(process.info['cmdline']).lower()
        if '--worker'in process.info['cmdline']or 'auditor_loading_probe'in command or 'probe_controller.py'in command:raise ValueError('another probe/worker')
    return result
def memory():
    import psutil
    m=psutil.Process().memory_info()
    return {'RSS_bytes':m.rss,'private_commit_bytes':m.private,'OS_lifetime_peak_RSS_bytes':m.peak_wset,'OS_lifetime_peak_commit_bytes':m.peak_pagefile}
def production():
    # Frozen direct dependencies precede old duplicate packages; no prerequisite/main is called.
    sys.path.insert(0,str(FROZEN));sys.path.insert(0,str(CHECKPOINT))
    import route2_auditor as auditor
    import numpy
    import duffy_integral271 as decoder
    import scene_support289 as support
    from whole_runtime274.pixel_fields258 import OwnedWork
    if Path(auditor.__file__).resolve()!=CHECKPOINT/'route2_auditor.py'or Path(decoder.__file__).resolve()!=FROZEN/'duffy_integral271.py':raise ValueError('exact production module resolution')
    if Path(sys.modules[OwnedWork.__module__].__file__).resolve()!=FROZEN/'whole_runtime274/pixel_fields258.py':raise ValueError('frozen ownership resolution')
    return auditor.read,decoder.decode,OwnedWork,support._owned_optional_owner
def ownership(o,deep=False):
    seen=o.storage.seen
    result={'seen_count':len(seen),'seen_set_shallow_bytes':sys.getsizeof(seen),'scientific_accounting_used_bytes':o.storage.used,'history_entries':len(o.history),'history_list_shallow_bytes':sys.getsizeof(o.history),'history_charged_header_bytes':o.header,'ownership_work_count':o.work.count,'failed':o.failed,'accounting_excludes_seen_set_and_identity_integer_storage':True}
    if deep:
        payload=sum(sys.getsizeof(identity)for identity in seen)
        result.update(seen_identity_integer_payload_bytes=payload,seen_total_set_plus_identity_bytes=sys.getsizeof(seen)+payload)
    return result
