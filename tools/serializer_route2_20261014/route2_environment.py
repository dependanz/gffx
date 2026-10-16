"""Fresh Windows commit/RAM admission and exact owned-temp accounting."""
import ctypes,os,shutil
from ctypes import wintypes
from pathlib import Path
from route2_common import load_file,sha
HELPER=Path("C:/dev/gffx/tools/serializer_prototype_20261014/serializer_saved_pixel486_benchmark_observer3.py")
if sha(HELPER)!="5a4096394f6d2268a83951b653eb5078230473e5f0834dcfc15fb2cba211408b":raise ValueError("qualified OS helper binding")
win=load_file("route2_os_helper",HELPER)
base=load_file("route2_storage_base",HELPER.with_name("owned_storage_observer3.py"))
class PI(ctypes.Structure):
    _fields_=[("cb",wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in("CommitTotal","CommitLimit","CommitPeak","PhysicalTotal","PhysicalAvailable","SystemCache","KernelTotal","KernelPaged","KernelNonpaged","PageSize")]+[(n,wintypes.DWORD)for n in("HandleCount","ProcessCount","ThreadCount")]
def snapshot(root):
    api=ctypes.WinDLL("psapi",use_last_error=True);api.GetPerformanceInfo.argtypes=[ctypes.POINTER(PI),wintypes.DWORD];api.GetPerformanceInfo.restype=wintypes.BOOL
    p=PI();p.cb=ctypes.sizeof(p)
    if not api.GetPerformanceInfo(ctypes.byref(p),p.cb):raise ctypes.WinError(ctypes.get_last_error())
    return {"total_RAM_bytes":p.PhysicalTotal*p.PageSize,"available_RAM_bytes":p.PhysicalAvailable*p.PageSize,"commit_limit_bytes":p.CommitLimit*p.PageSize,"commit_used_bytes":p.CommitTotal*p.PageSize,"commit_headroom_bytes":(p.CommitLimit-p.CommitTotal)*p.PageSize,"free_disk_bytes":shutil.disk_usage(root).free}
def quiet(allow=()):
    import psutil
    matches=[];denied=[];allowed={os.getpid(),*allow}
    for process in psutil.process_iter(["pid","name","cmdline"]):
        if process.pid in allowed:continue
        try:
            if "python"not in(process.info["name"]or"").lower():continue
            args=process.info["cmdline"]
            if args is None:denied.append(process.pid);continue
            text=" ".join(args).lower()
            if "--worker"in args or any(w in text for w in("run_whole136","route2_checkpoint.py","serializer_saved_pixel486_benchmark","g4-")):matches.append(process.info)
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied:denied.append(process.pid)
    if matches or denied:raise ValueError("quiet refusal: "+repr((matches,denied)))
    import subprocess,json
    result=subprocess.run(["pwsh","-NoProfile","-Command","@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python'} | Select-Object ProcessId,Name,CommandLine) | ConvertTo-Json -Depth 4 -Compress"],capture_output=True,text=True,encoding="utf-8",check=True,timeout=30)
    rows=json.loads(result.stdout)if result.stdout.strip()else[]
    rows=[rows]if isinstance(rows,dict)else rows
    for row in rows:
        if row["ProcessId"]in allowed:continue
        text=(row["CommandLine"]or "").lower()
        if not text or any(w in text for w in ("--worker","run_whole136","route2_checkpoint.py","serializer_saved_pixel486_benchmark","g4-")):raise ValueError("independent CIM quiet refusal: "+repr(row))
    return {"matching_heavy_commands":[],"unreadable_python":[],"allowed_exact_pids":sorted(allowed),"fresh_CIM_independent_check":True}
class StorageObserver(base.StorageObserver):
    def __init__(self,root,*,limit,report_paths=()):
        super().__init__(root,limit=limit)
        self.worker_temps={self.root/"scratch"/("worker-"+str(i))for i in (5288,5289,5290)}
        self.report_temporaries={Path(str(Path(p).absolute())+".tmp")for p in report_paths}
        if any(not p.is_relative_to(self.root)for p in self.report_temporaries):raise ValueError("owned report temp scope")
    def _temporary(self,path):return path in self.report_temporaries or any(path==root or path.is_relative_to(root)for root in self.worker_temps)or super()._temporary(path)
def storage_inventory(root):
    total=files=0
    def fail(error):raise error
    for directory,dirs,names in os.walk(root,onerror=fail):
        for name in names:total+=(Path(directory)/name).stat().st_size;files+=1
    return {"bytes":total,"files":files}
