"""Serial resource-supervised camera qualification against exact installed wheels."""
import argparse,os,sys,time,json,unittest
from pathlib import Path
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact

def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",action="store_true");p.add_argument("--fixtures",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--package",required=True);p.add_argument("--wheel",type=Path,required=True);p.add_argument("--sha",required=True);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");p.add_argument("--regression",action="store_true");p.add_argument("--gffit-package");p.add_argument("--gffit-wheel",type=Path)
    a=p.parse_args();sys.path.insert(0,a.package)
    if a.gffit_package:sys.path.insert(0,a.gffit_package)
    if not a.worker:
        if a.device=="cuda":cost.PROFILE["sampled_working_set_limit_bytes"]=2*1024**3
        args=["--package",a.package,"--wheel",str(a.wheel),"--sha",a.sha,"--device",a.device]+(["--regression"] if a.regression else [])
        if a.gffit_package:args += ["--gffit-package",a.gffit_package,"--gffit-wheel",str(a.gffit_wheel)]
        return cost.supervise(a.fixtures,a.output,worker_module="tools.qualify_parallel_camera",worker_args=args)
    start=time.perf_counter();row=dict(status="working",worker_pid=os.getpid(),device=a.device);cost.write_report(a.output,row)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row["wheel"]=artifact.identity(a.wheel,a.sha)
        if a.device=="cuda":torch.cuda.reset_peak_memory_stats()
        os.environ["GFFIT_TEST_DEVICE"]=a.device
        if a.gffit_package:
            import gffit,zipfile
            with zipfile.ZipFile(a.gffit_wheel) as z:
                members=[n for n in z.namelist() if n.startswith("gffit/") and not n.endswith("/")]
                if not members or any((Path(gffit.__file__).parent.parent/n).read_bytes()!=z.read(n) for n in members):raise ValueError("gffit wheel mismatch")
            row["gffit"]={"sha256":artifact.digest(a.gffit_wheel),"matched_members":len(members)}
        if a.regression:
            import pytest
            class Results:
                def __init__(self):self.reports=[]
                def pytest_runtest_logreport(self,report):
                    if report.when=="call" or report.failed:self.reports.append(dict(node=report.nodeid,outcome=report.outcome,detail=str(report.longrepr) if report.failed else None))
            results=Results();rc=pytest.main(["-q","-p","no:cacheprovider","C:/dev/gffx/tests/python/test_camera.py","C:/dev/gffx/tests/python/test_camera_inverse_order.py"],plugins=[results]);row["tests"]=results.reports
            if rc:raise ValueError("perspective regressions failed")
        else:
            suite=(unittest.defaultTestLoader.discover("C:/dev/gffit/tests") if a.gffit_package else
                   unittest.defaultTestLoader.discover("C:/dev/gffx/tests/python",pattern="test_orthographic_camera.py"))
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            row.update(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),failure_details=[str(t)+"\n"+reason for t,reason in result.failures+result.errors])
            if not result.wasSuccessful():raise ValueError("parallel camera controls failed")
        if a.device=="cuda":
            torch.cuda.synchronize();row["gpu_peak_bytes"]=torch.cuda.max_memory_allocated()
            if row["gpu_peak_bytes"]>64*1024**2:raise MemoryError("GPU cap")
        cost.check_deadline(start+120);row["status"]="passed"
    except Exception as e:row.update(status="stopped",error=f"{type(e).__name__}: {e}");raise
    finally:
        row["seconds"]=time.perf_counter()-start
        row["sources"]={str(p):artifact.digest(p) for p in [Path(__file__),Path("C:/dev/gffx/tests/python/test_orthographic_camera.py"),*Path("C:/dev/gffit/tests").glob("test_*.py")]}
        cost.write_report(a.output,row)
    return 0
if __name__=="__main__":raise SystemExit(main())
