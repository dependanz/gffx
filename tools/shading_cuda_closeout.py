"""Exact-artifact CUDA reconciliation using the unchanged finite acceptance tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
from tools import material_loss_cost as cost

MODES=('chains','directional','mean','furnace','repeat')
COUNTS=dict(directional=80,mean=16,furnace=192)
HOST_BYTES=2*1024**3


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(wheel, expected):
    import gffx
    if len(expected)!=64 or digest(wheel)!=expected:raise ValueError('wheel identity')
    root=Path(gffx.__file__).resolve().parent.parent
    with zipfile.ZipFile(wheel) as archive:
        names=[n for n in archive.namelist() if n.startswith('gffx/') and not n.endswith('/')]
        if not names or any((root/name).read_bytes()!=archive.read(name) for name in names):
            raise ValueError('installed member identity')
    return dict(sha256=expected,installed_root=str(root),matched_members=len(names))


def worker(fixtures,output,wheel,expected,mode):
    deadline=time.perf_counter()+120
    row=dict(status='working',worker_pid=os.getpid(),mode=mode,profile=dict(host_bytes=HOST_BYTES,gpu_bytes=64*1024**2,cooperative_seconds=120,watchdog_seconds=180))
    cost.write_report(output,row)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=identity(wheel,expected)
        row['driver_sha256']=digest(__file__)
        if not torch.cuda.is_available():raise ValueError('actual CUDA device required')
        row['device']=dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),torch=torch.__version__,cuda=torch.version.cuda)
        torch.cuda.reset_peak_memory_stats(0)
        if mode in COUNTS:
            from tools import public_budget_probe as original
            row['references']=original.verify_references(Path(__file__).parent)
            row['input_hashes']={name:digest(fixtures/(name+'.json')) for name in cost.HASHES}
            if row['input_hashes']!=cost.HASHES:raise ValueError('physical input identity')
            os.environ['GFFX_CUDA_BUDGET_FIXTURES']=str(fixtures.resolve())
            os.environ['GFFX_CUDA_BUDGET_DEADLINE']=str(deadline)
            measurements=output.with_suffix('.rows.json')
            os.environ['GFFX_CUDA_BUDGET_ROWS']=str(measurements.resolve())
            tests=['tests/python/test_principled_cuda_physical.py'];selection=['-k',mode]
        elif mode=='repeat':
            measurements=output.with_suffix('.measurements.json')
            os.environ['GFFX_CUDA_REPEAT_RESULTS']=str(measurements.resolve())
            os.environ['GFFX_CUDA_REPEAT_DEADLINE']=str(deadline)
            tests=['tests/python/test_cuda_repetition.py'];selection=[]
        elif mode=='chains':
            tests=['tests/python/test_principled_cuda_qualification.py','tests/python/test_shading_m1_gradients.py'];selection=[]
        else:raise ValueError('undeclared stage')
        row['tests']={name:digest(name) for name in tests}
        import pytest
        code=pytest.main(['-q','-x',*tests,*selection,'--junitxml='+str(output.with_suffix('.xml'))])
        row['pytest_exit_code']=int(code)
        torch.cuda.synchronize(0)
        row['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
        row['reserved_bytes']=torch.cuda.max_memory_reserved(0)
        if mode in COUNTS or mode=='repeat':row['measurements']=json.loads(measurements.read_text())
        if code:raise ValueError('qualification tests failed')
        if mode in COUNTS and len(row['measurements'])!=COUNTS[mode]:raise ValueError('missing physical rows')
        if mode=='repeat' and row['measurements']['repetitions']!=20:raise ValueError('missing repeats')
        if row['peak_allocated_bytes']>64*1024**2:raise MemoryError('CUDA allocation limit')
        cost.check_deadline(deadline)
        row['status']='passed exact-wheel finite CUDA stage; not catalog admission'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(output,row)


def stage(fixtures,output,wheel,expected,mode):
    if output.exists():raise FileExistsError('preserve previous qualification')
    output.parent.mkdir(parents=True,exist_ok=True)
    original=cost.PROFILE
    try:
        cost.PROFILE=dict(original,sampled_working_set_limit_bytes=HOST_BYTES)
        return cost.supervise(fixtures,output,worker_module='tools.shading_cuda_closeout',worker_args=(
            '--wheel',str(wheel),'--sha256',expected,'--mode',mode))
    finally:cost.PROFILE=original


def run(fixtures,output,wheel,expected):
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    source=digest(__file__)
    row=dict(status='working',wheel_sha256=expected,source_sha256=source,stages=[])
    cost.write_report(output/'cohort.json',row)
    try:
        for mode in MODES:
            cost.check_deadline(start+1500-180)
            if digest(__file__)!=source:raise ValueError('source changed during qualification')
            path=output/(mode+'.json')
            code=stage(fixtures,path,wheel,expected,mode)
            if code:raise RuntimeError('CUDA stage failed; no retry or later stage')
            result=json.loads(path.read_text())
            supervision=result['supervisor']
            if (result['mode']!=mode or result['wheel']['sha256']!=expected or result['driver_sha256']!=source
                    or supervision['failure'] is not None or supervision['exit_code']!=0
                    or supervision['monitored_pid']!=result['worker_pid']
                    or not 0<supervision['sampled_peak_working_set_bytes']<=HOST_BYTES
                    or not 0<supervision['elapsed_seconds']<=180):raise ValueError('unqualified stage evidence')
            row['stages'].append(dict(mode=mode,path=path.name,sha256=digest(path)))
            cost.write_report(output/'cohort.json',row)
        cost.check_deadline(start+1500)
        row['status']='passed all five exact-wheel CUDA stages; not catalog admission'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:
        row['elapsed_seconds']=time.perf_counter()-start
        cost.write_report(output/'cohort.json',row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--mode',choices=(*MODES,'all'),required=True)
    args=parser.parse_args()
    if args.worker:worker(args.fixtures,args.output,args.wheel,args.sha256,args.mode);return 0
    if args.mode=='all':run(args.fixtures,args.output,args.wheel,args.sha256);return 0
    return stage(args.fixtures,args.output,args.wheel,args.sha256,args.mode)


if __name__=='__main__':raise SystemExit(main())
