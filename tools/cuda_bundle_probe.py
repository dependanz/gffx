"""Exact-wheel physical replay plus explicit named CUDA bundle qualification."""
import argparse
import json
import os
from pathlib import Path
import time
from tools import shading_cuda_closeout as old


def worker(fixtures,output,wheel,expected):
    start=time.perf_counter()
    row=dict(status='working',worker_pid=os.getpid(),driver_sha256=old.digest(__file__))
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=old.identity(wheel,expected)
        if not torch.cuda.is_available():raise ValueError('physical CUDA device required')
        row['device']=dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
                           torch=torch.__version__,cuda=torch.version.cuda)
        torch.cuda.reset_peak_memory_stats(0)
        import pytest
        test='tests/python/test_principled_cuda_bundle.py'
        row['test_sha256']=old.digest(test)
        code=pytest.main(['-q','-x',test,'--junitxml='+str(output.with_suffix('.xml'))])
        row['pytest_exit_code']=int(code)
        torch.cuda.synchronize(0)
        row['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
        row['reserved_bytes']=torch.cuda.memory_reserved(0)
        if code:raise ValueError('named bundle tests failed')
        if row['peak_allocated_bytes']>64*1024**2:raise MemoryError('GPU allocation limit')
        from gffx.torch import _budget_catalog
        row['resource_id']=_budget_catalog.CUDA_RESOURCE_ID
        row['acceptance_record_sha256']=_budget_catalog.CUDA_RECORD_SHA256
        old.cost.check_deadline(start+120)
        row.update(status='passed six installed named CUDA bundle checks',elapsed_seconds=time.perf_counter()-start)
    except Exception as error:
        row.update(status='stopped',error=type(error).__name__+': '+str(error));old.cost.write_report(output,row);raise
    old.cost.write_report(output,row)


def stage(fixtures,output,wheel,expected):
    original=old.cost.PROFILE
    try:
        old.cost.PROFILE=dict(original,sampled_working_set_limit_bytes=2*1024**3)
        return old.cost.supervise(fixtures,output,worker_module='tools.cuda_bundle_probe',worker_args=(
            '--wheel',str(wheel),'--sha256',expected))
    finally:old.cost.PROFILE=original


def run(fixtures,output,wheel,expected):
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    row=dict(status='working',wheel_sha256=expected,driver_sha256=old.digest(__file__),aggregate_seconds=1800)
    old.cost.write_report(output/'admission.json',row)
    try:
        old.run(fixtures,output/'physical',wheel,expected)
        row['physical_cohort_sha256']=old.digest(output/'physical/cohort.json')
        old.cost.check_deadline(start+1800-180)
        path=output/'named.json'
        if stage(fixtures,path,wheel,expected):raise RuntimeError('named bundle stage failed; no retry')
        named=json.loads(path.read_text());sup=named['supervisor']
        if (named['wheel']['sha256']!=expected or named['driver_sha256']!=old.digest(__file__)
                or sup['failure'] is not None or sup['exit_code']!=0 or sup['monitored_pid']!=named['worker_pid']
                or not 0<sup['sampled_peak_working_set_bytes']<=2*1024**3
                or not 0<sup['elapsed_seconds']<=180):raise ValueError('named evidence identity/limits')
        row['named_sha256']=old.digest(path);row['resource_id']=named['resource_id']
        row['acceptance_record_sha256']=named['acceptance_record_sha256']
        old.cost.check_deadline(start+1800)
        row['status']='passed exact-wheel named CUDA admission; finite declared profile only'
    except Exception as error:
        row.update(status='stopped',error=type(error).__name__+': '+str(error));raise
    finally:
        row['elapsed_seconds']=time.perf_counter()-start
        old.cost.write_report(output/'admission.json',row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fixtures','output','wheel'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if args.worker:worker(args.fixtures,args.output,args.wheel,args.sha256)
    else:run(args.fixtures,args.output,args.wheel,args.sha256)
    return 0


if __name__=='__main__':raise SystemExit(main())
