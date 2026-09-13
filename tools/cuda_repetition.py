"""Finite CUDA material forward/backward repetition on an exact isolated wheel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost,cuda_budget_probe as physical

PROFILE=dict(host_bytes=2*1024**3,gpu_allocated_bytes=64*1024**2,
    cooperative_seconds=120,watchdog_seconds=180,repetitions=20,postwarm_live_growth_bytes=65536)


def save_result(result):
    cost.write_report(Path(os.environ['GFFX_CUDA_REPEAT_RESULTS']),result)


def worker(output,wheel):
    deadline=time.perf_counter()+PROFILE['cooperative_seconds']
    row=dict(status='working',worker_pid=os.getpid(),profile=PROFILE)
    cost.write_report(output,row)
    try:
        row['wheel']=physical.identity(wheel)
        row['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        row['tests_sha256']=hashlib.sha256(Path('tests/python/test_cuda_repetition.py').read_bytes()).hexdigest()
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        result=output.with_suffix('.measurements.json')
        os.environ['GFFX_CUDA_REPEAT_DEADLINE']=str(deadline)
        os.environ['GFFX_CUDA_REPEAT_RESULTS']=str(result)
        import pytest
        code=pytest.main(['-q','tests/python/test_cuda_repetition.py','--junitxml='+str(output.with_suffix('.xml'))])
        if code:raise ValueError('CUDA repetition test failed')
        row['measurement']=json.loads(result.read_bytes());cost.check_deadline(deadline)
        row['status']='passed finite repeats; not sustained acceptance'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(output,row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker(a.output,a.wheel)
    else:
        previous=cost.PROFILE
        try:
            cost.PROFILE=dict(previous,sampled_working_set_limit_bytes=PROFILE['host_bytes'])
            return cost.supervise(a.fixtures,a.output,worker_module='tools.cuda_repetition',
                worker_args=('--wheel',str(a.wheel)))
        finally:cost.PROFILE=previous


if __name__=='__main__':raise SystemExit(main())
