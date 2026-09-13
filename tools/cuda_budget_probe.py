"""Supervised CUDA-f64 physical qualification, without automatic admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
from tools import material_loss_cost as cost,public_budget_probe as original
from tools.package_principled_budget import validate_reports

WHEEL='4c2b5c5d4dbd1bae064123754a476b235717176a37f1010fff53faebaee00ba9'
MODES=('directional','mean','furnace')
COUNTS=dict(directional=80,mean=16,furnace=192)
HOST_BYTES=2*1024**3  # User-approved CUDA-only qualification ceiling, 2026-09-08.


def supervise_cuda(fixtures,output,*,worker_args):
    # A serial CLI-only override; never mutate the frozen CPU profile object or
    # its source. Other processes (including active CPU workers) are unaffected.
    original_profile=cost.PROFILE
    try:
        cost.PROFILE=dict(original_profile,sampled_working_set_limit_bytes=HOST_BYTES)
        return cost.supervise(fixtures,output,worker_module='tools.cuda_budget_probe',worker_args=worker_args)
    finally:
        cost.PROFILE=original_profile


def identity(wheel):
    import gffx
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=WHEEL:raise ValueError('CUDA wheel identity mismatch')
    root=Path(gffx.__file__).resolve().parent.parent
    with zipfile.ZipFile(wheel) as z:
        names=[n for n in z.namelist() if n.startswith('gffx/') and not n.endswith('/')]
        if any((root/n).read_bytes()!=z.read(n) for n in names):raise ValueError('installed wheel member mismatch')
    return dict(sha256=WHEEL,installed_root=str(root),matched_members=len(names))


def worker(fixtures,output,wheel,mode,prior):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    deadline=time.perf_counter()+120
    row=dict(status='working',worker_pid=os.getpid(),mode=mode,rows=[],profile=dict(
        name='cuda-budget-f64-qualification-v2-host-2gib',device='cuda:0',dtype='float64',expected_rows=COUNTS[mode],
        cooperative_seconds=120,watchdog_seconds=180,worker_bytes=HOST_BYTES,gpu_allocated_bytes=64*1024**2))
    cost.write_report(output,row)
    try:
        row['wheel']=identity(wheel);row['references']=original.verify_references(Path(__file__).parent)
        row['cpu_reference_reports']=validate_reports(Path('build/public-budget-20260908'))
        row['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        row['tests_sha256']=hashlib.sha256(Path('tests/python/test_principled_cuda_physical.py').read_bytes()).hexdigest()
        if mode!='directional':
            if prior is None:raise ValueError('previous CUDA stage required')
            blob=prior.read_bytes();p=json.loads(blob);s=p.get('supervisor',{})
            if (p['mode']!=MODES[MODES.index(mode)-1] or p['status']!='passed; not CUDA admission'
                or p['wheel']['sha256']!=WHEEL or s.get('exit_code')!=0 or s.get('failure') is not None
                or p['source_sha256']!=row['source_sha256'] or p['tests_sha256']!=row['tests_sha256']):raise ValueError('unqualified CUDA predecessor')
            row['prior_sha256']=hashlib.sha256(blob).hexdigest()
        if not torch.cuda.is_available():raise ValueError('CUDA device unavailable')
        row['device']=dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),torch=torch.__version__)
        torch.cuda.reset_peak_memory_stats(0)
        os.environ['GFFX_CUDA_BUDGET_FIXTURES']=str(fixtures.resolve())
        os.environ['GFFX_CUDA_BUDGET_DEADLINE']=str(deadline)
        rows_path=output.with_suffix('.rows.json');os.environ['GFFX_CUDA_BUDGET_ROWS']=str(rows_path.resolve())
        import pytest
        code=pytest.main(['-q','tests/python/test_principled_cuda_physical.py','-k',mode,'--junitxml='+str(output.with_suffix('.xml'))])
        torch.cuda.synchronize(0);row['pytest_exit_code']=int(code)
        row['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(0)
        row['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved(0)
        if rows_path.exists():row['rows']=json.loads(rows_path.read_bytes())
        if code or len(row['rows'])!=COUNTS[mode]:raise ValueError('CUDA physical check failed or incomplete')
        if row['peak_cuda_allocated_bytes']>64*1024**2:raise MemoryError('CUDA allocation ceiling exceeded')
        cost.check_deadline(deadline);row['status']='passed; not CUDA admission'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(output,row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--prior',type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker(a.fixtures,a.output,a.wheel,a.mode,a.prior)
    else:return supervise_cuda(a.fixtures,a.output,worker_args=(
        '--wheel',str(a.wheel),'--mode',a.mode,*(('--prior',str(a.prior)) if a.prior else ())))


if __name__=='__main__':raise SystemExit(main())
