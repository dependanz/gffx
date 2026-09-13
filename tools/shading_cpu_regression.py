"""Fixed installed-wheel CPU shading regression under the existing worker limits."""
import argparse
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost, shading_cuda_closeout as artifact

TESTS=('material_public_contract','shading_boundary_contract','shading_custom_contract',
       'shading_public_lobes_lights','shading_public_lobe_layouts','shading_public_sources',
       'shading_public_complete','shading_public_edges','classic_material_runtime',
       'principled_runtime','principled_bundle','principled_curve_reuse',
       'material_light_batches','shading_execution_boundary')


def worker(output,wheel,sha):
    row=dict(status='working',worker_pid=os.getpid())
    cost.write_report(output,row);deadline=time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if torch.cuda.is_available():raise ValueError('CPU regression must hide CUDA')
        row['wheel']=artifact.identity(wheel,sha)
        paths=['tests/python/test_'+name+'.py' for name in TESTS]
        row['tests']={path:artifact.digest(path) for path in paths}
        import pytest
        code=pytest.main(['-q','-x','-p','tools.testing_dashboard.pytest_plugin',*paths,'--junitxml='+str(output.with_suffix('.xml'))])
        row['pytest_exit_code']=int(code)
        cost.check_deadline(deadline)
        if code:raise ValueError('CPU regression failed')
        row['status']='passed fixed CPU regression; not whole-repository coverage'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(output,row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True);args=parser.parse_args()
    if args.worker:worker(args.output,args.wheel,args.sha256);return 0
    if args.output.exists():raise FileExistsError('preserve previous evidence')
    return cost.supervise(args.fixtures,args.output,worker_module='tools.shading_cpu_regression',worker_args=(
        '--wheel',str(args.wheel),'--sha256',args.sha256))


if __name__=='__main__':raise SystemExit(main())
