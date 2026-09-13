"""One bounded, exact installed-wheel execution-boundary test process."""
import argparse
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost, shading_fit_closure as fit


def worker(output):
    row = dict(status='working',worker_pid=os.getpid())
    cost.write_report(output,row)
    deadline = time.perf_counter()+120
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['identity'] = fit.identity()
        row['tests_sha256'] = fit.digest('tests/python/test_shading_execution_boundary.py')
        import pytest
        code = pytest.main(['-q','-x','tests/python/test_shading_execution_boundary.py','--junitxml='+str(output.with_suffix('.xml'))])
        row['pytest_exit_code'] = int(code)
        cost.check_deadline(deadline)
        if code: raise ValueError('execution-boundary check failed')
        row['status'] = 'passed negative execution checks; compiled/exported shading remains unsupported'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:cost.write_report(output,row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.worker:worker(args.output);return 0
    if args.output.exists():raise FileExistsError('preserve previous evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    return cost.supervise(args.fixtures,args.output,worker_module='tools.shading_execution_probe')


if __name__=='__main__':raise SystemExit(main())
