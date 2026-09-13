"""Diagnose CUDA resource host footprint under the unchanged worker ceiling."""
import argparse
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost


def worker(output):
    deadline=time.perf_counter()+120
    report=dict(status='working',worker_pid=os.getpid(),stages=[])
    read,close=cost.working_set_reader(os.getpid())
    def mark(stage):
        report['stages'].append(dict(stage=stage,working_set_bytes=read()))
        cost.write_report(output,report);cost.check_deadline(deadline)
    try:
        mark('stdlib')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);mark('torch_import')
        from gffx.torch.material import PrincipledBudget
        mark('material_import')
        cpu=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
        mark('cpu_resource')
        torch.cuda.init();mark('cuda_init')
        gpu=cpu.to(device='cuda:0',dtype=torch.float64,allow_unverified=True);mark('cuda_resource')
        x=torch.tensor(.5,device='cuda:0',dtype=torch.float64,requires_grad=True)
        value=gpu.directional_loss(x,x,x);mark('cuda_query')
        torch.autograd.grad(value,x);torch.cuda.synchronize();mark('cuda_backward')
        report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(0)
        report['status']='diagnostic completed; not qualification'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:close();cost.write_report(output,report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker(a.output)
    else:return cost.supervise(a.fixtures,a.output,worker_module='tools.cuda_budget_cost')


if __name__=='__main__':raise SystemExit(main())
