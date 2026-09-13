"""Finite same-device repeat/allocation checks, not sustained performance."""
import gc
import os
import time
import pytest


def test_cuda_repeat_harness_has_finite_fixed_limits():
    from tools import cuda_repetition as driver
    assert driver.PROFILE==dict(host_bytes=2*1024**3,gpu_allocated_bytes=64*1024**2,
        cooperative_seconds=120,watchdog_seconds=180,repetitions=20,postwarm_live_growth_bytes=65536)


def test_cuda_material_repetition_has_fresh_graphs_and_bounded_live_allocations():
    if 'GFFX_CUDA_REPEAT_DEADLINE' not in os.environ:
        pytest.skip('explicit supervised repeat worker required')
    import torch
    from gffx.torch.material import PrincipledBudget,principled
    from tools import material_loss_cost as cost,cuda_repetition as driver
    deadline=float(os.environ['GFFX_CUDA_REPEAT_DEADLINE'])
    torch.use_deterministic_algorithms(True)
    resource=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    gpu=resource.to(device='cuda:0',dtype=torch.float64,allow_unverified=True)
    before={key:value.clone() for key,value in gpu._data.items()}
    def evaluate():
        args=[torch.tensor(v,device='cuda:0',dtype=torch.float64,requires_grad=True) for v in
              ([[.1,.2,1.]]*4,[[.2,.1,1.]]*4,[[.3,.4,1.]]*256,[.2,.4,.6],.63,.09)]
        output=principled(*args[:3],albedo=args[3],roughness=args[4],f0=args[5],budget=gpu,
                          allow_experimental_budget=True)
        grads=torch.autograd.grad(output.sum(),args)
        return [output.detach().cpu(),*[value.detach().cpu() for value in grads]]
    expected=evaluate();gc.collect();torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated(0)
    peaks=[];live=[];started=time.perf_counter()
    for _ in range(driver.PROFILE['repetitions']):
        cost.check_deadline(deadline);torch.cuda.reset_peak_memory_stats(0)
        actual=evaluate();torch.cuda.synchronize()
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))
        peaks.append(torch.cuda.max_memory_allocated(0))
        del actual;gc.collect();torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated(0))
        assert live[-1]<=baseline+driver.PROFILE['postwarm_live_growth_bytes']
        assert peaks[-1]<=driver.PROFILE['gpu_allocated_bytes']
    assert all(torch.equal(value,before[key]) for key,value in gpu._data.items())
    driver.save_result(dict(repetitions=20,points=4,lights=256,pairs=1024,deterministic=True,
        elapsed_seconds=time.perf_counter()-started,baseline_allocated_bytes=baseline,
        peak_allocated_bytes=max(peaks),live_allocated_bytes=live,
        reserved_bytes=torch.cuda.memory_reserved(0)))
