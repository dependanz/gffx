"""Whole-frame admission policy; no large allocation in these controls."""
import importlib
import pytest
import torch
from test_deferred import fixture,api


def test_vram_budget_headroom_and_caller_ceiling():
    choose=importlib.import_module('tools.deferred_vram_budget').choose_budget
    gib=1024**3;mib=1024**2
    x=choose(free_bytes=8*gib,total_bytes=10*gib,allocated_bytes=100*mib)
    assert x['headroom_bytes']==(8*gib+4)//5
    assert x['limit_bytes']==((100*mib+8*gib-x['headroom_bytes'])//mib)*mib
    assert choose(free_bytes=2*gib,total_bytes=10*gib,allocated_bytes=0)['limit_bytes']==gib
    assert choose(free_bytes=8*gib,total_bytes=10*gib,allocated_bytes=0,ceiling_bytes=64*mib)['limit_bytes']==64*mib
    for bad in [dict(free_bytes=-1,total_bytes=10*gib,allocated_bytes=0),dict(free_bytes=11*gib,total_bytes=10*gib,allocated_bytes=0),dict(free_bytes=gib,total_bytes=10*gib,allocated_bytes=0),dict(free_bytes=8*gib,total_bytes=10*gib,allocated_bytes=3*gib)]:
        with pytest.raises(ValueError):choose(**bad)
    with pytest.raises(TypeError):choose(free_bytes=True,total_bytes=10*gib,allocated_bytes=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_deferred_explicit_cuda_budget_preserves_small_outputs_and_gradients(dtype):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    d=api();model,camera,v,f,a=fixture(dtype=dtype,device='cuda');a.requires_grad_()
    values=[]
    for limit in (64*1024**2,256*1024**2):
        out=d.render(model,camera,outputs=('colour','mask'),workspace_limit_bytes=limit)
        cot=torch.arange(out['colour'].numel(),device='cuda',dtype=dtype).reshape(out['colour'].shape)/32-0.5
        grad,=torch.autograd.grad(out['colour'],a,cot)
        values.append((out['colour'].detach(),out['mask'].detach(),grad))
    for x,y in zip(*values):assert x.cpu().contiguous().view(torch.uint8).equal(y.cpu().contiguous().view(torch.uint8))
    total=torch.cuda.get_device_properties(camera.fx.device).total_memory
    with pytest.raises(ValueError,match='capacity'):d._plan_workload(model,camera,1,total+1,('colour','mask'))


def test_deferred_cpu_budget_ceiling_remains_unchanged():
    d=api();model,camera,*_=fixture()
    with pytest.raises(ValueError,match='64 MiB'):d._plan_workload(model,camera,1,64*1024**2+1,('colour','mask'))
