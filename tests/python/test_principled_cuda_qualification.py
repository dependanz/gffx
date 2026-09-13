"""Explicit experimental CUDA-f64 transport, never CPU-profile admission by cast."""
import pytest
import torch


@pytest.fixture(autouse=True)
def cuda_lane():
    if not torch.cuda.is_available():pytest.skip('CUDA qualification requires a real device')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def cpu():
    from gffx.torch.material import PrincipledBudget
    return PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)


def transferred(cpu):return cpu.to(device='cuda:0',dtype=torch.float64,allow_unverified=True)


def test_cuda_transfer_is_explicit_and_cannot_inherit_cpu_admission(cpu):
    with pytest.raises(ValueError):cpu.to(device='cuda:0',dtype=torch.float64)
    gpu=transferred(cpu)
    assert gpu.metadata['admission']=='experimental'
    assert gpu.metadata['acceptance_profile_id'] is None and gpu.metadata['resource_id'] is None
    assert gpu.metadata['execution_device']=='cuda:0' and gpu.metadata['execution_dtype']=='float64'
    assert gpu.to_envelope()==cpu.to_envelope()
    assert all(value.device.type=='cuda' for value in gpu._data.values())
    back=gpu.to(device='cpu',dtype=torch.float64)
    assert back.metadata['admission']=='accepted' and back.to_envelope()==cpu.to_envelope()


@pytest.mark.parametrize('c,r,f',[(.463,.213,.47),(1e-8,.0193,.93),(.947,.827,1.)])
def test_cuda_budget_queries_and_partials_match_cpu(cpu,c,r,f):
    gpu=transferred(cpu)
    def run(resource,device):
        args=[torch.tensor(x,device=device,dtype=torch.float64,requires_grad=True) for x in (c,r,f)]
        value=resource.directional_loss(*args)+resource.mean_loss(*args[1:])
        return [value.detach().cpu(),*[g.detach().cpu() for g in torch.autograd.grad(value,args)]]
    for expected,actual in zip(run(cpu,'cpu'),run(gpu,'cuda:0')):
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)


def test_cuda_budget_rejects_wrong_query_device_and_preserves_input(cpu):
    gpu=transferred(cpu);x=torch.tensor(.5,dtype=torch.float64)
    with pytest.raises(ValueError):gpu.directional_loss(x,x,x)
    assert x.device.type=='cpu' and x.item()==.5


def test_cuda_complete_material_matches_cpu_values_and_all_partials(cpu):
    from gffx.torch.material import principled
    gpu=transferred(cpu)
    def run(resource,device,experimental):
        args=[torch.tensor(x,dtype=torch.float64,device=device,requires_grad=True) for x in
              ([[.1,.2,1.]],[[.2,.1,1.]],[[.3,.4,1.],[-.2,.1,1.]],[.2,.4,.6],.63,.09)]
        kw=dict(albedo=args[3],roughness=args[4],f0=args[5],budget=resource)
        if experimental:
            with pytest.raises(ValueError,match='experimental'):principled(*args[:3],**kw)
        value=principled(*args[:3],**kw,allow_experimental_budget=experimental)
        return [value.detach().cpu(),*[g.detach().cpu() for g in torch.autograd.grad(value.square().sum(),args)]]
    for expected,actual in zip(run(cpu,'cpu',False),run(gpu,'cuda:0',True)):
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)


def test_cuda_supervision_changes_only_host_cap_and_restores_cpu_policy(monkeypatch,tmp_path):
    from tools import cuda_budget_probe as probe,material_loss_cost as cost
    original=cost.PROFILE
    seen=[]
    def supervise(*args,**kwargs):
        seen.append(dict(cost.PROFILE))
        raise RuntimeError('simulated worker failure')
    monkeypatch.setattr(cost,'supervise',supervise)
    with pytest.raises(RuntimeError,match='simulated'):
        probe.supervise_cuda(tmp_path,tmp_path/'fresh.json',worker_args=())
    assert cost.PROFILE is original
    assert seen==[dict(original,sampled_working_set_limit_bytes=2*1024**3)]
    assert original['sampled_working_set_limit_bytes']==1024**3
    assert seen[0]['watchdog_seconds']==180
