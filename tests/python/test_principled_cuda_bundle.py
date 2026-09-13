"""Installed named CUDA bundle: acceptance does not leak into generic transfers."""
import importlib.resources
import hashlib
import json
import pytest
import torch

ID='ggx-smith-loss-cuda-f64-v1'


@pytest.fixture(autouse=True)
def lane():
    if not torch.cuda.is_available():pytest.skip('named CUDA bundle requires a physical CUDA device')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def pair():
    from gffx.torch.material import PrincipledBudget
    cpu=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    gpu=PrincipledBudget.bundled(ID,device='cuda:0',dtype=torch.float64)
    return cpu,gpu


def test_named_cuda_bundle_has_exact_installed_receipt_and_local_tensors(pair):
    from gffx.torch import _budget_catalog
    cpu,gpu=pair
    receipt=importlib.resources.files('gffx').joinpath('data','principled','public-budget-cuda-v0.1.json').read_bytes()
    assert hashlib.sha256(receipt).hexdigest()==_budget_catalog.CUDA_RECORD_SHA256
    assert json.loads(receipt)['resource_id']==ID
    assert gpu.metadata['admission']=='accepted' and gpu.metadata['resource_id']==ID
    assert gpu.metadata['acceptance_profile_id']=='public-budget-cuda-v0.1'
    assert gpu.metadata['acceptance_record_sha256']==_budget_catalog.CUDA_RECORD_SHA256
    assert gpu.metadata['execution_device']=='cuda:0'
    assert gpu.to_envelope()==cpu.to_envelope()
    assert all(x.device==torch.device('cuda:0') for x in gpu._data.values())


def test_named_cuda_bundle_does_not_promote_generic_transfer_or_loading(pair):
    from gffx.torch.material import PrincipledBudget
    cpu,gpu=pair
    with pytest.raises(ValueError):cpu.to(device='cuda:0',dtype=torch.float64)
    with pytest.raises(ValueError):PrincipledBudget.from_envelope(gpu.to_envelope(),device='cuda:0',dtype=torch.float64)
    research=cpu.to(device='cuda:0',dtype=torch.float64,allow_unverified=True)
    assert research.metadata['admission']=='experimental' and research.metadata['resource_id'] is None
    moved=gpu.to(device='cuda:0',dtype=torch.float64,allow_unverified=True)
    assert moved.metadata['admission']=='experimental'


@pytest.mark.parametrize('device,dtype',[('cpu',torch.float64),('cuda:0',torch.float32)])
def test_named_cuda_bundle_rejects_a_different_lane(device,dtype):
    from gffx.torch.material import PrincipledBudget
    with pytest.raises(ValueError):PrincipledBudget.bundled(ID,device=device,dtype=dtype)


def test_named_cuda_bundle_matches_experimental_and_cpu_query_partials(pair):
    cpu,gpu=pair;research=cpu.to(device='cuda:0',dtype=torch.float64,allow_unverified=True)
    results=[]
    for resource,device in ((cpu,'cpu'),(research,'cuda:0'),(gpu,'cuda:0')):
        fields=[torch.tensor(x,device=device,dtype=torch.float64,requires_grad=True) for x in (.463,.213,.47)]
        value=resource.directional_loss(*fields)+resource.mean_loss(*fields[1:])
        results.append([value.detach().cpu(),*[g.cpu() for g in torch.autograd.grad(value,fields)]])
    for expected,actual in zip(results[0],results[2]):torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
    for expected,actual in zip(results[1],results[2]):assert torch.equal(actual,expected)


def test_named_cuda_bundle_material_needs_no_experimental_flag(pair):
    from gffx.torch.material import principled
    cpu,gpu=pair;research=cpu.to(device='cuda:0',dtype=torch.float64,allow_unverified=True)
    outputs=[]
    for resource,experimental in ((research,True),(gpu,False)):
        fields=[torch.tensor(x,dtype=torch.float64,device='cuda:0',requires_grad=True) for x in
            ([[.1,.2,1.]],[[.2,.1,1.]],[[.3,.4,1.],[-.2,.1,1.]],[.2,.4,.6],.63,.09)]
        kwargs=dict(albedo=fields[3],roughness=fields[4],f0=fields[5],budget=resource)
        if experimental:kwargs['allow_experimental_budget']=True
        value=principled(*fields[:3],**kwargs)
        outputs.append([value.detach(),*torch.autograd.grad(value.square().sum(),fields)])
    for expected,actual in zip(*outputs):assert torch.equal(actual,expected)
