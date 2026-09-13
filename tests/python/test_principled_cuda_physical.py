"""Re-evaluate the fixed physical screen with actual CUDA public queries."""
import importlib.util
import json
import os
from pathlib import Path
import pytest


@pytest.fixture
def measurement(request,monkeypatch,tmp_path):
    root=os.environ.get('GFFX_CUDA_BUDGET_FIXTURES')
    if not root:pytest.skip('explicit supervised CUDA physical worker required')
    import torch
    from gffx.torch.material import PrincipledBudget
    from tools import public_budget_probe as probe,material_loss_cost as cost
    deadline=float(os.environ['GFFX_CUDA_BUDGET_DEADLINE'])
    check=lambda:cost.check_deadline(deadline)
    spec=importlib.util.spec_from_file_location('cuda_budget_controls',Path(__file__).with_name('test_material_loss_resource.py'))
    controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
    controls.SCREENS['public_v1']=(probe.ROUGHNESSES,probe.VIEWS,probe.FRESNELS,.0027)
    def save(request,root,name,rows):
        Path(os.environ['GFFX_CUDA_BUDGET_ROWS']).write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    monkeypatch.setattr(controls,'save_rows',save)
    class Queries:
        # Test-only explicit transport retains gradients back to the independent
        # CPU control's input leaves; the public query itself receives CUDA tensors.
        def __init__(self,value):self.value=value
        def directional_loss(self,c,r,f):
            check();return self.value.directional_loss(c.to('cuda:0'),r.to('cuda:0'),f.to('cuda:0')).to('cpu')
        def mean_loss(self,r,f):
            check();return self.value.mean_loss(r.to('cuda:0'),f.to('cuda:0')).to('cpu')
    resources=[]
    for name,digest in cost.HASHES.items():
        check();data=cost.load_json(Path(root)/(name+'.json'),digest)
        resource=PrincipledBudget.from_envelope(probe.envelope(data,name),device='cuda:0',dtype=torch.float64,allow_unverified=True)
        assert resource.metadata['admission']=='experimental'
        resources.append(probe.PublicComparison(Queries(resource),cost.construct(data),check))
    yield controls,(*resources,tmp_path)
    check()


def test_cuda_directional_values_and_gradients(measurement,request):
    controls,resources=measurement
    controls.test_full_resource_held_out_directional_accuracy_and_gradients(resources,request,'public_v1')


def test_cuda_mean_values_and_gradients(measurement,request):
    controls,resources=measurement
    controls.test_full_resource_mean_matches_reference_and_its_gradients(resources,request,'public_v1')


def test_cuda_furnace_energy_balance(measurement,request):
    controls,resources=measurement
    controls.test_full_resource_independently_integrated_furnace(resources,request,'public_v1')
