"""Native lighting fusion keeps the previous fixed-profile gradient bytes."""
import pytest,torch,importlib.util
from pathlib import Path
from test_deferred_stream_explicit_vjp import fixture
from test_principled_native_operation import raw

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_delta_brdf_uses_native_checked_lighting(device,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 from gffx.torch import light
 args=fixture(device,3,'two_sided');operation=torch.ops.gffx.shading_delta_brdf;calls=[]
 def observe(*a):calls.append(a[1].shape);return operation(*a)
 def forbidden(*a):raise AssertionError('repeated eager lighting integration')
 monkeypatch.setattr(torch.ops.gffx,'shading_delta_brdf',observe);monkeypatch.setattr(light,'integrate',forbidden)
 result=_cohort_partials(*args,{})
 assert len(calls)==1 and calls[0]==(127,1,3) and len(result)==32

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_delta_brdf_retains_zero_and_tiny_complete_group_partials(device):
 from tools.deferred_stream import _cohort_partials
 from dataclasses import replace
 path=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque200-20261002/deferred_stream-before.py')
 spec=importlib.util.spec_from_file_location('old_stream',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 packed,camera,groups,cot=fixture(device,3,'two_sided')
 for q in (1,3,8):
  lamp=packed.lights[0];params=dict(lamp.parameters)
  params['directions']=torch.tensor([[.1*i,.05*i,1.] for i in range(q)],dtype=torch.float64,device=device)
  params['irradiance']=torch.ones((q,3),dtype=torch.float64,device=device)
  local=replace(packed,lights=(replace(lamp,parameters=params),))
  for scale in (-0.,1e-280,-1e-280):
   actual=_cohort_partials(local,camera,groups,cot*scale,{})
   expected=old._cohort_partials(local,camera,groups,cot*scale,{})
   assert all(torch.equal(raw(a),raw(b)) for a,b in zip(actual,expected))
