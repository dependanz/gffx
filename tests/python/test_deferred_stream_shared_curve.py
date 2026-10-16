"""A fixed albedo stream shares one native curve without dropping Jacobian columns."""
import pytest, torch
from test_deferred_stream_explicit_vjp import fixture
from test_principled_native_operation import raw

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('tail',[1,4])
def test_shared_stream_table_keeps_dense_gradient_bytes_and_full_jacobian(device,tail,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 packed,camera,groups,cot=fixture(device,tail,'two_sided')
 dense=torch.ops.gffx.shading_principled_groups
 shared=torch.ops.gffx.shading_principled_shared_groups
 seen=[]
 def observed(*args):
  assert args[2].shape[0]==args[3].shape[0]==args[4].shape[0]==args[5].shape[0]==1
  assert args[-1] is True
  out=shared(*args);assert out[1].shape[-2:]==(3,16)
  seen.append(1);return out
 for factor in (1.,-1.,1e-280):
  expected=_cohort_partials(packed,camera,groups,cot*factor,None)
  with monkeypatch.context() as m:
   def forbidden(*a):raise AssertionError('shared stream must not expand its table into dense rows')
   m.setattr(torch.ops.gffx,'shading_principled_groups',forbidden)
   m.setattr(torch.ops.gffx,'shading_principled_shared_groups',observed)
   actual=_cohort_partials(packed,camera,groups,cot*factor,{})
  assert len(actual)==len(expected)==32
  for a,b in zip(actual,expected):assert torch.equal(raw(a),raw(b))
  assert any(bool((a!=0).any()) for a in actual)
 assert len(seen)==3

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_stream_reuses_table_only_inside_explicit_invocation(device,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 from gffx.torch import _principled_groups as groups_module
 packed,camera,groups,cot=fixture(device,3,'lit');calls=[]
 original=groups_module._prepare_material_curve
 def observe(*a,**k):calls.append(1);return original(*a,**k)
 monkeypatch.setattr(groups_module,'_prepare_material_curve',observe)
 prepared={}
 first=_cohort_partials(packed,camera,groups,cot,prepared)
 second=_cohort_partials(packed,camera,groups,cot,prepared)
 assert len(calls)==1
 for a,b in zip(first,second):assert torch.equal(raw(a),raw(b))
 _cohort_partials(packed,camera,groups,cot,{})
 assert len(calls)==2
