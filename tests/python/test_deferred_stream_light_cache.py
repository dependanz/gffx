"""Fixed directional light preparation belongs to one guarded stream invocation."""
import pytest,torch
from test_deferred_stream_explicit_vjp import fixture,previous_partials
from test_principled_native_operation import raw

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_directional_packet_is_prepared_once_for_different_group_sizes(device,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 from gffx.torch import _deferred as d
 packed,camera,groups,cot=fixture(device,3,'two_sided')
 expected=[previous_partials(packed,camera,g,cot,{}) for g in (groups,groups[:3])]
 call=d._light_packet;calls=[]
 def observed(*a):calls.append(len(a[-1]));return call(*a)
 monkeypatch.setattr(d,'_light_packet',observed);state={}
 for g,wanted in zip((groups,groups[:3]),expected):
  actual=_cohort_partials(packed,camera,g,cot,state)
  assert all(torch.equal(raw(a),raw(b)) for a,b in zip(actual,wanted))
 assert calls==[1], 'fixed directional packet should be prepared once, then expanded'

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_light_cache_cannot_escape_an_invocation(device):
 from tools.deferred_stream import _cohort_partials
 packed,camera,groups,cot=fixture(device,4,'lit')
 for direction in ((.1,.2,1.),(-.3,.4,1.)):
  with torch.no_grad():packed.lights[0].parameters['directions'].copy_(torch.tensor([direction],dtype=torch.float64,device=device))
  actual=_cohort_partials(packed,camera,groups,cot,{})
  expected=previous_partials(packed,camera,groups,cot,{})
  assert all(torch.equal(raw(a),raw(b)) for a,b in zip(actual,expected))

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_light_cache_keeps_larger_packets_on_uncached_path(device,monkeypatch):
 from dataclasses import replace
 from tools.deferred_stream import _cohort_partials
 from gffx.torch import _deferred as d
 packed,camera,groups,cot=fixture(device,4,'lit');groups=groups[:1]
 lamp=packed.lights[0];params=dict(lamp.parameters)
 params['directions']=torch.tensor([[.1*i,.05*i,1.] for i in range(9)],dtype=torch.float64,device=device)
 params['irradiance']=torch.ones((9,3),dtype=torch.float64,device=device)
 packed=replace(packed,lights=(replace(lamp,parameters=params),));expected=previous_partials(packed,camera,groups,cot,{})
 call=d._light_packet;calls=[]
 def observed(*a):calls.append(len(a[-1]));return call(*a)
 monkeypatch.setattr(d,'_light_packet',observed);state={}
 for _ in range(2):
  actual=_cohort_partials(packed,camera,groups,cot,state)
  assert all(torch.equal(raw(a),raw(b)) for a,b in zip(actual,expected))
 assert calls==[4,4] and 'light_packet' not in state
