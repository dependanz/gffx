"""Native checks replace duplicate private-helper scans without losing errors."""
import pytest,torch,importlib.util,importlib.machinery
from pathlib import Path
from test_deferred_stream_explicit_vjp import fixture

def old_helper():
 p=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque206-20261002/deferred_stream.py.before')
 loader=importlib.machinery.SourceFileLoader('tools._prior_cohort',str(p));spec=importlib.util.spec_from_loader(loader.name,loader);m=importlib.util.module_from_spec(spec);loader.exec_module(m);return m

@pytest.mark.parametrize('bad',['normal_nan','view_inf','albedo_nan','f0_inf','roughness_nan','albedo_domain','f0_domain','roughness_domain','incoming_nan','incoming_negative','direction_nan','zero_normal','multiple_errors'])
def test_native_cohort_validation_retains_prior_errors(bad):
 from tools.deferred_stream import _cohort_partials
 args=fixture('cuda',4,'lit');packed,camera,groups,cot=args;p=packed.materials[0].parameters
 state={};_cohort_partials(*args,state);packet=state['light_packet']
 with torch.no_grad():
  if bad=='normal_nan':groups[0]['normal'][0,0]=float('nan')
  elif bad=='view_inf':groups[0]['world_position'][0,0]=float('inf')
  elif bad=='albedo_nan':p['albedo'][0]=float('nan')
  elif bad=='f0_inf':p['f0'].fill_(float('inf'))
  elif bad=='roughness_nan':p['roughness'].fill_(float('nan'))
  elif bad=='albedo_domain':p['albedo'][0]=-1e-200
  elif bad=='f0_domain':p['f0'].fill_(1.01)
  elif bad=='roughness_domain':p['roughness'].fill_(p['budget']._metadata['domain']['roughness'][0]-.001)
  elif bad=='incoming_nan':packet['irradiance'].fill_(float('nan'))
  elif bad=='incoming_negative':packet['irradiance'].fill_(-1e-200)
  elif bad=='direction_nan':packet['direction'].fill_(float('nan'))
  elif bad=='zero_normal':groups[0]['normal'][0].zero_()
  elif bad=='multiple_errors':p['albedo'][0]=float('nan');packet['irradiance'].fill_(-1.)
 with pytest.raises((ValueError,RuntimeError,TypeError)) as expected:old_helper()._cohort_partials(*args,state)
 with pytest.raises(type(expected.value)) as got:_cohort_partials(*args,state)
 assert str(got.value)==str(expected.value)


def test_native_cohort_does_not_duplicate_native_finite_scan(monkeypatch):
 from tools.deferred_stream import _cohort_partials
 args=fixture('cuda',4,'lit');state={};_cohort_partials(*args,state)
 real=torch.isfinite;scans=[]
 def observed(x):
  if x.numel()==1541:scans.append(1)
  return real(x)
 monkeypatch.setattr(torch,'isfinite',observed)
 actual=_cohort_partials(*args,state)
 assert len(actual)==32
 assert not scans,'native-checked cohort values are redundantly rescanned'
