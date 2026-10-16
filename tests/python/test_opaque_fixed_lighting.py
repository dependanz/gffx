"""Scoped fixed lighting uses qualified native arithmetic with exact first VJPs."""
import os,torch,pytest
from contextlib import nullcontext
from test_opaque_full_original import original
from test_native_unit_vectors import values,reference
from test_native_delta_brdf import inputs,bits
from gffx.torch import light,_deferred
from gffx.torch._principled_groups import _fixed_albedo_curve


def scope():
 s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));p=s.prims[1].parameters
 return _fixed_albedo_curve(p['albedo'],p['roughness'],p['budget'])


@pytest.mark.parametrize('module',[light,_deferred])
def test_fixed_replay_normalization_is_native_and_exact(module,monkeypatch):
 x=values(os.environ.get('GFFX_KNOT_DEVICE','cpu'));expected=reference(x)
 def forbidden(*a):raise AssertionError('repeated fixed-input normalization')
 monkeypatch.setattr(module,'sqrt_unit',forbidden)
 with scope(),torch.enable_grad():actual=module._unit(x,'vectors')
 assert bits(actual)==bits(expected)


def test_fixed_replay_normalization_keeps_live_derivatives(monkeypatch):
 x=values(os.environ.get('GFFX_KNOT_DEVICE','cpu'))[:1].requires_grad_()
 with torch.enable_grad():y=light._unit(x,'vectors');expected=torch.autograd.grad(y,x,torch.ones_like(y))[0]
 def forbidden(*a):raise AssertionError('differentiated normalization fused')
 monkeypatch.setattr(torch.ops.gffx,'shading_unit_vectors',forbidden)
 with scope(),torch.enable_grad():
  y=light._unit(x,'vectors');actual=torch.autograd.grad(y,x,torch.ones_like(y))[0]
 assert bits(actual)==bits(expected)


def packet(d,i):return dict(measure='delta',direction=d,irradiance=i,valid=torch.ones(d.shape[:-1],dtype=torch.bool,device=d.device))


@pytest.mark.parametrize('p',[0,1,3,127,128,129,800])
@pytest.mark.parametrize('scale',[1.,-0.,float.fromhex('0x0.0000000000001p-1022')])
def test_fixed_replay_lighting_preserves_value_and_vjp(p,scale,monkeypatch):
 n,b,d,i,g=inputs(p,1);g=g*scale;result=[];calls=[];old=torch.ops.gffx.shading_delta_brdf
 def observed(*a):calls.append(len(a[0]));return old(*a)
 monkeypatch.setattr(torch.ops.gffx,'shading_delta_brdf',observed)
 for scoped in (False,True):
  with scope() if scoped else nullcontext(),torch.enable_grad():
   leaf=b.detach().requires_grad_();out=light.integrate(n,leaf,packet(d,i));grad=torch.autograd.grad(out,leaf,g)[0];result.append((bits(out),bits(grad)))
 assert result[0]==result[1]
 assert len(calls)==2*((p+127)//128) and all(x<=128 for x in calls)


def test_fixed_replay_lighting_uses_native_composition(monkeypatch):
 n,b,d,i,g=inputs(129,1)
 def forbidden(*a):raise AssertionError('eager lighting multiply used')
 monkeypatch.setattr(light,'multiply',forbidden)
 with scope(),torch.enable_grad():
  leaf=b.requires_grad_();out=light.integrate(n,leaf,packet(d,i));torch.autograd.grad(out,leaf,g)


@pytest.mark.parametrize('field',['normal','direction','irradiance','multiple','invalid'])
def test_fixed_replay_lighting_retains_unsupported_fallback(field,monkeypatch):
 n,b,d,i,g=inputs(3,2 if field=='multiple' else 1)
 if field in ('normal','direction','irradiance'):{'normal':n,'direction':d,'irradiance':i}[field].requires_grad_()
 p=packet(d,i)
 if field=='invalid':p['valid'][0,0]=False;d[0,0]=0;i[0,0]=0
 def forbidden(*a):raise AssertionError('unsupported native lighting')
 monkeypatch.setattr(torch.ops.gffx,'shading_delta_brdf',forbidden)
 with scope(),torch.enable_grad():
  leaf=b.requires_grad_();out=light.integrate(n,leaf,p);torch.autograd.grad(out,leaf,g)


@pytest.mark.parametrize('when',['forward','backward'])
def test_fixed_replay_lighting_propagates_native_failure(when,monkeypatch):
 n,b,d,i,g=inputs(3,1);old=torch.ops.gffx.shading_delta_brdf;calls=[]
 def failure(*a):
  calls.append(1)
  if len(calls)==(1 if when=='forward' else 2):raise RuntimeError('injected native lighting failure')
  return old(*a)
 monkeypatch.setattr(torch.ops.gffx,'shading_delta_brdf',failure)
 with scope(),torch.enable_grad(),pytest.raises(RuntimeError,match='injected'):
  leaf=b.requires_grad_();out=light.integrate(n,leaf,packet(d,i));torch.autograd.grad(out,leaf,g)


@pytest.mark.parametrize('bad',['negative','nan','invalid_placeholder','zero_normal','zero_direction'])
def test_fixed_replay_lighting_retains_packet_and_domain_checks(bad):
 n,b,d,i,g=inputs(3,1);p=packet(d,i)
 if bad=='negative':b[0,0,0]=-1
 if bad=='nan':i[0,0,0]=float('nan')
 if bad=='invalid_placeholder':p['valid'][0,0]=False
 if bad=='zero_normal':n[0]=0
 if bad=='zero_direction':d[0]=0
 with scope(),torch.enable_grad(),pytest.raises((RuntimeError,ValueError)):
  light.integrate(n,b.requires_grad_(),p)
