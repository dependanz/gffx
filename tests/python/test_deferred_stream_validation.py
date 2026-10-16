"""Combined fixed-cohort predicates preserve original diagnostic semantics."""
import pytest, torch
from test_deferred_stream_explicit_vjp import fixture

def arguments(device):
 from gffx.torch import _deferred as d
 packed,camera,groups,_=fixture(device,4,'lit')
 n=torch.cat([g['normal'] for g in groups]);points=torch.cat([g['world_position'] for g in groups])
 v=camera.pose[0,:3,3]-points
 packet=d._light_packet(packed.lights[0],packed,0,points)
 return n,v,packet,dict(packed.materials[0].parameters)

def original(n,v,packet,params):
 from gffx.torch import light
 from gffx.torch._principled_groups import _validate_original
 direction,incoming,valid,omega=light._packet(n,packet)
 safe=torch.where(valid[...,None],direction,direction.new_tensor([0.,0.,1.]))
 keys=tuple(sorted(k for k,x in params.items() if isinstance(x,torch.Tensor)))
 static={k:x for k,x in params.items() if k not in keys}
 _validate_original(keys,static,(n,v,safe,*(params[k] for k in keys)))
 return direction,incoming,valid,omega,safe

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('bad',['normal_nan','view_inf','direction_nan','incoming_inf','albedo_nan','roughness_nan','f0_inf','albedo_domain','f0_domain','roughness_domain','incoming_negative','placeholder_direction','placeholder_incoming','valid_dtype','normal_dtype','view_shape','packet_keys','ordered_errors','empty','signed_tiny'])
def test_combined_cohort_validation_preserves_original(device,bad):
 from tools.deferred_stream import _validate_cohort_inputs
 n,v,packet,p=arguments(device)
 packet={k:x.clone() if isinstance(x,torch.Tensor) else x for k,x in packet.items()}
 p={k:x.clone() if isinstance(x,torch.Tensor) else x for k,x in p.items()}
 with torch.no_grad():
  if bad=='normal_nan':n[0,0]=float('nan')
  elif bad=='view_inf':v[0,0]=float('inf')
  elif bad=='direction_nan':packet['direction'][0,0,0]=float('nan')
  elif bad=='incoming_inf':packet['irradiance'][0,0,0]=float('inf')
  elif bad=='albedo_nan':p['albedo'][0]=float('nan')
  elif bad=='roughness_nan':p['roughness'].fill_(float('nan'))
  elif bad=='f0_inf':p['f0'].fill_(float('inf'))
  elif bad=='albedo_domain':p['albedo'][0]=-1e-200
  elif bad=='f0_domain':p['f0'].fill_(1.01)
  elif bad=='roughness_domain':p['roughness'].fill_(p['budget']._metadata['domain']['roughness'][0]-.001)
  elif bad=='incoming_negative':packet['irradiance'][0,0,0]=-1e-200
  elif bad.startswith('placeholder_'):
   packet['valid'][0,0]=False;packet['direction'][0,0]=0;packet['irradiance'][0,0]=0
   packet['direction' if bad.endswith('direction') else 'irradiance'][0,0,0]=1e-200
  elif bad=='valid_dtype':packet['valid']=packet['valid'].to(torch.int32)
  elif bad=='normal_dtype':n=n.float()
  elif bad=='view_shape':v=v[:1]
  elif bad=='packet_keys':packet['extra']=0
  elif bad=='ordered_errors':n[0,0]=float('nan');p['albedo'][0]=-1;packet['irradiance'][0,0,0]=-1
  elif bad=='empty':n=n[:0];v=v[:0];packet={k:x[:0] if isinstance(x,torch.Tensor) else x for k,x in packet.items()}
  elif bad=='signed_tiny':p['albedo'][0]=-0.;p['albedo'][1]=1e-200
 try:expected=original(n,v,packet,p)
 except (ValueError,TypeError) as error:
  with pytest.raises(type(error)) as actual:_validate_cohort_inputs(n,v,packet,p)
  assert str(actual.value)==str(error)
 else:
  actual=_validate_cohort_inputs(n,v,packet,p)
  for a,b in zip(actual,expected):
   if a is None:assert b is None
   else:assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

@pytest.mark.parametrize('device',['cuda'])
def test_combined_cohort_checks_finite_fields_once(device,monkeypatch):
 from tools.deferred_stream import _validate_cohort_inputs
 args=arguments(device);calls=[];real=torch.isfinite
 def observed(x):calls.append(x.numel());return real(x)
 monkeypatch.setattr(torch,'isfinite',observed)
 _validate_cohort_inputs(*args)
 assert len(calls)==1
 assert calls[0]==1541
