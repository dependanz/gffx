"""Cohort orchestration must retain each original four-point partial."""
import pytest,torch
from test_deferred_native_storage import scale
from test_principled_native_operation import raw

def reference_partials(packed,camera,groups,cotangent,prepared):
 from gffx.torch import _deferred as d,light as lighting
 from gffx.torch._principled_groups import _material,_validate_original
 from gffx.torch._reference_channels import sum3
 params=packed.materials[0].parameters;lamp=packed.lights[0]
 keys=tuple(sorted(k for k,v in params.items() if isinstance(v,torch.Tensor)))
 static={k:v for k,v in params.items() if k not in keys};rows=[];packets=[];leaves=[]
 with torch.enable_grad():
  for record in groups:
   n=record['normal'];points=record['world_position'];v=camera.pose[0,:3,3].to(dtype=points.dtype)-points
   packet=d._light_packet(lamp,packed,0,points);direction,_,valid,_=lighting._packet(n,packet)
   safe=torch.where(valid[...,None],direction,direction.new_tensor([0.,0.,1.]))
   leaf=params['albedo'].detach().requires_grad_();local=dict(params,albedo=leaf)
   rows.append((n,v,safe,*(local[k] for k in keys)));packets.append(packet);leaves.append(leaf)
  original=tuple(torch.cat([row[k] for row in rows]) for k in range(3))+tuple(params[k] for k in keys)
  _validate_original(keys,static,original)
  outputs=_material(keys,static,rows,prepared);rgbs=[]
  for row,brdf,packet,record in zip(rows,outputs,packets,groups):
   n,v=row[:2];oriented=n
   if params.get('two_sided',False):
    sign=torch.where(sum3(n*v)<0,-torch.ones_like(n[:,0]),torch.ones_like(n[:,0]));oriented=n*sign[:,None]
   value=record['world_position']*0
   for field in row[3:]:value=value+field.reshape(-1)[:0].sum()
   value=value+lighting.integrate(oriented,brdf,packet)
   rgbs.append(value*record['mask'][:,None])
  return torch.autograd.grad(rgbs,leaves,[torch.ones_like(rgb)*cotangent for rgb in rgbs])


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('tail',[1,3,4])
@pytest.mark.parametrize('mode',['lit','unlit','two_sided'])
def test_stream_cohort_keeps_each_original_partial_and_prepares_light_once(device,tail,mode,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 from gffx.torch import _deferred as d
 model,camera,albedo=scale(device,32)
 if mode=='unlit':
  next(p for p in model.prims if p.name=='key').parameters['irradiance'].zero_()
 if mode=='two_sided':
  from dataclasses import replace
  from gffx.torch.scene import Scene
  model=Scene([replace(p,parameters=dict(p.parameters,two_sided=True)) if p.name=='skin' else p for p in model.prims])
 with torch.no_grad():
  packed,camera=d._prepare(model,camera);buf=d._buffers(packed,camera,.2,2)
  select=buf['_covered'];records={k:buf[k][select] for k in ('world_position','normal','mask')}
 groups=[{k:v[:tail].clone() for k,v in records.items()}]
 for start in range(4,128,4):groups.append({k:v[start:start+4].clone() for k,v in records.items()})
 cotangent=albedo.new_tensor(1/(3*32*32))
 expected=reference_partials(packed,camera,groups,cotangent,{})
 calls=[];original=d._light_packet
 def observe(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
 monkeypatch.setattr(d,'_light_packet',observe)
 actual=_cohort_partials(packed,camera,groups,cotangent,{})
 assert len(actual)==len(expected)==32
 for got,want in zip(actual,expected):assert torch.equal(raw(got),raw(want))
 assert len(calls)==1, len(calls)
