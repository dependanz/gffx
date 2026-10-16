"""Explicit constant-albedo VJP keeps every original four-point partial byte-exact."""
import pytest, torch
from test_deferred_native_storage import scale
from test_principled_native_operation import raw

def fixture(device,tail,mode):
 from dataclasses import replace
 from gffx.torch import _deferred as d
 from gffx.torch.scene import Scene
 model,camera,albedo=scale(device,32)
 if mode=='unlit':next(p for p in model.prims if p.name=='key').parameters['irradiance'].zero_()
 if mode=='two_sided':model=Scene([replace(p,parameters=dict(p.parameters,two_sided=True)) if p.name=='skin' else p for p in model.prims])
 with torch.no_grad():
  packed,camera=d._prepare(model,camera);buf=d._buffers(packed,camera,.2,2)
  records={k:buf[k][buf['_covered']] for k in ('normal','world_position','mask')}
 groups=[{k:v[:tail].clone() for k,v in records.items()}]
 groups += [{k:v[start:start+4].clone() for k,v in records.items()} for start in range(4,128,4)]
 return packed,camera,groups,albedo.new_tensor(1/(3*32*32))

def previous_partials(packed,camera,groups,cotangent,prepared):
 from gffx.torch import _deferred as d,light as lighting
 from gffx.torch._principled_groups import _material,_validate_original
 from gffx.torch._reference_channels import sum3
 params=packed.materials[0].parameters;lamp=packed.lights[0]
 keys=tuple(sorted(k for k,v in params.items() if isinstance(v,torch.Tensor)))
 static={k:v for k,v in params.items() if k not in keys};rows=[];leaves=[]
 # These fields have no derivatives in the admitted constant-albedo profile.
 # Pack at most the already admitted128 points; retain independent four-point
 # albedo leaves and exactly the same native group/layout and reduction paths.
 sizes=[len(record['mask']) for record in groups]
 n=torch.cat([record['normal'] for record in groups])
 points=torch.cat([record['world_position'] for record in groups])
 mask=torch.cat([record['mask'] for record in groups])
 v=camera.pose[0,:3,3].to(dtype=points.dtype)-points
 packet=d._light_packet(lamp,packed,0,points);direction,_,valid,_=lighting._packet(n,packet)
 safe=torch.where(valid[...,None],direction,direction.new_tensor([0.,0.,1.]))
 with torch.enable_grad():
  left=0
  for size in sizes:
   leaf=params['albedo'].detach().requires_grad_();local=dict(params,albedo=leaf)
   rows.append((n[left:left+size],v[left:left+size],safe[left:left+size],*(local[k] for k in keys)))
   leaves.append(leaf);left+=size
  _validate_original(keys,static,(n,v,safe,*(params[k] for k in keys)))
  brdf=torch.cat(_material(keys,static,rows,prepared));oriented=n
  if params.get('two_sided',False):
   sign=torch.where(sum3(n*v)<0,-torch.ones_like(n[:,0]),torch.ones_like(n[:,0]));oriented=n*sign[:,None]
  value=points*0
  for field in rows[0][3:]:value=value+field.reshape(-1)[:0].sum()
  value=value+lighting.integrate(oriented,brdf,packet)
  rgb=value*mask[:,None]
  return torch.autograd.grad(rgb,leaves,torch.ones_like(rgb)*cotangent)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('tail',[1,2,3,4])
@pytest.mark.parametrize('mode',['lit','unlit','two_sided'])
def test_explicit_vjp_matches_original_partials_without_eager_backward(device,tail,mode,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 args=fixture(device,tail,mode)
 expected=[previous_partials(*args[:3],sign*args[3],{}) for sign in (1,-1)]
 def forbidden(*args,**kw):raise AssertionError('explicit profile must not build an eager backward graph')
 monkeypatch.setattr(torch.autograd,'grad',forbidden)
 for sign,wanted in zip((1,-1),expected):
  actual=_cohort_partials(*args[:3],sign*args[3],{})
  assert len(actual)==len(wanted)==32
  for got,want in zip(actual,wanted):
   assert not got.requires_grad
   assert torch.equal(raw(got),raw(want))

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('bad',['albedo','roughness','backward_workspace','normal','irradiance'])
def test_explicit_vjp_preserves_value_and_backward_admission_errors(device,bad):
 from tools.deferred_stream import _cohort_partials
 packed,camera,groups,cot=fixture(device,3,'lit');p=packed.materials[0].parameters
 with torch.no_grad():
  if bad=='albedo':p['albedo'][0]=-1
  elif bad=='roughness':p['roughness'].fill_(float('nan'))
  elif bad=='normal':groups[0]['normal'][0].zero_()
  elif bad=='irradiance':packed.lights[0].parameters['irradiance'].fill_(-1)
  else:
   forward=3*p['budget'].workspace(12,backward=False)+8*4*64
   backward=3*p['budget'].workspace(12,backward=True)+8*4*192
   assert forward<backward
   from dataclasses import replace
   packed=replace(packed,materials=(replace(packed.materials[0],parameters=dict(p,workspace_limit_bytes=(forward+backward)//2)),))
 with pytest.raises((ValueError,TypeError)) as old:previous_partials(packed,camera,groups,cot,{})
 with pytest.raises(type(old.value)) as new:_cohort_partials(packed,camera,groups,cot,{})
 assert str(old.value)==str(new.value)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_explicit_vjp_uses_full_native_derivatives_and_propagates_failure(device,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 args=fixture(device,2,'lit');calls=[];forward=torch.ops.gffx.shading_principled_shared_groups
 def observed(*a):
  assert a[-1] is True
  result=forward(*a);calls.append(result[1].shape);return result
 def broken(*a):raise RuntimeError('injected full VJP failure')
 monkeypatch.setattr(torch.ops.gffx,'shading_principled_shared_groups',observed)
 monkeypatch.setattr(torch.ops.gffx,'shading_principled_groups_backward',broken)
 with pytest.raises(RuntimeError,match='injected full VJP failure'):_cohort_partials(*args,{})
 assert len(calls)==1

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_explicit_vjp_does_not_reuse_curve_across_invocations(device):
 from tools.deferred_stream import _cohort_partials
 packed,camera,groups,cot=fixture(device,3,'lit');p=packed.materials[0].parameters
 for rough in (.31,.67):
  with torch.no_grad():p['roughness'].fill_(rough)
  actual=_cohort_partials(packed,camera,groups,cot,{})
  expected=previous_partials(packed,camera,groups,cot,{})
  for got,want in zip(actual,expected):assert torch.equal(raw(got),raw(want))

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('samples',[3,8])
def test_explicit_vjp_keeps_sample_axis_and_1024_pair_boundary(device,samples):
 from dataclasses import replace
 from tools.deferred_stream import _cohort_partials
 packed,camera,groups,cot=fixture(device,4,'two_sided')
 lamp=packed.lights[0];params=dict(lamp.parameters)
 params['directions']=torch.tensor([[.1*i,.05*i,1.] for i in range(samples)],dtype=torch.float64,device=device)
 params['irradiance']=torch.tensor([[.5,.7,.9] for _ in range(samples)],dtype=torch.float64,device=device)
 packed=replace(packed,lights=(replace(lamp,parameters=params),))
 expected=previous_partials(packed,camera,groups,-cot,{})
 actual=_cohort_partials(packed,camera,groups,-cot,{})
 for got,want in zip(actual,expected):assert torch.equal(raw(got),raw(want))
