"""Identical constant material layouts may share setup within one call only."""
import pytest,torch
from test_deferred_stream_explicit_vjp import fixture,previous_partials
from test_principled_native_operation import raw

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('tail',[1,3,4])
def test_constant_cohort_reuses_layout_without_changing_partials(device,tail,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 from gffx.torch import material
 args=fixture(device,tail,'two_sided');wanted=previous_partials(*args,{})
 calls=[];original=material._principled_layout
 def observe(*a,**kw):calls.append(1);return original(*a,**kw)
 monkeypatch.setattr(material,'_principled_layout',observe)
 got=_cohort_partials(*args,{})
 for a,b in zip(got,wanted):assert torch.equal(raw(a),raw(b))
 assert len(calls)<=2,len(calls)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('tail',[1,3,4])
def test_constant_forward_reuses_layout_with_exact_group_outputs(device,tail,monkeypatch):
 from gffx.torch import _deferred as d,light,material
 from gffx.torch._principled_groups import _material,_validate_original,_GroupedMaterial
 packed,camera,groups,cot=fixture(device,tail,'lit');params=packed.materials[0].parameters
 keys=tuple(sorted(k for k,v in params.items() if isinstance(v,torch.Tensor)));static={k:v for k,v in params.items() if k not in keys}
 rows=[];fields=[];sizes=[]
 with torch.no_grad():
  for group in groups:
   n=group['normal'];points=group['world_position'];v=camera.pose[0,:3,3]-points
   packet=d._light_packet(packed.lights[0],packed,0,points);direction,_,valid,_=light._packet(n,packet)
   safe=torch.where(valid[...,None],direction,direction.new_tensor([0.,0.,1.]))
   row=(n,v,safe,*(params[k] for k in keys));rows.append(row)
   layout,(_,p,_),_=material._principled_layout(n,v,safe,**params);fields.extend(layout);sizes.append(p)
  wanted=_GroupedMaterial.apply(params['budget'],False,False,tuple(sizes),{},*fields)
  calls=[];original=material._principled_layout
  def observe(*a,**kw):calls.append(1);return original(*a,**kw)
  monkeypatch.setattr(material,'_principled_layout',observe)
  got=_material(keys,static,rows,{})
  for a,b in zip(got,wanted):assert torch.equal(raw(a),raw(b))
  assert len(calls)<=2,len(calls)
