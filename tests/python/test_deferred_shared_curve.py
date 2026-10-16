"""Call-local shared scalar curves preserve independent point VJPs and refresh."""
import torch,pytest
from test_deferred_groups import case,ordered_reference
from test_principled_native_operation import raw


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_roughness_curve_rebuilt_once_per_pass_and_after_parameter_update(monkeypatch,device):
 from gffx.torch import _deferred as d
 budget,base=case(129,3,device);n,v,l,a,r,f=base
 directions=l.detach()[None].expand(129,-1,-1).clone().requires_grad_();args=(n,v,directions,a,r,f)
 cot=(torch.arange(1161,dtype=n.dtype,device=device).reshape(129,3,3)%17-8)/19
 expected=ordered_reference(budget,args,cot);calls=[];original=type(budget)._prepare_curve
 def observe(self,roughness,**kw):calls.append(len(roughness));return original(self,roughness,**kw)
 monkeypatch.setattr(type(budget),'_prepare_curve',observe)
 def evaluate():
  y=d._principled_blocks(n,v,directions,dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=device=='cuda'))
  return (y,*torch.autograd.grad(y,args,cot))
 actual=evaluate()
 for x,z in zip(actual,expected):assert torch.equal(raw(x),raw(z))
 assert calls==[1,1],calls
 with torch.no_grad():r.add_(.013)
 calls.clear();fresh=evaluate();assert calls==[1,1]
 assert not torch.equal(raw(fresh[0]),raw(actual[0]))
 monkeypatch.setattr(type(budget),'_prepare_curve',original)
 expected=ordered_reference(budget,args,cot)
 for x,z in zip(fresh,expected):assert torch.equal(raw(x),raw(z))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_point_roughness_field_keeps_independent_owners(device):
 from gffx.torch import material,_deferred as d
 budget,base=case(9,3,device);n,v,l,a,_,f=(x.detach() for x in base)
 directions=l[None].expand(9,-1,-1).clone();r=torch.linspace(.43,.79,9,dtype=n.dtype,device=device,requires_grad=True)
 cot=(torch.arange(81,dtype=n.dtype,device=device).reshape(9,3,3)%17-8)/19
 params=dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=device=='cuda')
 expected=torch.cat([material.principled(n[i:i+4],v[i:i+4],directions[i:i+4],**dict(params,roughness=r[i:i+4])) for i in range(0,9,4)])
 eg,=torch.autograd.grad(expected,r,cot)
 actual=d._principled_blocks(n,v,directions,params);ag,=torch.autograd.grad(actual,r,cot)
 assert torch.equal(raw(actual),raw(expected)) and torch.equal(raw(ag),raw(eg))
 assert bool((ag!=0).all())
