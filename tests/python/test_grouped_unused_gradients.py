"""Unused autograd edges must not allocate discarded gradient copies."""
import pytest,torch
from test_deferred_groups import case
from test_principled_native_operation import raw

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_grouped_albedo_backward_avoids_unrequested_gradient_copies(device,monkeypatch):
 from gffx.torch._principled_groups import _material
 from gffx.torch import material
 budget,args=case(125,1,device);n,v,l,a,r,f=[x.detach() for x in args]
 direction=l[None].expand(len(n),-1,-1)
 keys=('albedo','f0','roughness');static=dict(budget=budget,allow_experimental_budget=device=='cuda')
 rows=[];leaves=[];expected=[];cots=[]
 for start in range(0,len(n),4):
  leaf=a.clone().requires_grad_();leaves.append(leaf)
  row=(n[start:start+4],v[start:start+4],direction[start:start+4],leaf,f,r);rows.append(row)
  y=material.principled(*row[:3],albedo=leaf,f0=f,roughness=r,**static)
  cot=(torch.arange(y.numel(),device=device,dtype=torch.float64).reshape(y.shape)%7-3)/11;cots.append(cot)
  expected.append(torch.autograd.grad(y,leaf,cot)[0])
 outputs=_material(keys,static,rows,{})
 calls=[];original=torch.Tensor.contiguous
 def observe(self,*a,**kw):
  result=original(self,*a,**kw)
  if result.untyped_storage().data_ptr()!=self.untyped_storage().data_ptr():calls.append(tuple(self.shape))
  return result
 monkeypatch.setattr(torch.Tensor,'contiguous',observe)
 actual=torch.autograd.grad(outputs,leaves,cots);copy_count=len(calls)
 for got,want in zip(actual,expected):assert torch.equal(raw(got),raw(want))
 assert copy_count<=len(rows),(copy_count,len(rows))
