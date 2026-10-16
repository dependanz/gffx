"""Keep complete output ownership while fitting native scratch to remaining space."""
import torch
import pytest
from test_deferred_native_storage import scale
from gffx.torch import render


def test_cpu64_backward_selects_a_fitting_cohort_before_allocation():
 from gffx.torch._deferred import _plan_workload
 model,camera,albedo=scale('cpu',64)
 plan=_plan_workload(model,camera,1,64*1024**2,('colour','mask'))
 assert plan['backward'] and plan['estimated_bytes']<=64*1024**2
 assert 4<=plan['principled_group_point_limit']<128
 y=render(model,camera)['colour'];g,=torch.autograd.grad(y.mean(),albedo)
 assert torch.isfinite(y).all() and torch.isfinite(g).all() and bool((g!=0).all())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fitting_cohort_size_preserves_each_shared_gradient(device):
 from test_deferred_groups import case,ordered_reference
 from test_principled_native_operation import raw
 from gffx.torch._deferred import _principled_blocks
 budget,base=case(37,3,device);n,v,l,a,r,f=base
 directions=l.detach()[None].expand(37,-1,-1).clone().requires_grad_();args=(n,v,directions,a,r,f)
 cot=(torch.arange(333,dtype=n.dtype,device=device).reshape(37,3,3)%17-8)/19
 expected=ordered_reference(budget,args,cot)
 for limit in (4,12,28,128):
  y=_principled_blocks(n,v,directions,dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=device=='cuda'),group_point_limit=limit)
  actual=(y,*torch.autograd.grad(y,args,cot))
  for x,z in zip(actual,expected):assert torch.equal(raw(x),raw(z))
