"""Pack already validated constant forward cohorts without rebuilding four-point lists."""
import os
import torch
from test_deferred_groups import case,ordered_reference


def test_constant_forward_avoids_group_slicing_and_preserves_ordered_vjp(monkeypatch):
    from gffx.torch import _principled_groups as groups,_deferred as deferred
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');budget,base=case(65,3,dev)
    n,v,l,a,r,f=base;directions=l.detach()[None].expand(65,-1,-1).clone().requires_grad_();args=(n,v,directions,a,r,f)
    calls=[];old=groups._groups
    def observed(*a,**kw):calls.append(True);return old(*a,**kw)
    monkeypatch.setattr(groups,'_groups',observed)
    y=deferred._principled_blocks(n,v,directions,dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=dev=='cuda'))
    assert not calls,'constant forward rebuilt per-four-point groups'
    cot=(torch.arange(y.numel(),device=dev,dtype=torch.float64).reshape(y.shape)%17-8)/19
    result=(y,*torch.autograd.grad(y,args,cot));assert calls,'canonical backward group replay was bypassed'
    expected=ordered_reference(budget,args,cot)
    for actual,reference in zip(result,expected):
        assert actual.detach().cpu().contiguous().numpy().tobytes()==reference.detach().cpu().contiguous().numpy().tobytes()


def test_varying_fields_keep_general_group_layout(monkeypatch):
    from gffx.torch import _principled_groups as groups,_deferred as deferred
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');budget,base=case(9,2,dev);n,v,l,a,r,f=base
    a=a.detach().expand(9,3).clone().requires_grad_();directions=l.detach()[None].expand(9,-1,-1).clone().requires_grad_();args=(n,v,directions,a,r,f)
    calls=[];old=groups._groups
    def observed(*a,**kw):calls.append(True);return old(*a,**kw)
    monkeypatch.setattr(groups,'_groups',observed)
    y=deferred._principled_blocks(n,v,directions,dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=dev=='cuda'))
    assert calls
    expected=torch.cat([deferred._principled_recompute(n[i:i+4],v[i:i+4],directions[i:i+4],dict(albedo=a[i:i+4],roughness=r,f0=f,budget=budget,allow_experimental_budget=dev=='cuda')) for i in range(0,9,4)])
    assert torch.equal(y,expected)
