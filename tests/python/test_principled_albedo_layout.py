"""Constant albedo backward preserves full native checks and group reduction order."""
import os,torch,pytest
from gffx.torch import material
from gffx.torch import _principled_groups as groups


def fixture(points=5,lights=1):
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');t=lambda x:torch.tensor(x,dtype=torch.float64,device=dev)
    n=t([[.02,.03,1.]]).expand(points,3).clone();v=t([[.1,-.1,2.]]).expand(points,3).clone();l=t([[[.1,.2,1.]]]).expand(points,lights,3).clone()
    if points>1:l[::3,:,2]=-1
    params=dict(albedo=t([.2,.4,.6]).requires_grad_(),roughness=t(.827),f0=t([.04,.05,.08]),budget=material.PrincipledBudget.bundled('ggx-smith-loss-'+dev+'-f64-v1',device=dev,dtype=torch.float64))
    return n,v,l,params


def bits(t):return t.detach().cpu().contiguous().numpy().tobytes()


def test_constant_albedo_backward_avoids_temporary_group_graphs(monkeypatch):
    n,v,l,p=fixture(17);calls=[];old=groups._material
    def traced(*args,**kw):calls.append(1);return old(*args,**kw)
    monkeypatch.setattr(groups,'_material',traced)
    y=groups.render(n,v,l,p,4);torch.autograd.grad(y.sum(),p['albedo'])
    assert calls==[]


@pytest.mark.parametrize('points,lights',[(1,1),(3,2),(4,1),(5,3),(127,1),(129,2),(800,1),(128,8)])
@pytest.mark.parametrize('edge',[False,True])
def test_constant_albedo_backward_matches_retained_graphs(points,lights,edge,monkeypatch):
    n,v,l,p=fixture(points,lights);a=p['albedo']
    cot=((torch.arange(points*lights*3,dtype=torch.float64)%13-6)/17).reshape(points,lights,3).to(a.device)
    if edge:cot[:]=a.new_tensor([-.0,5e-324,-5e-324])
    y=groups.render(n,v,l,p,4);actual,=torch.autograd.grad(y,a,cot)
    monkeypatch.setattr(groups,'_albedo_cohort_partials',lambda *args,**kw:None)
    z=groups.render(n,v,l,p,4);expected,=torch.autograd.grad(z,a,cot)
    assert bits(y)==bits(z) and bits(actual)==bits(expected)


def test_other_derivative_targets_retain_original_graph_path(monkeypatch):
    n,v,l,p=fixture();n.requires_grad_();calls=[];old=groups._material
    def traced(*args,**kw):calls.append(1);return old(*args,**kw)
    monkeypatch.setattr(groups,'_material',traced);y=groups.render(n,v,l,p,4);g=torch.autograd.grad(y.sum(),(n,p['albedo']))
    assert calls and all(torch.isfinite(x).all() for x in g)


@pytest.mark.parametrize('operation',['shading_principled_shared_groups','shading_principled_groups_backward'])
def test_constant_albedo_backward_preserves_native_derivative_failure(operation,monkeypatch):
    n,v,l,p=fixture();y=groups.render(n,v,l,p,4)
    def fail(*args,**kw):raise RuntimeError('injected full native VJP failure')
    monkeypatch.setattr(torch.ops.gffx,operation,fail)
    with pytest.raises((RuntimeError,ValueError),match='full native VJP failure'):torch.autograd.grad(y.sum(),p['albedo'])


def test_constant_albedo_layout_scalar_f0_and_full_jacobian(monkeypatch):
    n,v,l,p=fixture(5,2);p['f0']=p['f0'][0].clone();seen=[];op=torch.ops.gffx.shading_principled_shared_groups
    def checked(*args):
        result=op(*args)
        if args[-1]:seen.append(result[1].shape[-1])
        return result
    monkeypatch.setattr(torch.ops.gffx,'shading_principled_shared_groups',checked)
    actual,=torch.autograd.grad(groups.render(n,v,l,p,4).sum(),p['albedo'])
    assert seen and set(seen)=={16}
    monkeypatch.setattr(groups,'_albedo_cohort_partials',lambda *args,**kw:None)
    expected,=torch.autograd.grad(groups.render(n,v,l,p,4).sum(),p['albedo'])
    assert bits(actual)==bits(expected)
