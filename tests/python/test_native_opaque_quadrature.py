"""Installed native visible-polygon quadrature and its first VJP."""
import torch,pytest
from fractions import Fraction as Q
from tools.opaque_pixel_oracle import integrate_affine

def api():
    from gffx.torch.render import opaque_quadrature
    return opaque_quadrature

def sample(v,f,cuts=None,h=1,w=1,origin=(0.,0.)):
    if cuts is None:cuts=v.new_empty((len(f),3,0))
    return api()(v,f,cuts,h,w,origin=origin)

def colour(v,f,c,cut=None,h=1,w=1):
    ids,q=sample(v,f,cut,h,w);valid=ids>=0;owner=ids.clamp_min(0).long()
    beta=q[...,:3];weight=q[...,3];rgb=(beta[...,None]*c[owner]).sum(-2)
    return torch.cat([(rgb*weight[...,None]).sum(-2),weight.sum(-1,keepdim=True)],-1)

@pytest.mark.parametrize('depth',[False,True])
def test_native_quadrature_matches_exact_affine_visibility_and_geometry_gradient(depth):
    xy=[(-2.,-2.),(6.,-2.),(-2.,6.)];rows=[(x,y,x-.25 if depth else 0.) for x,y in xy]
    if depth:rows += [(x,y,0.) for x,y in xy]
    v=torch.tensor(rows,dtype=torch.float64,requires_grad=True);f=torch.tensor([(0,1,2),(3,4,5)] if depth else [(0,1,2)],dtype=torch.int32)
    c=torch.tensor([[[.2,.4,.6],[.8,.1,.3],[.3,.9,.4]],[[.7,.2,.1],[.1,.8,.3],[.3,.1,.8]]][:len(f)],dtype=torch.float64,requires_grad=True)
    out=colour(v,f,c)[0];target=integrate_affine(rows,f.tolist(),c.tolist(),height=1,width=1)
    assert torch.allclose(out,torch.tensor([*[float(x) for x in target['colour'][0][0]],float(target['mask'][0][0])],dtype=v.dtype),atol=1e-12,rtol=0)
    motion=torch.zeros_like(v);motion[:3,2]=-1 if depth else 0;motion[:3,0]=0 if depth else 1
    gv,gc=torch.autograd.grad((out*out.new_tensor([.3,-.2,.7,.4])).sum(),(v,c))
    for epsilon in (Q(1,1024),Q(1,4096)):
        a=[[Q(x)+epsilon*Q(d) for x,d in zip(p,m)] for p,m in zip(rows,motion.tolist())];b=[[Q(x)-epsilon*Q(d) for x,d in zip(p,m)] for p,m in zip(rows,motion.tolist())]
        ia=integrate_affine(a,f.tolist(),c.tolist(),height=1,width=1);ib=integrate_affine(b,f.tolist(),c.tolist(),height=1,width=1)
        da=[(x-y)/(2*epsilon) for x,y in zip(ia['colour'][0][0],ib['colour'][0][0])]+[(ia['mask'][0][0]-ib['mask'][0][0])/(2*epsilon)]
        expected=sum(float(x)*y for x,y in zip(da,[.3,-.2,.7,.4]));assert abs((gv*motion).sum().item()-expected)<1e-12


def test_native_quadrature_moving_material_cut_preserves_boundary_flux():
    v=torch.tensor([[-2.,-2.,0.],[6.,-2.,0.],[-2.,6.,0.]],dtype=torch.float64);f=torch.tensor([[0,1,2]],dtype=torch.int32);theta=torch.tensor(.25,dtype=v.dtype,requires_grad=True)
    cut=(v[f.long(),0]-theta)[...,None].contiguous();ids,q=sample(v,f,cut);x=(q[...,:3]*v[f.long(),0][ids.clamp_min(0).long()]).sum(-1)
    rgb=torch.where(x<theta,torch.ones_like(x),torch.full_like(x,3.));image=(rgb*q[...,3]).sum();derivative=torch.autograd.grad(image,theta)[0]
    assert abs(image.item()-2.5)<1e-12 and abs(derivative.item()+2)<1e-12


def test_native_quadrature_all_geometry_and_cut_coordinates_gradcheck():
    v=torch.tensor([[.15,.1,0.],[.8,.2,0.],[.2,.85,0.]],dtype=torch.float64,requires_grad=True);f=torch.tensor([[0,1,2]],dtype=torch.int32);cut=torch.tensor([[[-.2],[.5],[-.2]]],dtype=v.dtype,requires_grad=True)
    def fn(v,cut):
        ids,q=sample(v,f,cut);beta=q[...,:3];return ((beta[...,0]**2+beta[...,1]**3+beta[...,2])*q[...,3]).sum()
    assert torch.autograd.gradcheck(fn,(v,cut),eps=1e-6,atol=1e-6,rtol=1e-4)


def test_native_quadrature_empty_padding_capacity_and_budget_refusals():
    v=torch.empty((0,3),dtype=torch.float64);f=torch.empty((0,3),dtype=torch.int32);ids,q=sample(v,f)
    assert ids.shape==(1,800) and q.shape==(1,800,4);assert torch.all(ids==-1) and torch.count_nonzero(q)==0
    with pytest.raises(ValueError):api()(v,f,torch.empty((0,3,17),dtype=v.dtype),1,1)
    with pytest.raises(ValueError):api()(v,f,torch.empty((0,3,0),dtype=v.dtype),1080,1920)
    vv=torch.tensor([[-1.,-1.,0.],[3.,-1.,0.],[-1.,3.,0.]],dtype=torch.float64);ff=torch.tensor([[0,1,2]]*33,dtype=torch.int32)
    with pytest.raises(OverflowError,match='capacity'):sample(vv,ff)


@pytest.mark.parametrize('dtype,tolerance',[(torch.float32,1e-6),(torch.float64,1e-12)])
def test_native_quadrature_preserves_subpixel_area_at_large_origins(dtype,tolerance):
    f=torch.tensor([[0,1,2]],dtype=torch.int32)
    for origin in (0.,1024.,65536.):
        v=torch.tensor([[origin+.25,origin+.25,0.],[origin+.5,origin+.25,0.],[origin+.25,origin+.5,0.]],dtype=dtype,requires_grad=True)
        ids,q=sample(v,f,origin=(origin,origin));total=q[...,3].sum();g,=torch.autograd.grad(total,v)
        assert abs(total.item()-.03125)<=tolerance;torch.testing.assert_close(g,torch.tensor([[-.125,-.125,0.],[.125,0.,0.],[0.,.125,0.]],dtype=dtype),atol=tolerance,rtol=0)

def test_native_quadrature_refuses_nonfinite_second_derivatives():
    v=torch.tensor([[.1,.1,0.],[.8,.1,0.],[.1,.8,0.]],dtype=torch.float64,requires_grad=True);f=torch.tensor([[0,1,2]],dtype=torch.int32)
    ids,q=sample(v,f)
    with pytest.raises(RuntimeError,match='first derivatives'):torch.autograd.grad(q[...,3].sum(),v,create_graph=True)
    with pytest.raises(ValueError,match='finite'):sample(v.detach()*float('nan'),f)
