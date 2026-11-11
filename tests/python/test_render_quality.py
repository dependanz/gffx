"""Analytic sidedness, shadow-plane and rendered-gradient regressions."""
import math
from dataclasses import replace
import pytest
import torch
import gffx as g
from gffx.torch import material, light
from gffx.torch import _area, _deferred

@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    return request.param

def test_geometric_sidedness_does_not_cut_a_smooth_front_face(device):
    dtype=torch.float64
    model=g.Scene([g.Material.diffuse('m',(.4,.5,.6)),
        g.Mesh.quad('q',u=(2,0,0),v=(0,2,0),material='m'),
        g.Light('l',light.point,dict(positions=torch.tensor([[2.,0.,1.]]),intensity=torch.ones(1,3)))])
    prims=list(model.prims);q=prims[1]
    normal=torch.tensor([1.,0.,-.1]);normal/=normal.norm()
    prims[1]=replace(q,attributes={'normals':g.scene.Attribute(normal.expand(4,3).clone(),'vertex')})
    model=g.Scene(prims).to(device,dtype=dtype)
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(3,3),dtype=dtype,device=device)
    image=g.render(model,camera)['colour'][0,1,1]
    n=normal.to(device,dtype=dtype);direction=n.new_tensor([2.,0.,1.]);direction/=direction.norm()
    expected=n.new_tensor([.4,.5,.6])/math.pi*(n*direction).sum()/5
    torch.testing.assert_close(image,expected,atol=1e-8,rtol=1e-7)
    assert (image>0).all()
    # Standalone material still interprets its sole normal as the surface side.
    standalone=material.classic(n[None],n.new_tensor([[0.,0.,1.]]),direction[None],albedo=n.new_tensor([.4,.5,.6]),specular=n.new_tensor(0.),exponent=n.new_tensor(1.))
    assert torch.count_nonzero(standalone)==0

def test_unoccluded_shadow_receiver_uses_triangle_plane_not_shading_normal(device):
    q=g.Mesh.quad('q',u=(3,0,0),v=(0,3,0),material='m')
    q=replace(q,attributes={'normals':g.scene.Attribute(torch.tensor([[.6,0,1.]]).expand(4,3).clone(),'vertex')})
    model=g.Scene([g.Material.diffuse('m',(.4,.5,.6)),q,
        g.Light.rectangle('l',center=(.4,.4,2),u=(.5,0,0),v=(0,-.5,0),radiance=5,samples=2,visible=False)]).to(device,dtype=torch.float64)
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(17,17),dtype=torch.float64,device=device)
    plain=g.render(model,camera)['colour'];shadow=g.render(model,camera,shadows=True,shadow_map_size=128)['colour']
    # The only surface is a plane: every ray from its interior toward the light
    # is unobstructed. With receiver-plane correction, all PCF samples are lit.
    torch.testing.assert_close(shadow,plain,rtol=1e-12,atol=1e-12)

def test_shadow_filter_is_continuous_and_retains_analytic_occluder(device):
    model=g.Scene([g.Material.diffuse('m',.5),
        g.Mesh.quad('floor',u=(3,0,0),v=(0,3,0),material='m'),
        g.Mesh.quad('block',center=(0,0,1),u=(.4,0,0),v=(0,.4,0),material='m')]).to(device,dtype=torch.float64)
    packed=model.flatten();x=torch.linspace(.2,.6,401,device=device,dtype=torch.float64)
    points=torch.stack((x,torch.zeros_like(x),torch.zeros_like(x)),-1);normals=points.new_tensor([0.,0.,1.]).expand_as(points)
    vis=_area.visibility(packed,0,points,normals,points.new_tensor([0.,0.,2.]),256,{},'probe')
    # Pinhole projection doubles the occluder's half-width: edge x=.4.
    assert vis[0]==0 and vis[-1]==1
    assert (vis[1:]-vis[:-1]>=-1e-12).all()
    assert ((vis>0)&(vis<1)).sum()>30
    assert (vis[1:]-vis[:-1]).abs().max()<.06
    assert abs(float(x[(vis-.5).abs().argmin()])-.4)<.035

def test_glossy_render_gradients_match_central_differences(device):
    a=torch.tensor([.3,.45,.6],device=device,dtype=torch.float64,requires_grad=True)
    r=torch.tensor(.4,device=device,dtype=torch.float64,requires_grad=True)
    intensity=torch.tensor(8.,device=device,dtype=torch.float64,requires_grad=True)
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(12,12),device=device,dtype=torch.float64)
    def render(aa,rr,ii):
        model=g.Scene([g.Material.glossy('m',aa,roughness=rr,specular=.35),
            g.Mesh.sphere('s',radius=.7,material='m'),
            g.Light.rectangle('l',center=(.8,.8,2),u=(.8,0,0),v=(0,-.8,0),radiance=ii,samples=2,visible=False)]).to(device,dtype=torch.float64)
        rgb=g.render(model,camera)['colour']
        weights=torch.linspace(.2,1.,rgb.numel(),device=device,dtype=rgb.dtype).reshape_as(rgb)
        return (rgb*weights).mean()
    grads=torch.autograd.grad(render(a,r,intensity),(a,r,intensity))
    h=1e-5
    for value,gradient in zip((a,r,intensity),grads):
        assert torch.isfinite(gradient).all() and (gradient.abs()>1e-8).all()
        for j in range(value.numel()):
            plus=[v.detach().clone() for v in (a,r,intensity)];minus=[v.detach().clone() for v in (a,r,intensity)]
            k=0 if value is a else 1 if value is r else 2
            plus[k].reshape(-1)[j]+=h;minus[k].reshape(-1)[j]-=h
            fd=(render(*plus)-render(*minus))/(2*h)
            torch.testing.assert_close(gradient.reshape(-1)[j],fd,rtol=2e-5,atol=2e-8)

@pytest.mark.parametrize('roughness',[0.,-.1,1.1,float('nan')])
def test_glossy_rejects_invalid_roughness(roughness):
    with pytest.raises(ValueError):g.Material.glossy('m',.5,roughness=roughness)


def test_rendered_geometry_and_camera_gradients_inside_fixed_visibility(device):
    radius=torch.tensor(.7,device=device,dtype=torch.float64,requires_grad=True)
    eye_x=torch.tensor(.07,device=device,dtype=torch.float64,requires_grad=True)
    def objective(r,x):
        eye=torch.stack((x,x.new_tensor(.13),x.new_tensor(3.)))
        camera=g.Camera.look_at(eye,(0,0,0),size=(16,16))
        model=g.Scene([g.Material.glossy('m',(.3,.4,.5),roughness=.4),
            g.Mesh.sphere('s',radius=r,material='m'),
            g.Light.rectangle('l',center=(.8,.8,2),u=(.8,0,0),v=(0,-.8,0),radiance=8,samples=2,visible=False)]).to(device,dtype=torch.float64)
        return g.render(model,camera)['colour'][0,6:10,6:10].mean()
    grads=torch.autograd.grad(objective(radius,eye_x),(radius,eye_x))
    h=1e-5
    for j,gradient in enumerate(grads):
        plus=[radius.detach().clone(),eye_x.detach().clone()];minus=[v.clone() for v in plus]
        plus[j]+=h;minus[j]-=h
        fd=(objective(*plus)-objective(*minus))/(2*h)
        assert abs(float(gradient))>1e-8
        torch.testing.assert_close(gradient,fd,rtol=2e-4,atol=1e-7)


def test_bent_glossy_normal_preserves_diffuse_below_highlight_horizon(device):
    n=torch.tensor([[1.,0.,-4.]],device=device,dtype=torch.float64)
    v=n.new_tensor([[0.,0.,1.]]);l=n.new_tensor([[1.,0.,.1]])
    a=n.new_tensor([.2,.3,.4])
    value=material.classic(n,v,l,albedo=a,specular=n.new_tensor(.4),exponent=n.new_tensor(8.),_geometric_normals=v)
    torch.testing.assert_close(value[0,0],a/math.pi,rtol=1e-12,atol=1e-12)
    # Light below the actual surface must not leak through a bent normal.
    below=n.new_tensor([[1.,0.,-.1]])
    value=material.classic(n,v,below,albedo=a,specular=n.new_tensor(.4),exponent=n.new_tensor(8.),_geometric_normals=v)
    assert torch.count_nonzero(value)==0

@pytest.mark.parametrize('name',['cornell_box','cornell_glossy','fit_glossy_material'])
def test_installed_quality_examples_match_public_source(name):
    from pathlib import Path
    import importlib.util
    public=Path(__file__).resolve().parents[2]/'examples'/(name+'.py')
    installed=Path(importlib.util.find_spec('gffx.examples.'+name).origin)
    assert installed.read_bytes()==public.read_bytes()
