"""Small independent deferred controls; no beauty image is numerical acceptance."""
import importlib
import pytest
import torch
from gffx.torch import scene, material, light
from gffx.torch.camera import Camera


def api():
    return importlib.import_module('gffx.torch._deferred')


def test_deferred_facade_namespace_is_lazy():
    import subprocess
    import sys
    script='''import importlib,sys
import gffx
surface=gffx.render
assert callable(surface)
assert 'torch' not in sys.modules
assert getattr(surface,'__path__',None) is None
assert getattr(surface,'not_a_public_name',None) is None
assert 'torch' not in sys.modules
assert importlib.import_module('gffx.render') is surface
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr


def fixture(*, dtype=torch.float64, device='cpu', z=-2., colour=None):
    def t(x): return torch.tensor(x,dtype=dtype,device=device)
    vertices=t([[-1.,-1.,z],[1.,-1.,z],[1.,1.,z],[-1.,1.,z]])
    faces=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32,device=device)
    albedo=t([.2,.4,.6]) if colour is None else colour
    mat=scene.Material('mat',material.classic,dict(albedo=albedo,specular=t(0.),exponent=t(4.)))
    lamp=scene.Light('key',light.directional,dict(directions=t([[0.,0.,1.]]),irradiance=t([[1.,1.,1.]])))
    camera=Camera.from_intrinsics(t(4.),t(4.),t(2.),t(2.),4,4,.1,10.)
    model=scene.Scene([scene.Mesh('quad',vertices,faces,material='mat'),mat,lamp])
    return model,camera,vertices,faces,albedo


def test_deferred_public_facades_delegate_without_mutating_inputs():
    import inspect
    import gffx
    d=api();model,camera,vertices,faces,albedo=fixture()
    inputs=(vertices,faces,albedo,camera.fx,camera.pose)
    before=[value.clone() for value in inputs]
    names=('colour','mask','depth','normal','world_position','face_index')
    direct=d.render(model,camera,outputs=names)
    base=gffx.render(model,camera,outputs=names)
    bound=gffx.torch.render(model,camera,outputs=names)
    assert inspect.ismodule(gffx.render) and inspect.ismodule(gffx.torch.render)
    assert gffx.render.AttributeSource is gffx.torch.render.AttributeSource
    assert gffx.render.mesh is gffx.torch.render.mesh
    assert {'mesh','visibility','texture','AttributeSource'}<=set(dir(gffx.render))
    for key in names:assert torch.equal(base[key],direct[key]) and torch.equal(bound[key],direct[key])
    for original,current in zip(before,inputs):assert torch.equal(original,current)


def test_deferred_contract_analytic_quad():
    d=api();model,camera,v,f,a=fixture()
    result=d.render(model,camera,outputs=('colour','mask','normal','world_position','depth','face_index'))
    assert set(result)=={'colour','mask','normal','world_position','depth','face_index'}
    assert torch.equal(result['mask'],torch.ones((1,4,4),dtype=v.dtype))
    torch.testing.assert_close(result['colour'],(a/torch.pi).expand(1,4,4,3),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(result['normal'],v.new_tensor([0.,0.,1.]).expand(1,4,4,3))
    expected=(torch.arange(4,dtype=v.dtype)+.5-2.)/2.
    yy,xx=torch.meshgrid(-expected,expected,indexing='ij')
    torch.testing.assert_close(result['world_position'][0],torch.stack((xx,yy,torch.full_like(xx,-2.)),-1),atol=1e-14,rtol=1e-14)
    assert torch.isfinite(result['depth']).all()


def test_deferred_empty_background_and_options():
    d=api();_,c,_,_,_=fixture()
    empty=scene.Scene([])
    result=d.render(empty,c,background=c.fx.new_tensor([.1,.2,.3]),outputs=('colour','mask','depth','visibility'))
    assert result['visibility'].shape==(1,0)
    assert torch.equal(result['colour'],c.fx.new_tensor([.1,.2,.3]).expand(1,4,4,3))
    assert (result['mask']==0).all() and torch.isinf(result['depth']).all()
    for option in ({'outputs':('oops',)},{'soft':float('inf')},{'supersample':True},{'workspace_limit_bytes':0}):
        with pytest.raises((TypeError,ValueError)):d.render(empty,c,**option)


def test_deferred_triangle_perspective_plane():
    d=api();_,c,_,_,_=fixture()
    # Plane z=-2-x/4: independently unproject each covered pixel ray onto it.
    v=c.fx.new_tensor([[-1.,-1.,-1.75],[1.,-1.,-2.25],[0.,1.,-2.]])
    f=torch.tensor([[0,1,2]],dtype=torch.int32)
    s=scene.Scene([scene.Mesh('tilted',v,f)])
    out=d.render(s,c,outputs=('world_position','mask','barycentric'))
    y,x=torch.meshgrid(torch.arange(4,dtype=v.dtype)+.5,torch.arange(4,dtype=v.dtype)+.5,indexing='ij')
    rx=(x-2)/4;ry=-(y-2)/4;t=2/(1-rx/4)
    expected=torch.stack((rx*t,ry*t,-t),-1)
    covered=out['mask'][0].bool()
    torch.testing.assert_close(out['world_position'][0][covered],expected[covered],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(out['barycentric'].sum(-1)[out['mask'].bool()],torch.ones_like(out['mask'][out['mask'].bool()]))


def test_deferred_albedo_gradient_and_mesh_convenience():
    d=api();a=torch.tensor([.2,.4,.6],dtype=torch.float64,requires_grad=True)
    model,c,v,f,_=fixture(colour=a)
    result=d.render(model,c)['colour']
    g,=torch.autograd.grad(result.sum(),a)
    torch.testing.assert_close(g,torch.full_like(a,16/torch.pi),rtol=1e-13,atol=1e-13)
    mat=next(p for p in model.prims if type(p)==scene.Material)
    lamps=[p for p in model.prims if type(p)==scene.Light]
    other=d.mesh(v,f,mat,lamps,c)['colour']
    assert torch.equal(result,other)


def test_deferred_hard_mask_has_zero_geometric_derivative():
    d=api();model,c,v,f,_=fixture();v.requires_grad_()
    result=d.render(model,c,outputs=('mask',))['mask']
    g,=torch.autograd.grad(result.sum(),v)
    assert torch.equal(g,torch.zeros_like(v))


def test_deferred_soft_mask_has_nonzero_derivative():
    d=api();_,c,v,f,_=fixture();v=v*v.new_tensor([.7,.7,1.]);v.requires_grad_()
    s=scene.Scene([scene.Mesh('quad',v,f)])
    result=d.render(s,c,soft=.6,outputs=('mask',))['mask']
    g,=torch.autograd.grad(result.sum(),v)
    assert torch.isfinite(g).all() and bool((g!=0).any())


def test_deferred_interpolate_rejects_unstructured_landmarks():
    from gffx.torch import render
    with pytest.raises(ValueError):
        render.interpolate(torch.tensor([0],dtype=torch.int32),torch.tensor([[1.,0.,0.]],dtype=torch.float64),torch.zeros((1,3,3),dtype=torch.float64))


def test_deferred_budget_preflight_and_duplicate_outputs():
    d=api();s,c,*_=fixture()
    with pytest.raises(ValueError,match='budget'):d.render(s,c,workspace_limit_bytes=1)
    with pytest.raises(ValueError):d.render(s,c,outputs=('mask','mask'))


def test_deferred_supersampling_is_manual_box_and_preserves_ids():
    d=api();s,c,*_=fixture()
    from gffx.torch.image import reconstruct
    doubled=Camera.from_intrinsics(c.fx*2,c.fy*2,c.cx*2,c.cy*2,8,8,c.near,c.far,pose=c.pose)
    fine=d.render(s,doubled,outputs=('colour','mask'))
    actual=d.render(s,c,supersample=2,outputs=('colour','mask','face_index','depth'))
    expected=reconstruct(torch.cat((fine['colour'],fine['mask'][...,None]),-1),2)
    assert torch.equal(actual['colour'],expected[...,:3])
    assert torch.equal(actual['mask'],expected[...,3])
    base=d.render(s,c,outputs=('face_index','depth'))
    assert torch.equal(actual['face_index'],base['face_index'])
    assert torch.equal(actual['depth'],base['depth'])


def test_deferred_attribute_texture_source_and_map_gradient():
    d=api();s,c,v,f,_=fixture()
    from gffx.torch import render
    texture=torch.tensor([[[.2,.3,.4],[.4,.5,.6]],[[.3,.4,.5],[.5,.6,.7]]],dtype=v.dtype,requires_grad=True)
    pyramid,offsets=render.texture_pyramid(texture,levels=1)
    uv=v.new_tensor([[0.,1.],[1.,1.],[1.,0.],[0.,0.]])
    lamp=[p for p in s.prims if type(p)==scene.Light][0]
    mat=scene.Material('mapped',material.classic,dict(albedo=d.TextureSource(pyramid,offsets,2,2,3,
        wrap_u=render.WRAP_CLAMP,wrap_v=render.WRAP_CLAMP),specular=v.new_tensor(0.),exponent=v.new_tensor(4.)))
    model=scene.Scene([scene.Mesh('quad',v,f,attributes={'uv':scene.Attribute(uv,'vertex')},material='mapped'),mat,lamp])
    actual=d.render(model,c,outputs=('colour','uv','mask'))
    sampled=render.texture(pyramid,offsets,2,2,actual['uv'].reshape(-1,2).contiguous(),channel_count=3,
        wrap_u=render.WRAP_CLAMP,wrap_v=render.WRAP_CLAMP).reshape(1,4,4,3)
    torch.testing.assert_close(actual['colour'],sampled/torch.pi,atol=1e-13,rtol=1e-13)
    gradient,=torch.autograd.grad(actual['colour'].sum(),texture)
    assert torch.isfinite(gradient).all() and (gradient>0).all()


def test_deferred_rig_matches_separate_views_and_shared_gradient():
    d=api();s,c,v,_,_=fixture();v.requires_grad_()
    fx=c.fx.clone().requires_grad_();cx=c.cx.clone().requires_grad_()
    other=Camera.from_intrinsics(fx,c.fy,cx,c.cy,4,4,c.near,c.far,pose=c.pose)
    rig=Camera.stack([c,other])
    actual=d.render(s,rig,outputs=('world_position','face_index'))
    one=d.render(s,c,outputs=('world_position','face_index'))
    two=d.render(s,other,outputs=('world_position','face_index'))
    assert torch.equal(actual['world_position'][0],one['world_position'][0])
    assert torch.equal(actual['world_position'][1],two['world_position'][0])
    assert torch.equal(actual['face_index'][1],two['face_index'][0]+2)
    weights=torch.arange(actual['world_position'].numel(),dtype=v.dtype).reshape_as(actual['world_position'])/100
    gv,gfx,gcx=torch.autograd.grad((actual['world_position']*weights).sum(),(v,fx,cx))
    assert all(torch.isfinite(g).all() and bool((g!=0).any()) for g in (gv,gfx,gcx))


def test_deferred_visibility_pixel_centre_and_occlusion():
    d=api();s,c,v,f,_=fixture()
    # Unreferenced samples share geometry storage but do not alter the front quad.
    vertices=torch.cat((v,v.new_tensor([[.25,-.25,-2.],[.375,-.375,-3.],[0.,0.,1.],[4.,0.,-2.]])))
    model=scene.Scene([scene.Mesh('quad',vertices,f)])
    visible=d.visibility(model,c)
    assert visible.shape==(1,8)
    assert visible[0,4:].tolist()==[True,False,False,False]


def test_deferred_soft_boundary_extension_keeps_albedo_in_domain():
    d=api();_,c,v,f,_=fixture();v=v*v.new_tensor([.55,.55,1.])
    mat=scene.Material('colour',material.classic,dict(albedo=d.AttributeSource('colour'),
        specular=v.new_tensor(0.),exponent=v.new_tensor(4.)))
    lamp=scene.Light('key',light.directional,dict(directions=v.new_tensor([[0.,0.,1.]]),irradiance=v.new_tensor([[1.,1.,1.]])))
    source=v.new_tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,1.],[0.,1.,0.]])
    model=scene.Scene([scene.Mesh('q',v,f,material='colour',attributes={'colour':scene.Attribute(source,'vertex')}),mat,lamp])
    result=d.render(model,c,soft=1.)
    assert torch.isfinite(result['colour']).all() and (result['colour']>=0).all()
    assert (result['colour']<=result['mask'][...,None]/torch.pi+1e-14).all()
