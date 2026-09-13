"""Independent central differences away from raster and sampler discontinuities."""
import torch
import pytest
from gffx.torch import scene,material,light,render
from gffx.torch.camera import Camera
from test_deferred import api,fixture


def camera_one(like):
    return Camera.from_intrinsics(like.new_tensor(1.),like.new_tensor(1.),like.new_tensor(.45),
        like.new_tensor(.55),1,1,.1,10.)


def check_two_steps(function,value,direction):
    derivative,=torch.autograd.grad(function(value),value)
    expected=(derivative*direction).sum()
    assert torch.isfinite(expected) and expected!=0
    with torch.no_grad():
        for h in (1e-5,5e-6):
            measured=(function(value+h*direction)-function(value-h*direction))/(2*h)
            torch.testing.assert_close(expected,measured,atol=1e-7,rtol=1e-4)


def test_deferred_soft_silhouette_gradient_matches_two_steps_in_stable_selection():
    d=api();_,_,vertices,_,_=fixture()
    vertices=vertices.new_tensor([[-1.,-.8,-2.],[.9,-1.1,-2.],[.2,1.,-2.]]).requires_grad_()
    faces=torch.tensor([[0,1,2]],dtype=torch.int32)
    camera=camera_one(vertices)
    def loss(v):return d.render(scene.Scene([scene.Mesh('q',v,faces)]),camera,soft=.7,outputs=('mask',))['mask'].sum()
    direction=vertices.new_tensor([[.11,-.07,.03],[.13,.02,-.05],[-.06,.08,.01]])
    check_two_steps(loss,vertices,direction)


def test_deferred_authored_normal_photometric_gradient_matches_two_steps():
    d=api();model,_,vertices,faces,_=fixture();camera=camera_one(vertices)
    normals=vertices.new_tensor([[.1,.2,1.]]).expand(4,3).clone().requires_grad_()
    mat=next(p for p in model.prims if type(p)==scene.Material)
    lamp=scene.Light('key',light.directional,dict(directions=vertices.new_tensor([[.3,.4,1.]]),irradiance=vertices.new_ones((1,3))))
    def loss(n):
        model=scene.Scene([scene.Mesh('q',vertices,faces,material='mat',attributes={'normals':scene.Attribute(n,'vertex')}),mat,lamp])
        return d.render(model,camera)['colour'].sum()
    direction=vertices.new_tensor([[.05,-.02,.01],[.03,.07,-.01],[-.02,.04,.01],[.01,-.01,.02]])
    check_two_steps(loss,normals,direction)


def test_deferred_uv_photometric_gradient_matches_two_steps():
    d=api();_,_,vertices,faces,_=fixture();camera=camera_one(vertices)
    uv=vertices.new_tensor([[.35,.4],[.45,.4],[.45,.5],[.35,.5]]).requires_grad_()
    image=vertices.new_tensor([[[.2,.3,.4],[.4,.5,.6]],[[.3,.4,.5],[.6,.7,.8]]])
    pyramid,offsets=render.texture_pyramid(image,levels=1)
    mat=scene.Material('mapped',material.classic,dict(albedo=d.TextureSource(pyramid,offsets,2,2,3,
        wrap_u=render.WRAP_CLAMP,wrap_v=render.WRAP_CLAMP),specular=vertices.new_tensor(0.),exponent=vertices.new_tensor(4.)))
    lamp=scene.Light('key',light.directional,dict(directions=vertices.new_tensor([[0.,0.,1.]]),irradiance=vertices.new_ones((1,3))))
    def loss(coords):
        model=scene.Scene([scene.Mesh('q',vertices,faces,material='mapped',attributes={'uv':scene.Attribute(coords,'vertex')}),mat,lamp])
        return d.render(model,camera)['colour'].sum()
    check_two_steps(loss,uv,vertices.new_tensor([[.1,.03],[.05,-.01],[.07,.02],[-.02,.04]]))


def test_deferred_masked_rig_photometry_reaches_shared_geometry_and_every_camera():
    d=api();_,camera,v,faces,base_albedo=fixture()
    vertices=v.clone().requires_grad_();albedo=base_albedo.clone().requires_grad_()
    fx=v.new_tensor([4.1,3.9]).requires_grad_();cx=v.new_tensor([1.95,2.05]).requires_grad_()
    mask=v.new_tensor([[[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,0.,0.],[0.,0.,1.,1.]],
                       [[0.,1.,0.,1.],[1.,0.,1.,0.],[0.,0.,1.,1.],[1.,1.,0.,0.]]])
    target=v.new_tensor([.015,.025,.035])
    lamp=scene.Light('point',light.point,dict(positions=v.new_tensor([[.3,-.2,.4]]),intensity=v.new_tensor([[1.,.8,.6]])))
    def image(vv,aa,ff,cc):
        mat=scene.Material('mat',material.classic,dict(albedo=aa,specular=aa.new_tensor(0.),exponent=aa.new_tensor(4.)))
        model=scene.Scene([scene.Mesh('q',vv,faces,material='mat'),mat,lamp])
        eye=Camera.from_intrinsics(ff,camera.fy.expand(len(ff)),cc,camera.cy.expand(len(ff)),4,4,.1,10.,
                                   pose=camera.pose.expand(len(ff),4,4))
        return d.render(model,eye)['colour']
    def loss(vv,aa,ff,cc):return ((image(vv,aa,ff,cc)-target).square()*mask[...,None]).sum()
    actual=image(vertices,albedo,fx,cx)
    separate=torch.cat([image(vertices,albedo,fx[i:i+1],cx[i:i+1]) for i in range(2)])
    assert torch.equal(actual,separate)
    parameters=(vertices,albedo,fx,cx)
    combined=torch.autograd.grad(((actual-target).square()*mask[...,None]).sum(),parameters)
    expected=torch.autograd.grad(((separate-target).square()*mask[...,None]).sum(),parameters)
    for got,want in zip(combined,expected):
        torch.testing.assert_close(got,want,atol=1e-12,rtol=1e-11)
        assert torch.isfinite(got).all() and bool((got!=0).any())
    assert bool((combined[2]!=0).all()) and bool((combined[3]!=0).all())
    # Compare the complete masked composition with independent value changes,
    # not only another composition sharing the same derivative implementation.
    for index in (0,2,3):
        direction=(torch.arange(parameters[index].numel(),dtype=v.dtype).reshape_as(parameters[index])+1)/100
        def changed_loss(changed,index=index):
            return loss(*(changed if i==index else p for i,p in enumerate(parameters)))
        check_two_steps(changed_loss,parameters[index],direction)
    with torch.no_grad():
        before=loss(*parameters)
        step=.01/(1+max(float(g.abs().max()) for g in combined))
        after=loss(*(p-step*g for p,g in zip(parameters,combined)))
        assert after<before


def test_deferred_masked_sh_photometric_gradient_matches_two_steps():
    d=api();model,camera,vertices,faces,_=fixture()
    mat=next(p for p in model.prims if type(p)==scene.Material)
    coefficients=vertices.new_zeros((9,3));coefficients[0]=1.;coefficients.requires_grad_()
    mask=vertices.new_tensor([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,0.,1.,0.],[0.,1.,0.,1.]])
    def loss(c):
        lamp=scene.Light('sky',light.spherical_harmonics,dict(coefficients=c))
        model=scene.Scene([scene.Mesh('q',vertices,faces,material='mat'),mat,lamp])
        image=d.render(model,camera)['colour']
        return ((image-.1).square()*mask[None,...,None]).sum()
    direction=vertices.new_zeros((9,3));direction[0]=.1;direction[2]=.03
    check_two_steps(loss,coefficients,direction)


@pytest.mark.parametrize('source_kind',['vertex','texture'])
def test_deferred_masked_albedo_source_gradient_matches_two_steps(source_kind):
    d=api();model,camera,vertices,faces,_=fixture()
    lamp=next(p for p in model.prims if type(p)==scene.Light)
    value=vertices.new_tensor([[.2,.3,.4],[.3,.4,.5],[.4,.5,.6],[.25,.35,.45]])
    if source_kind=='texture':value=value.reshape(2,2,3)
    value.requires_grad_()
    uv=vertices.new_tensor([[.35,.35],[.65,.35],[.65,.65],[.35,.65]])
    mask=vertices.new_tensor([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,0.,1.,0.],[0.,1.,0.,1.]])
    def loss(source):
        if source_kind=='vertex':
            binding=d.AttributeSource('colour');attributes={'colour':scene.Attribute(source,'vertex')}
        else:
            # Rebuild parameter-dependent pyramid in this exact forward graph.
            pyramid,offsets=render.texture_pyramid(source,levels=1)
            binding=d.TextureSource(pyramid,offsets,2,2,3,wrap_u=render.WRAP_CLAMP,wrap_v=render.WRAP_CLAMP)
            attributes={'uv':scene.Attribute(uv,'vertex')}
        mat=scene.Material('mapped',material.classic,dict(albedo=binding,specular=vertices.new_tensor(0.),exponent=vertices.new_tensor(4.)))
        model=scene.Scene([scene.Mesh('q',vertices,faces,material='mapped',attributes=attributes),mat,lamp])
        return ((d.render(model,camera)['colour']-.03).square()*mask[None,...,None]).sum()
    check_two_steps(loss,value,(torch.arange(12,dtype=vertices.dtype).reshape_as(value)+1)/100)
