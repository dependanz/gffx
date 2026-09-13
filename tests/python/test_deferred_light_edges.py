"""Invalid samples have zero numerical contributions, not physically dark lamps."""
import math
import torch
from gffx.torch import scene,material,light,render
from gffx.torch.camera import Camera
from test_deferred import api,fixture


def test_deferred_point_light_coincidence_is_finite_zero_with_zero_gradient():
    d=api()
    def t(value):return torch.tensor(value,dtype=torch.float64)
    source=t([[0.,0.,-2.]]).requires_grad_()
    lamp=scene.Light('point',light.point,dict(positions=source,intensity=t([[1.,1.,1.]])))
    mat=scene.Material('m',material.classic,dict(albedo=t([.2,.3,.4]),specular=t(0.),exponent=t(4.)))
    vertices=t([[-1.,-1.,-2.],[1.,-1.,-2.],[1.,1.,-2.],[-1.,1.,-2.]])
    faces=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)
    model=scene.Scene([scene.Mesh('q',vertices,faces,material='m'),mat,lamp])
    camera=Camera.from_intrinsics(t(1.),t(1.),t(.5),t(.5),1,1,.1,10.)
    result=d.render(model,camera)['colour']
    assert torch.equal(result,torch.zeros_like(result))
    gradient,=torch.autograd.grad(result.sum(),source)
    assert torch.equal(gradient,torch.zeros_like(source))


def test_deferred_diffuse_sh_respects_one_sided_view_hemisphere():
    d=api();_,camera,vertices,faces,albedo=fixture()
    coefficients=vertices.new_zeros((9,3));coefficients[0]=1;coefficients.requires_grad_()
    mat=scene.Material('m',material.classic,dict(albedo=albedo,specular=albedo.new_tensor(0.),exponent=albedo.new_tensor(4.)))
    lamp=scene.Light('sky',light.spherical_harmonics,dict(coefficients=coefficients))
    model=scene.Scene([scene.Mesh('q',vertices,faces[:,[0,2,1]].contiguous(),material='m'),mat,lamp])
    image=d.render(model,camera,cull_mode=render.CULL_NONE)['colour']
    assert torch.equal(image,torch.zeros_like(image))
    gradient,=torch.autograd.grad(image.sum(),coefficients)
    assert torch.equal(gradient,torch.zeros_like(coefficients))


def test_deferred_two_sided_sh_rotates_to_the_viewed_normal():
    d=api();_,camera,vertices,faces,albedo=fixture()
    coefficients=vertices.new_zeros((9,3));coefficients[0]=1;coefficients[2]=.2
    mat=scene.Material('m',material.classic,dict(albedo=albedo,specular=albedo.new_tensor(0.),
        exponent=albedo.new_tensor(4.),two_sided=True))
    lamp=scene.Light('sky',light.spherical_harmonics,dict(coefficients=coefficients))
    model=scene.Scene([scene.Mesh('q',vertices,faces[:,[0,2,1]].contiguous(),material='m'),mat,lamp])
    image=d.render(model,camera,cull_mode=render.CULL_NONE)['colour']
    expected=albedo*(1/math.sqrt(4*math.pi)+.2*math.sqrt(3/(4*math.pi))*2/3)
    torch.testing.assert_close(image,expected.expand_as(image),atol=1e-13,rtol=1e-13)
