"""Prospective 8x8 built-in profiles; execute only after planner controls pass."""
import torch
from gffx.torch import scene, material, light
from gffx.torch.camera import Camera
from test_deferred import api
from test_deferred_recompute import resource


def workload(backward):
    def t(value):return torch.tensor(value,dtype=torch.float64)
    roughness=t(.613).requires_grad_(backward)
    vertices=t([[-2.,-2.,-2.],[2.,-2.,-2.],[2.,2.,-2.],[-2.,2.,-2.]])
    faces=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)
    mat=scene.Material('p',material.principled,dict(albedo=t([.2,.3,.4]),roughness=roughness,f0=t(.071),budget=resource()))
    lamp=scene.Light('key',light.directional,dict(directions=t([[.2,.3,1.]]),irradiance=t([[1.,1.,1.]])))
    model=scene.Scene([scene.Mesh('q',vertices,faces,material='p'),mat,lamp])
    camera=Camera.from_intrinsics(t(8.),t(8.),t(4.),t(4.),8,8,.1,10.)
    return model,camera,roughness


def test_deferred_8x8_principled_forward_profile():
    d=api()
    with torch.no_grad():
        model,camera,_=workload(False)
        estimate=d._plan_workload(model,camera,1,67108864,('colour','mask'))
        assert estimate['builtins_only'] and estimate['estimated_bytes']<=67108864
        image=d.render(model,camera)
    assert image['colour'].shape==(1,8,8,3)
    assert torch.isfinite(image['colour']).all() and bool((image['colour']>0).all())
    assert torch.equal(image['mask'],torch.ones((1,8,8),dtype=torch.float64))


def test_deferred_8x8_principled_backward_profile():
    d=api();model,camera,roughness=workload(True)
    estimate=d._plan_workload(model,camera,1,67108864,('colour','mask'))
    assert estimate['builtins_only'] and estimate['backward'] and estimate['estimated_bytes']<=67108864
    image=d.render(model,camera)['colour']
    gradient,=torch.autograd.grad(image.sum(),roughness)
    assert torch.isfinite(image).all() and torch.isfinite(gradient) and gradient!=0
