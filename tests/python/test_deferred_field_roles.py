"""Explicit binding roles must not mistake three scalar pixels for constant RGB."""
import pytest
import torch
from gffx.torch import scene,material,light
from gffx.torch.camera import Camera
from test_deferred import api


def field_scene(scalar):
    d=api()
    def t(value):return torch.tensor(value,dtype=torch.float64)
    vertices=t([[-2.,-1.,-2.],[2.,-1.,-2.],[2.,1.,-2.],[-2.,1.,-2.]])
    faces=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)
    grey=t([[.1],[.3],[.3],[.1]])
    mat=scene.Material('grey',material.classic,dict(albedo=t([0.,0.,0.]),
        specular=d.AttributeSource('strength',scalar=scalar),exponent=t(4.)))
    lamp=scene.Light('key',light.directional,dict(directions=t([[0.,0.,1.]]),irradiance=t([[1.,1.,1.]])))
    model=scene.Scene([scene.Mesh('q',vertices,faces,material='grey',attributes={'strength':scene.Attribute(grey,'vertex')}),mat,lamp])
    camera=Camera.from_intrinsics(t(2.),t(2.),t(1.5),t(.5),1,3,.1,10.)
    return model,camera


def test_deferred_three_pixel_rgb_field_rejects_scalar_axis_removal():
    d=api();model,camera=field_scene(True)
    with pytest.raises(ValueError,match='retain.*channel.*axis'):
        d.render(model,camera)


def test_deferred_three_pixel_grey_field_retains_channel_axis():
    d=api();model,camera=field_scene(False)
    colour=d.render(model,camera)['colour']
    assert colour.shape==(1,1,3,3) and torch.isfinite(colour).all()
    assert torch.equal(colour[...,0],colour[...,1]) and torch.equal(colour[...,1],colour[...,2])
    assert bool((colour[...,0]>0).all())
    assert colour[0,0,0,0]!=colour[0,0,2,0]
