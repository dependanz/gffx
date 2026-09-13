"""Failure-first edge contracts, separate from the initial renderer smoke tranche."""
import pytest
import torch
from gffx.torch import scene,material
from test_deferred import api,fixture


@pytest.mark.parametrize('radius',[1e-200,1e200])
def test_deferred_soft_radius_squared_must_be_representable(radius):
    d=api();s,c,*_=fixture()
    with pytest.raises(ValueError,match='soft'):
        d.render(s,c,soft=radius,outputs=('mask',))


@pytest.mark.parametrize('kind',['name','scalar','height','offset_dtype'])
def test_deferred_empty_scene_still_validates_source_structure(kind):
    d=api();_,c,v,_,_=fixture()
    source=d.AttributeSource(42) if kind=='name' else d.AttributeSource('uv',scalar=1)
    if kind in ('height','offset_dtype'):
        source=d.TextureSource(v.new_tensor([.2,.3,.4]),torch.tensor([0,3],
            dtype=torch.float64 if kind=='offset_dtype' else torch.int32),0 if kind=='height' else 1,1,3)
    mat=scene.Material('unused',material.classic,dict(albedo=source,specular=v.new_tensor(0.),exponent=v.new_tensor(4.)))
    with pytest.raises((TypeError,ValueError)):
        d.render(scene.Scene([mat]),c,outputs=('mask',))


def test_deferred_source_parameters_reject_mixed_dtype_even_when_unused():
    d=api();_,c,v,_,_=fixture()
    mat=scene.Material('unused',material.classic,dict(albedo=torch.ones(3,dtype=torch.float32),
        specular=v.new_tensor(0.),exponent=v.new_tensor(4.)))
    with pytest.raises(TypeError):d.render(scene.Scene([mat]),c,outputs=('mask',))
