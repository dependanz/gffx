"""Pre-allocation refusal controls; spies avoid allocating refused workloads."""
import pytest
import torch
from gffx.torch import scene, material, light
from gffx.torch.camera import Camera
from test_deferred import api, fixture


def test_deferred_resource_refusal_precedes_scene_flatten(monkeypatch):
    d=api();model,camera,*_=fixture()
    def forbidden(*args,**kwargs):
        raise AssertionError('scene flatten allocated before workload refusal')
    monkeypatch.setattr(scene.Scene,'flatten',forbidden)
    with pytest.raises(ValueError,match='budget'):
        d.render(model,camera,workspace_limit_bytes=1)


def test_deferred_resource_estimate_counts_light_packets_before_raster(monkeypatch):
    d=api();model,_,vertices,faces,albedo=fixture()
    # Small input fixture, but expanded packet/BRDF storage exceeds 32 KiB.
    directions=vertices.new_tensor([[0.,0.,1.]]).expand(1024,3).contiguous()
    lamp=scene.Light('many',light.directional,dict(directions=directions,irradiance=torch.ones_like(directions)))
    mat=next(p for p in model.prims if type(p)==scene.Material)
    model=scene.Scene([scene.Mesh('q',vertices,faces,material='mat'),mat,lamp])
    camera=Camera.from_intrinsics(albedo.new_tensor(1.),albedo.new_tensor(1.),albedo.new_tensor(.5),
                                  albedo.new_tensor(.5),1,1,.1,10.)
    def forbidden(*args,**kwargs):
        raise AssertionError('raster allocated before light packet workload refusal')
    monkeypatch.setattr(d,'_buffers',forbidden)
    with pytest.raises(ValueError,match='budget'):
        d.render(model,camera,workspace_limit_bytes=32768)


def test_deferred_resource_components_sum_and_scale_with_samples_and_supersampling():
    d=api();model,camera,vertices,faces,_=fixture()
    mat=next(p for p in model.prims if type(p)==scene.Material)
    def plan(samples,factor):
        directions=vertices.new_tensor([[0.,0.,1.]]).expand(samples,3).contiguous()
        lamp=scene.Light('q',light.directional,dict(directions=directions,irradiance=torch.ones_like(directions)))
        model=scene.Scene([scene.Mesh('m',vertices,faces,material='mat'),mat,lamp])
        return d._plan_workload(model,camera,factor,67108864,('colour','mask'))
    one,many,fine=plan(1,1),plan(5,1),plan(1,2)
    for estimate in (one,many,fine):
        assert estimate['estimated_bytes']==sum(estimate['components'].values())
        assert estimate['builtins_only'] is True
        assert estimate['components']['inputs']>0 and estimate['components']['packed']>0
    assert many['components']['lighting']==5*one['components']['lighting']
    assert fine['components']['lighting']==4*one['components']['lighting']
    assert fine['components']['buffers']>one['components']['buffers']


def test_deferred_resource_principled_scratch_is_added_to_live_storage():
    from test_deferred_recompute import resource
    d=api();model,camera,vertices,faces,albedo=fixture()
    budget=resource();roughness=albedo.new_tensor(.613).requires_grad_()
    mat=scene.Material('p',material.principled,dict(albedo=albedo,roughness=roughness,f0=albedo.new_tensor(.071),budget=budget))
    lamp=next(p for p in model.prims if type(p)==scene.Light)
    model=scene.Scene([scene.Mesh('q',vertices,faces,material='p'),mat,lamp])
    estimate=d._plan_workload(model,camera,1,67108864,('colour','mask'))
    # Sixteen possible selected pixels form one native cohort in this fixture.
    expected=8+8*16*(64*len(budget._data['cosine_nodes'])+64+192)
    assert estimate['backward'] is True
    assert estimate['components']['material_scratch']==expected
    assert estimate['estimated_bytes']>expected


def test_deferred_resource_buffer_estimate_respects_actual_reuse():
    d=api();model,camera,vertices,_,_=fixture()
    plan=lambda factor,outputs:d._plan_workload(model,camera,factor,67108864,outputs)
    coarse=plan(1,('colour','mask'));same=plan(1,('colour','depth'))
    fine=plan(2,('colour','mask'));separate=plan(2,('colour','depth'))
    pixel_bytes=256*vertices.element_size()+32
    assert coarse['components']['buffers']==16*pixel_bytes
    assert same['components']['buffers']==coarse['components']['buffers']
    assert fine['components']['buffers']==4*coarse['components']['buffers']
    assert separate['components']['buffers']==fine['components']['buffers']+coarse['components']['buffers']
    soft=d._plan_workload(model,camera,1,67108864,('colour','depth'),soft=.5)
    assert soft['components']['buffers']==2*coarse['components']['buffers']


def test_deferred_resource_lighting_counts_disjoint_material_pixels_once():
    d=api();model,camera,vertices,faces,albedo=fixture()
    one=d._plan_workload(model,camera,1,67108864,('colour','mask'))
    mat1=next(p for p in model.prims if type(p)==scene.Material)
    mat2=scene.Material('other',material.classic,dict(albedo=albedo,specular=albedo.new_tensor(0.),exponent=albedo.new_tensor(4.)))
    lamp=next(p for p in model.prims if type(p)==scene.Light)
    split=scene.Scene([scene.Mesh('one',vertices,faces,material='mat'),scene.Mesh('two',vertices,faces,material='other'),mat1,mat2,lamp])
    two=d._plan_workload(split,camera,1,67108864,('colour','mask'))
    # One retained fragment selects at most one material at every pixel.
    assert one['components']['lighting']==two['components']['lighting']
    assert one['components']['selection_masks']>0
    assert two['components']['selection_masks']==2*one['components']['selection_masks']


def test_deferred_resource_light_only_fit_counts_backward_and_borrowed_storage():
    d=api();model,camera,vertices,faces,_=fixture()
    mat=next(p for p in model.prims if type(p)==scene.Material)
    def estimate(fitted):
        directions=vertices.new_tensor([[0.,0.,1.]])
        intensity=vertices.new_tensor([[.4,.7,1.]]).requires_grad_(fitted)
        lamp=scene.Light('fit',light.directional,dict(directions=directions,irradiance=intensity))
        bound=scene.Scene([scene.Mesh('q',vertices,faces,material='mat'),mat,lamp])
        return d._plan_workload(bound,camera,1,67108864,('colour','mask'))
    plain,fitted=estimate(False),estimate(True)
    assert plain['backward'] is False
    assert fitted['backward'] is True
    assert fitted['components']['inputs']==3*plain['components']['inputs']
    assert fitted['components']['lighting']==4*plain['components']['lighting']


def test_deferred_resource_immutable_budget_has_no_duplicate_adjoint_storage():
    from test_deferred_recompute import resource
    d=api();model,camera,vertices,faces,albedo=fixture();budget=resource()
    lamp=next(p for p in model.prims if type(p)==scene.Light)
    def estimate(fitted):
        roughness=albedo.new_tensor(.613).requires_grad_(fitted)
        mat=scene.Material('p',material.principled,dict(albedo=albedo,roughness=roughness,
            f0=albedo.new_tensor(.071),budget=budget))
        bound=scene.Scene([scene.Mesh('q',vertices,faces,material='p'),mat,lamp])
        return d._plan_workload(bound,camera,1,67108864,('colour','mask'))
    plain,fitted=estimate(False),estimate(True)
    persistent=budget.metadata['persistent_tensor_bytes']
    assert fitted['components']['inputs']==3*plain['components']['inputs']-2*persistent
    assert fitted['components']['material_scratch']==8+8*16*(64*len(budget._data['cosine_nodes'])+64+192)


def test_deferred_resource_alias_in_ordinary_binding_keeps_derivative_allowance():
    from test_deferred_recompute import resource
    d=api();model,camera,vertices,faces,albedo=fixture();budget=resource()
    roughness=albedo.new_tensor(.613).requires_grad_()
    mat=scene.Material('p',material.principled,dict(albedo=albedo,roughness=roughness,
        f0=albedo.new_tensor(.071),budget=budget))
    lamp=next(p for p in model.prims if type(p)==scene.Light)
    prims=[scene.Mesh('q',vertices,faces,material='p'),mat,lamp]
    plain=d._plan_workload(scene.Scene(prims),camera,1,67108864,('colour','mask'))
    alias=budget._data['_weights']
    other=scene.Material('trusted',lambda *args,**kwargs:None,{'coefficients':alias})
    shared=d._plan_workload(scene.Scene(prims+[other]),camera,1,67108864,('colour','mask'))
    assert shared['components']['inputs']==plain['components']['inputs']+2*alias.untyped_storage().nbytes()
    assert shared['builtins_only'] is False
