"""Forward-only largest-fitting point partitions; unchanged math and caps."""
import pytest
import torch
from gffx.torch import scene, material, light
from test_deferred import api, fixture
from test_deferred_recompute import resource, fields


def model_and_limits(*, fitted=False):
    d=api();original,camera,vertices,faces,albedo=fixture();budget=resource()
    roughness=albedo.new_tensor(.613).requires_grad_(fitted)
    mat=scene.Material('p',material.principled,dict(albedo=albedo,roughness=roughness,
        f0=albedo.new_tensor(.071),budget=budget))
    lamp=next(p for p in original.prims if type(p)==scene.Light)
    model=scene.Scene([scene.Mesh('q',vertices,faces,material='p'),mat,lamp])
    full=d._plan_workload(model,camera,1,67108864,('colour','mask'))
    fixed=full['estimated_bytes']-full['components']['material_scratch']
    scratch=lambda block:3*budget.workspace(3*block,backward=full['backward'])+8*block*(192 if full['backward'] else 64)
    return model,camera,{block:fixed+scratch(block) for block in (1,2,4)}


@pytest.mark.parametrize('block',[4,2,1])
def test_deferred_forward_planner_selects_largest_fitting_block(block):
    d=api();model,camera,limits=model_and_limits()
    plan=d._plan_workload(model,camera,1,limits[block],('colour','mask'))
    assert plan['backward'] is False
    assert plan['principled_point_block_limit']==block
    assert plan['estimated_bytes']==limits[block]
    assert plan['estimated_bytes']==sum(plan['components'].values())
    assert plan['estimated_bytes']<=67108864


def test_deferred_forward_one_point_refusal_precedes_flatten(monkeypatch):
    d=api();model,camera,limits=model_and_limits()
    def forbidden(*args,**kwargs):raise AssertionError('flatten before refusal')
    monkeypatch.setattr(scene.Scene,'flatten',forbidden)
    with pytest.raises(ValueError,match='budget'):
        d.render(model,camera,workspace_limit_bytes=limits[1]-1)


def test_deferred_backward_never_shrinks_its_four_point_partition():
    d=api();model,camera,limits=model_and_limits(fitted=True)
    plan=d._plan_workload(model,camera,1,67108864,('colour','mask'))
    assert plan['backward'] is True and plan['principled_point_block_limit']==4
    with pytest.raises(ValueError,match='budget'):
        d._plan_workload(model,camera,1,limits[4]-1,('colour','mask'))
    # An explicit no-grad context is genuinely forward-only even for fit leaves.
    with torch.no_grad():
        _,_,forward_limits=model_and_limits(fitted=True)
        forward=d._plan_workload(model,camera,1,forward_limits[1],('colour','mask'))
    assert forward['backward'] is False and forward['principled_point_block_limit']==1


@pytest.mark.parametrize('batched_fields',[False,True])
def test_deferred_forward_partition_values_are_bit_identical(batched_fields):
    d=api();dtype=torch.float64
    def t(values):return torch.tensor(values,dtype=dtype)
    normals=t([[.1,.2,1.],[-.1,.15,1.],[.05,-.1,1.],[-.2,-.1,1.]])
    view=t([[.2,.1,1.],[.1,-.2,1.],[-.3,.2,1.],[.1,.1,1.]])
    directions=t([[[.2,.3,1.]],[[.15,.2,1.]],[[.1,-.1,1.]],[[.3,.1,1.]]])
    params=dict(albedo=t([[.2,.3,.4],[.3,.4,.5],[.1,.2,.3],[.5,.4,.3]]),
        roughness=t([.45,.613,.7,.8]),f0=t([[.04],[.071],[.1],[.12]]),budget=resource())
    if batched_fields:
        # Preserve the explicit singleton batch and strided field layouts.
        params['albedo']=params['albedo'].T.contiguous().T[None]
        params['f0']=params['f0'][None]
        params['roughness']=params['roughness'][None]
    with torch.no_grad():
        expected=material.principled(normals,view,directions,**params)
        for block in (4,2,1):
            actual=d._principled_blocks(normals,view,directions,params,point_block_limit=block)
            assert torch.equal(actual,expected)


def test_deferred_forward_runtime_uses_the_admitted_partition(monkeypatch):
    d=api();model,camera,limits=model_and_limits();calls=[]
    original=d._principled_recompute
    def record(normals,view,directions,parameters):
        calls.append(len(normals))
        return original(normals,view,directions,parameters)
    monkeypatch.setattr(d,'_principled_recompute',record)
    expected=None
    for block in (4,2,1):
        calls.clear()
        actual=d.render(model,camera,workspace_limit_bytes=limits[block])
        assert calls and max(calls)<=block and sum(calls)==int(actual['mask'].sum())
        assert all(size==block for size in calls[:-1])
        if expected is None:expected=actual
        else:
            for name in expected:assert torch.equal(actual[name],expected[name])


def test_deferred_forward_partition_keeps_point_sample_limit(monkeypatch):
    d=api();calls=[];normals=torch.ones((3,3),dtype=torch.float64)
    directions=torch.ones((3,513,3),dtype=torch.float64)
    def record(n,v,l,params):
        calls.append((len(n),l.shape[-2]))
        return n.new_zeros((len(n),l.shape[-2],3))
    monkeypatch.setattr(d,'_principled_recompute',record)
    result=d._principled_blocks(normals,normals,directions,{},point_block_limit=4)
    assert result.shape==(3,513,3) and calls==[(1,513)]*3
    assert all(points*samples<=1024 for points,samples in calls)


@pytest.mark.parametrize('case',['empty','offscreen','no_lights','zero_samples','back_light'])
def test_deferred_forward_partition_handles_empty_and_inactive_work(case,monkeypatch):
    d=api();model,camera,_=model_and_limits()
    mesh=next(p for p in model.prims if type(p)==scene.Mesh)
    mat=next(p for p in model.prims if type(p)==scene.Material)
    lamp=next(p for p in model.prims if type(p)==scene.Light)
    if case=='empty':
        mesh=scene.Mesh('q',mesh.vertices[:0],mesh.faces[:0],material='p')
    elif case=='offscreen':
        mesh=scene.Mesh('q',mesh.vertices+mesh.vertices.new_tensor([100.,0.,0.]),mesh.faces,material='p')
    elif case in ('zero_samples','back_light'):
        direction=mesh.vertices.new_empty((0,3)) if case=='zero_samples' else mesh.vertices.new_tensor([[0.,0.,-1.]])
        lamp=scene.Light('key',light.directional,dict(directions=direction,irradiance=torch.ones_like(direction)))
    model=scene.Scene([mesh,mat]+([] if case=='no_lights' else [lamp]))
    plan=d._plan_workload(model,camera,1,67108864,('colour','mask'))
    assert plan['estimated_bytes']==sum(plan['components'].values())
    assert plan['backward'] is False
    calls=[];original=d._principled_recompute
    def record(n,v,l,parameters):
        calls.append((len(n),l.shape[-2]))
        return original(n,v,l,parameters)
    monkeypatch.setattr(d,'_principled_recompute',record)
    result=d.render(model,camera)
    assert torch.equal(result['colour'],torch.zeros_like(result['colour']))
    if case in ('empty','offscreen','no_lights'):assert calls==[]
    if case in ('no_lights','zero_samples'):
        assert plan['components']['material_scratch']==0
    if case=='zero_samples':assert calls and all(samples==0 for _,samples in calls)
    if case=='back_light':
        # Layout is charged before material hemisphere masks, never discounted.
        assert calls and plan['components']['material_scratch']>0


def test_deferred_backward_partition_preserves_every_matched_cotangent():
    d=api();arguments=fields();n,v,l,a,r,f=arguments
    params=dict(albedo=a,roughness=r,f0=f,budget=resource())
    direct=material.principled(n,v,l,**params)
    actual=d._principled_blocks(n,v,l,params,point_block_limit=4)
    assert torch.equal(actual,direct)
    weights=torch.arange(1,direct.numel()+1,dtype=n.dtype).reshape_as(direct)/10
    expected=torch.autograd.grad((direct*weights).sum(),arguments)
    observed=torch.autograd.grad((actual*weights).sum(),arguments)
    for left,right in zip(expected,observed):assert torch.equal(left,right)
    with pytest.raises(ValueError,match='backward partition'):
        d._principled_blocks(n,v,l,params,point_block_limit=1)
