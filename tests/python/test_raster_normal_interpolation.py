import math
import pytest
import torch
import gffx as g
from gffx.torch import _raster_accel as accel

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_normal_field_rejects_opposite_hemisphere_probe(device):
    # Independent two-layer field: only probes above the receiver carry energy.
    c=torch.zeros(2,2,2,9,3,device=device);c[:,:,1,0]=1/.28209479177387814
    v=g.IrradianceVolume(((-1,-1,-1),(1,1,1)),c,normal_bias=0,interpolation='normal')
    p=torch.zeros(2,3,device=device);n=p.clone();n[0,2]=1;n[1,2]=-1
    value,inside=v.sample(p,n);assert bool(inside.all())
    expected=torch.tensor([[1.,1,1],[0,0,0]],device=device);torch.testing.assert_close(value,expected)
    data=accel.pack_indirect(None,[],v,[],p);params=dict(albedo=torch.ones(3,device=device),specular=torch.zeros(3,device=device))
    actual=accel.indirect(p,n,n,params,data,torch.full((2,),-1,device=device,dtype=torch.int32),classic=True)
    assert actual is not None;torch.testing.assert_close(actual,expected/math.pi,atol=2e-6,rtol=1e-5)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_normal_weights_preserve_constant_and_closed_influence(device):
    c=torch.zeros(2,2,2,9,3,device=device);c[...,0,:]=1/.28209479177387814
    v=g.IrradianceVolume(((0,0,0),(1,1,1)),c,normal_bias=.2,interpolation='normal')
    p=torch.tensor([[0.,0,0],[1,1,1],[.5,.5,.5],[.5,.5,-.01]],device=device);n=torch.tensor([[-1.,0,0],[1,0,0],[0,0,1],[0,0,1]],device=device)
    actual,inside=v.sample(p,n);torch.testing.assert_close(actual,torch.ones_like(p));assert inside.tolist()==[True,True,True,False]
    env=g.Environment.constant([.1,.2,.3],device=device);data=accel.pack_indirect(env,[],v,[],p)
    params=dict(albedo=torch.ones(3,device=device),specular=torch.zeros(3,device=device));expected=torch.ones_like(p)/math.pi;expected[-1]=p.new_tensor([.1,.2,.3])
    native=accel.indirect(p,n,n,params,data,torch.full((4,),-1,device=device,dtype=torch.int32),classic=True)
    torch.testing.assert_close(native,expected,atol=2e-6,rtol=1e-5)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_baked_normal_field_constant_environment_energy(device):
    # Actual six-view bake, not a mocked interpolation implementation.
    mat=g.Material.diffuse('black',(0,0,0),device=device)
    obj=g.Mesh.quad('distant',center=(20,20,20),u=(.01,0,0),v=(0,0,.01),material='black',device=device)
    env=g.Environment.constant([.1,.2,.3],device=device)
    v=g.IrradianceVolume.bake(g.Scene([mat,obj]),bounds=((-1,-1,-1),(1,1,1)),resolution=(2,2,2),capture_size=4,environment=env,shadows=False,interpolation="normal")
    assert v.interpolation=='normal'
    p=torch.zeros(3,3,device=device);n=torch.eye(3,device=device);a,inside=v.sample(p,n)
    torch.testing.assert_close(a,p.new_tensor([.1,.2,.3]).expand_as(p)*math.pi,atol=2e-5,rtol=2e-5)

def test_invalid_irradiance_interpolation_mode():
    with pytest.raises(ValueError,match='interpolation'):
        g.IrradianceVolume(((0,0,0),(1,1,1)),torch.zeros(2,2,2,9,3),interpolation='unrecognised')


def test_volume_bake_default_preserves_trilinear_contract():
    from dataclasses import replace
    mat=g.Material.emissive('red',(1,0,0),device='cpu')
    quad=g.Mesh.quad('ceiling',center=(0,0,3),u=(2,0,0),v=(0,-2,0),material='red',device='cpu')
    v=g.IrradianceVolume.bake(g.Scene([mat,quad]),bounds=((-1,-1,-1),(1,1,1)),resolution=(2,2,2),capture_size=8,shadows=False)
    p=torch.zeros(1,3);n=torch.tensor([[0.,0.,1.]])
    expected,_=replace(v,interpolation='trilinear').sample(p,n)
    actual,_=v.sample(p,n);torch.testing.assert_close(actual,expected)
    alternative,_=replace(v,interpolation='normal').sample(p,n)
    assert float((alternative-expected).abs().max())>1e-4
