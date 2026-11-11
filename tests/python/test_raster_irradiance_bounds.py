"""An unbiased point decides influence; normal bias only chooses SH lookup."""
import math
import pytest
import torch
import gffx as g
from gffx.torch._raster_lighting import surface
from gffx.torch import _raster_accel as accel

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('bias',[0.,.02,.2])
def test_irradiance_normal_bias_does_not_expand_influence(device,bias):
    c=torch.zeros(2,2,2,9,3,device=device)
    c[:,:,0,0]=1/.28209479177387814
    c[:,:,1,0]=3/.28209479177387814
    volume=g.IrradianceVolume(((0,0,0),(1,1,1)),c,normal_bias=bias)
    p=torch.tensor([[.5,.5,-.01],[.5,.5,0],[.5,.5,.5],[.5,.5,.99],[-.01,.5,.5],[1.01,.5,.5]],device=device)
    n=torch.tensor([[0.,0.,1.]],device=device).expand_as(p)
    value,inside=volume.sample(p,n)
    expected_inside=torch.tensor([False,True,True,True,False,False],device=device)
    assert torch.equal(inside,expected_inside)
    query=(p[:,2]+bias).clamp(0,1)
    torch.testing.assert_close(value,(1+2*query)[:,None].expand_as(value))
    env=g.Environment.constant([.1,.2,.3],device=device)
    params=dict(albedo=torch.ones(3,device=device),specular=torch.zeros(3,device=device))
    expected=torch.tensor([.1,.2,.3],device=device).expand_as(p).clone()
    expected[expected_inside]=(1+2*query[expected_inside,None])/math.pi
    actual=surface(p,n,n,params,environment=env,probes=(),irradiance=volume,classic=True)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    data=accel.pack_indirect(env,[],volume,[],p)
    native=accel.indirect(p,n,n,params,data,torch.full((len(p),),-1,device=device,dtype=torch.int32),classic=True)
    assert native is not None
    torch.testing.assert_close(native,expected,atol=2e-6,rtol=2e-5)
