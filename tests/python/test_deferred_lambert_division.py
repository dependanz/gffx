"""Exact Lambert division regression; no tolerance-based device fallback."""
import math
import pytest
import torch
from gffx.torch import shade
from test_deferred_cuda import exact, cuda_required


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_deferred_lambert_division_matches_device_values_and_vjps(cuda_required,dtype):
    rows=[]
    for device in ('cpu','cuda'):
        albedo=torch.tensor([[.25,.5,.75]],dtype=dtype,device=device,requires_grad=True)
        cotangent=albedo.new_tensor([[.25,.5,.75]])
        value=shade.lambert(albedo)
        gradient,=torch.autograd.grad(value,albedo,cotangent)
        rows.append((value,gradient))
    exact(rows[0][0],rows[1][0],'Lambert value')
    exact(rows[0][1],rows[1][1],'Lambert albedo VJP')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_deferred_lambert_division_preserves_cpu_legacy_bytes(dtype):
    albedo=torch.tensor([[-0.,0.,1.],[.25,.5,.75],[.123,.337,.999]],dtype=dtype,
                        requires_grad=True).T
    cotangent=torch.tensor([[.031,.7,.91],[.125,.625,.875],[.19,.51,.83]],dtype=dtype)
    legacy=albedo/math.pi
    value=shade.lambert(albedo)
    expected,=torch.autograd.grad(legacy,albedo,cotangent)
    actual,=torch.autograd.grad(value,albedo,cotangent)
    exact(legacy,value,'CPU legacy Lambert')
    exact(expected,actual,'CPU legacy Lambert VJP')
