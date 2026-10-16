"""Degree-nine simplex moments and shape derivatives for the SH25 rule."""
import math
import torch
import pytest

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sh25_degree_nine_moments_and_geometry_vjp(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 from gffx.torch.render import opaque_material_quadrature_batch
 v=torch.tensor([[.1,.1,.2],[.8,.1,.2],[.1,.8,.2]],dtype=torch.float64,device=device,requires_grad=True)
 f=torch.tensor([[0,1,2]],dtype=torch.int32,device=device);c=v.new_empty((1,3,0))
 ids,q,r=opaque_material_quadrature_batch(v,f,c,rule=3)
 assert r[1] and r[2]==25 and r[3]==1
 q=q[ids>=0];det=(v[1,0]-v[0,0])*(v[2,1]-v[0,1])-(v[1,1]-v[0,1])*(v[2,0]-v[0,0])
 for i in range(10):
  for j in range(10-i):
   got=(q[:,0]**i*q[:,1]**j*q[:,3]).sum();expected=det*math.factorial(i)*math.factorial(j)/math.factorial(i+j+2)
   torch.testing.assert_close(got,expected,rtol=0,atol=2e-14)
 got=(q[:,0]**9*q[:,3]).sum();expected=det/110
 a=torch.autograd.grad(got,v,retain_graph=True)[0];b=torch.autograd.grad(expected,v)[0]
 torch.testing.assert_close(a,b,rtol=0,atol=2e-13)
