import math
import pytest
import torch
from gffx.torch import light

@pytest.mark.parametrize('batch,shared,points',[(1,True,16),(2,True,17),(2,False,13),(1,True,0)])
def test_sh_fixed_contraction_forward_and_first_vjp_bytes(batch,shared,points):
    if not torch.cuda.is_available():pytest.skip('CPU/CUDA conformance requires CUDA')
    saved=[]
    for device in ('cpu','cuda'):
        n=(torch.arange(batch*points*3,dtype=torch.float64).reshape(batch,points,3)*.001+.17).to(device).requires_grad_()
        c=(torch.arange((1 if shared else batch)*27,dtype=torch.float64).reshape(-1,9,3)*.007-.08).to(device).requires_grad_()
        cot=(torch.arange(batch*points*3,dtype=torch.float64).reshape(batch,points,3)*.03-.27).to(device)
        out=light.spherical_harmonics(n,c)
        dn,dc=torch.autograd.grad(out,(n,c),cot)
        saved.append([x.detach().cpu().contiguous().numpy().tobytes() for x in (out,dn,dc)])
    assert saved[0]==saved[1]

def test_sh_contraction_independent_sum_and_directional_derivative():
    n=torch.tensor([[.2,.4,.8],[-.3,.7,.5]],dtype=torch.float64,requires_grad=True)
    c=(torch.arange(27,dtype=n.dtype).reshape(9,3)*.03-.2).requires_grad_()
    unit=n.detach().numpy();unit=unit/(unit**2).sum(-1,keepdims=True)**.5
    c0=math.sqrt(1/(4*math.pi));c1=math.sqrt(3/(4*math.pi));c2=math.sqrt(15/(4*math.pi));c3=math.sqrt(5/(16*math.pi));c4=math.sqrt(15/(16*math.pi))
    rows=[]
    for x,y,z in unit:
        terms=[c0,c1*y,c1*z,c1*x,c2*x*y,c2*y*z,c3*(3*z*z-1),c2*x*z,c4*(x*x-y*y)]
        factors=[math.pi]+[2*math.pi/3]*3+[math.pi/4]*5
        rows.append([math.fsum(terms[i]*factors[i]*float(c.detach()[i,k]) for i in range(9)) for k in range(3)])
    out=light.spherical_harmonics(n,c);torch.testing.assert_close(out,n.new_tensor(rows),atol=1e-15,rtol=1e-15)
    direction=n.new_tensor([[.07,-.03,.02],[-.06,.04,.08]])
    dc=c.detach()*.17+.02;cot=n.new_tensor([[.3,-.4,.7],[-.2,.6,.9]])
    gn,gc=torch.autograd.grad(out,(n,c),cot)
    expected=(gn*direction).sum()+(gc*dc).sum()
    for h in (1e-5,5e-6):
        plus=light.spherical_harmonics(n.detach()+h*direction,c.detach()+h*dc)
        minus=light.spherical_harmonics(n.detach()-h*direction,c.detach()-h*dc)
        observed=((plus-minus)*cot).sum()/(2*h)
        torch.testing.assert_close(observed,expected,atol=1e-10,rtol=1e-9)
