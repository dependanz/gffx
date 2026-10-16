"""Matched GGX constant-divisor regression and independent analytic derivatives."""
import math,pytest,torch
from gffx.torch import shade
from test_deferred_cuda import exact

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_ggx_constant_division_device_values_vjps_and_cpu_legacy(dtype):
    c=torch.cat((torch.linspace(0,1,257,dtype=dtype),torch.tensor([.9895880081431454,.9904780038426806],dtype=dtype)))
    r=torch.full_like(c,.63);cot=torch.arange(len(c),dtype=dtype).remainder(7)*.125-.375
    rows=[]
    for device in ('cpu','cuda'):
        q=c.to(device).requires_grad_();a=r.to(device).requires_grad_()
        y=shade.ggx_distribution(q,a);grad=torch.autograd.grad(y,(q,a),cot.to(device));rows.append((y,*grad))
    for i,(a,b) in enumerate(zip(*rows)):exact(a,b,'GGX field '+str(i))
    q=c.clone().requires_grad_();a=r.clone().requires_grad_();fourth=a.square().square()
    legacy=(fourth/((1-q)*(1+q)+fourth*q.square()).square())/math.pi
    gradients=torch.autograd.grad(legacy,(q,a),cot)
    for actual,expected in zip(rows[0],(legacy,*gradients)):exact(actual,expected,'legacy CPU GGX')
    den=1+(r**4-1)*c.square()
    dc=-4*(r**4)*(r**4-1)*c/(math.pi*den**3)
    dr=4*r**3*(den-2*(r**4)*c.square())/(math.pi*den**3)
    for actual,expected in zip(rows[0][1:],(cot*dc,cot*dr)):
        torch.testing.assert_close(actual,expected,atol=1e-5 if dtype==torch.float32 else 1e-11,rtol=1e-5 if dtype==torch.float32 else 1e-11)
