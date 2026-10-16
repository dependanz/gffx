"""Independent positive-unit logarithm bounds and captured device regressions."""
from decimal import Decimal,localcontext
from fractions import Fraction
import math
import pytest,torch


def test_log_atanh_remainder_and_split_constant_certificate():
    z=Fraction(9,50)
    assert 2*z**41/(41*(1-z*z))<Fraction(1,10**31)
    with localcontext() as c:
        c.prec=100
        hi=Decimal.from_float(float.fromhex('0x1.62e4200000000p-1'))
        lo=Decimal.from_float(float.fromhex('0x1.fdf473de6af28p-22'))
        assert abs(Decimal(2).ln()-hi-lo)*1074<Decimal('1e-19')


def test_log_unit_independent_decimal_and_device_first_vjp():
    from gffx.torch._curve_math import log_unit
    powers=torch.tensor([math.ldexp(1.,k) for k in range(-1074,1)],dtype=torch.float64)
    linear=torch.linspace(2**-10,1.,4097,dtype=torch.float64)
    fixed=torch.tensor([0.6378118574468598,0.9202568782722006,1/math.sqrt(2)],dtype=torch.float64)
    base=torch.cat((powers,linear,fixed))
    x=torch.cat((base,torch.nextafter(base,torch.zeros_like(base)),torch.nextafter(base,torch.ones_like(base))))
    x=x[(x>0)&(x<=1)].unique();lanes=[]
    cot=x*torch.tensor([-.5,.75,1.],dtype=x.dtype)[torch.arange(len(x)).remainder(3)]
    for device in ('cpu','cuda'):
        v=x.to(device).requires_grad_();y=log_unit(v);g,=torch.autograd.grad(y,v,cot.to(device));lanes.append((y.detach().cpu(),g.cpu()))
    for a,b in zip(*lanes):assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    assert bool((lanes[0][0][x<1]<0).all()) and float(lanes[0][0][x==1])==0
    with localcontext() as c:
        c.prec=90;u=Decimal(2)**-53
        for v,y,g,w in zip(x.tolist(),lanes[0][0].tolist(),lanes[0][1].tolist(),cot.tolist()):
            exact=Decimal.from_float(v).ln()
            assert abs(Decimal.from_float(y)-exact)<=128*u*(1+abs(exact))
            expected=Decimal.from_float(w)/Decimal.from_float(v)
            assert abs(Decimal.from_float(g)-expected)<=2*u
    v=torch.tensor([.13,.37,.6378118574468598,.91],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(log_unit,(v,),atol=1e-8,rtol=1e-6)
    assert torch.autograd.gradgradcheck(log_unit,(v,),atol=1e-8,rtol=1e-6)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_fresnel_fixed_fifth_power_values_and_vjp(dtype):
    from gffx.torch import shade
    c=torch.linspace(0,1,4097,dtype=dtype)
    c=torch.cat((c,torch.tensor([0.9202568782722006],dtype=dtype)))
    f=torch.tensor([.04,.19,.37],dtype=dtype);outputs=[]
    cot=torch.tensor([.25,-.5,.75],dtype=dtype).expand(len(c),3).clone()
    for device in ('cpu','cuda'):
        q=c.to(device).requires_grad_();ff=f.to(device).requires_grad_()
        y=shade.fresnel_schlick(q,ff);g=torch.autograd.grad(y,(q,ff),cot.to(device));outputs.append([z.detach().cpu() for z in (y,*g)])
    for a,b in zip(*outputs):assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    expected=f+(1-f)*(1-c[:,None]).pow(5)
    torch.testing.assert_close(outputs[0][0],expected,atol=1e-7 if dtype==torch.float32 else 1e-14,rtol=1e-6 if dtype==torch.float32 else 1e-13)
    expected_grad=(-5*(1-f)*(1-c[:,None]).pow(4)*cot).sum(-1)
    torch.testing.assert_close(outputs[0][1],expected_grad,atol=2e-6 if dtype==torch.float32 else 1e-13,rtol=1e-5 if dtype==torch.float32 else 1e-12)
