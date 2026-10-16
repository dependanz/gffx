"""Independent real-exponent Classic arithmetic and device regression."""
import math,pytest,torch
from decimal import Decimal,localcontext
from test_deferred_cuda import exact
from gffx.torch import shade

def test_power_underflow_floor_certificate():
    with localcontext() as ctx:
        ctx.prec=100
        assert Decimal(-746).exp()<Decimal(2)**-1075
        assert Decimal(-745).exp()>Decimal(2)**-1075

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_blinn_real_exponents_independent_decimal_values_vjps(dtype):
    tiny=torch.nextafter(torch.tensor(0.,dtype=dtype),torch.tensor(1.,dtype=dtype)).item()
    xs=[tiny,torch.finfo(dtype).tiny,1e-6,.003,.07,.3,.5909855619936794,.8,.97,1-torch.finfo(dtype).eps,1.]
    ns=[0.,.125,.5,1.,2.,2.3,16.,64.,1024.]
    c=torch.tensor([x for x in xs for n in ns],dtype=dtype);n=torch.tensor(ns*len(xs),dtype=dtype)
    cot=torch.arange(len(c),dtype=dtype).remainder(7)*.125-.375;rows=[]
    for device in ('cpu','cuda'):
        q=c.to(device).requires_grad_();e=n.to(device).requires_grad_();value=shade.blinn_phong(q,e)
        grads=torch.autograd.grad(value,(q,e),cot.to(device));rows.append((value,*grads))
    for i,(a,b) in enumerate(zip(*rows)):exact(a,b,'real power field '+str(i))
    expected=[[],[],[]]
    with localcontext() as ctx:
        ctx.prec=90;ctx.Emin=-999999999;ctx.Emax=999999999
        pi=Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825')
        for x,e,g in zip(c.tolist(),n.tolist(),cot.tolist()):
            x=Decimal.from_float(x);e=Decimal.from_float(e);g=Decimal.from_float(g);power=x**e;scale=(e+8)/(8*pi)
            values=(scale*power,g*scale*e*(x**(e-1)),g*power*(1/(8*pi)+scale*x.ln()))
            for arr,v in zip(expected,values):arr.append(float(v))
    for actual,values in zip(rows[0],expected):
        ref=torch.tensor(values,dtype=dtype)
        assert torch.isfinite(ref).all()
        torch.testing.assert_close(actual,ref,atol=1e-6 if dtype==torch.float32 else 1e-12,rtol=1e-5 if dtype==torch.float32 else 1e-11)
        # Only assert positivity/sign where a nonzero reference is representable.
        nonzero=ref!=0
        assert bool((actual[nonzero]!=0).all())

def test_blinn_real_exponent_first_and_mixed_second_derivatives():
    q=torch.tensor([.31,.73],dtype=torch.float64,requires_grad=True);n=torch.tensor([.5,2.3],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(shade.blinn_phong,(q,n),eps=1e-6,atol=1e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(shade.blinn_phong,(q,n),eps=1e-6,atol=1e-6,rtol=1e-4)
