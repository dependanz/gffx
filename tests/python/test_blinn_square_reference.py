"""Exact exponent-two highlight regression; exponent remains differentiable."""
import math,pytest,torch
from decimal import Decimal,localcontext
from gffx.torch import shade
from test_deferred_cuda import exact

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_blinn_square_values_and_both_vjps_exact(dtype):
    c=torch.cat((torch.linspace(.003,1,257,dtype=dtype),torch.tensor([.5909855619936794],dtype=dtype)))
    cot=torch.arange(len(c),dtype=dtype).remainder(7)*.125-.375;rows=[]
    for device in ('cpu','cuda'):
        q=c.to(device).requires_grad_();n=torch.full_like(q,2.,requires_grad=True)
        y=shade.blinn_phong(q,n);grad=torch.autograd.grad(y,(q,n),cot.to(device));rows.append((y,*grad))
    for i,(a,b) in enumerate(zip(*rows)):exact(a,b,'Blinn square field '+str(i))
    with localcontext() as ctx:
        ctx.prec=90;pi=Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825')
        v=[];dc=[];dn=[]
        for x,g in zip(c.tolist(),cot.tolist()):
            q=Decimal.from_float(x);up=Decimal.from_float(g);scale=Decimal(10)/(8*pi)
            v.append(float(scale*q*q));dc.append(float(up*scale*2*q));dn.append(float(up*q*q*(1/(8*pi)+scale*q.ln())))
    for actual,values in zip(rows[0],(v,dc,dn)):
        torch.testing.assert_close(actual,torch.tensor(values,dtype=dtype),atol=1e-6 if dtype==torch.float32 else 1e-12,rtol=1e-5 if dtype==torch.float32 else 1e-11)

def test_blinn_square_exponent_gradient_and_mixed_second_derivatives():
    q=torch.tensor([.31,.73],dtype=torch.float64,requires_grad=True);n=torch.tensor([2.,2.],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(shade.blinn_phong,(q,n),eps=1e-6,atol=1e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(shade.blinn_phong,(q,n),eps=1e-6,atol=1e-6,rtol=1e-4)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_blinn_existing_noninteger_contract_operands_exact(dtype):
    rows=[]
    for device in ('cpu','cuda'):
        q=torch.tensor([.3,.8],dtype=dtype,device=device,requires_grad=True)
        n=torch.tensor([2.3,16.],dtype=dtype,device=device,requires_grad=True)
        value=shade.blinn_phong(q,n)
        gradients=torch.autograd.grad(value,(q,n),q.new_tensor([.25,-.75]))
        rows.append((value,*gradients))
    for i,(a,b) in enumerate(zip(*rows)):exact(a,b,'existing noninteger Blinn field '+str(i))
