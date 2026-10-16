"""Independent bound, Decimal and device controls for the private unit-range sqrt."""
from decimal import Decimal,localcontext
from fractions import Fraction
import pytest,torch

def test_unit_sqrt_roundoff_bound_before_implementation():
    u=Fraction(1,2**53);e=Fraction(1,2)
    for _ in range(6):
        assert 0<e<1
        e=e*e/(2*(1-e))+16*u
    assert e<32*u

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_unit_sqrt_independent_values_and_cpu_cuda_bytes(dtype):
    from gffx.torch._reference_channels import sqrt_unit
    x=torch.linspace(1,3,4097,dtype=dtype)
    x=torch.cat((x,torch.nextafter(x,x.new_tensor(0.))[1:],torch.nextafter(x,x.new_tensor(4.))[:-1]))
    outputs=[];grads=[]
    for device in ('cpu','cuda'):
        v=x.to(device).requires_grad_();result=sqrt_unit(v);cot=torch.arange(len(v),device=device,dtype=dtype).remainder(13)*.125-.5
        grad,=torch.autograd.grad(result,v,cot);outputs.append(result.detach().cpu());grads.append(grad.cpu())
    for a,b in (outputs,grads):assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    bound=2**-23 if dtype==torch.float32 else 32*2**-53
    with localcontext() as ctx:
        ctx.prec=90
        for a,b in zip(x.tolist(),outputs[0].tolist()):assert abs(Decimal.from_float(b)-Decimal.from_float(a).sqrt())<=Decimal.from_float(bound)
    v=torch.tensor([1.,1.125,1.73,2.3,3.],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(sqrt_unit,(v,),eps=1e-6,atol=1e-10,rtol=1e-9)
    assert torch.autograd.gradgradcheck(sqrt_unit,(v,),eps=1e-6,atol=1e-10,rtol=1e-9)
