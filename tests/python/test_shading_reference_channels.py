"""Independent mathematical controls for explicit three-channel reductions."""
from decimal import Decimal,localcontext
import pytest
import torch


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_unit_gradient_independent_scaled_vectors(dtype):
    from gffx.torch.light import _unit
    scales=(1e-12,1.,1e12) if dtype==torch.float32 else (1e-100,1.,1e100)
    values=torch.tensor([[.1*s,-.3*s,.7*s] for s in scales],dtype=dtype,requires_grad=True)
    cot=torch.tensor([[.17,-.29,.41]]*3,dtype=dtype)
    out=_unit(values,'independent');g,=torch.autograd.grad(out,values,grad_outputs=cot)
    with localcontext() as ctx:
        ctx.prec=90
        for v,w,y,grad in zip(values.detach().tolist(),cot.tolist(),out.detach().tolist(),g.tolist()):
            x=[Decimal.from_float(q) for q in v];c=[Decimal.from_float(q) for q in w]
            length=sum(q*q for q in x).sqrt();unit=[q/length for q in x];dot=sum(a*b for a,b in zip(unit,c))
            expected=[(a-b*dot)/length for a,b in zip(c,unit)]
            eps=Decimal.from_float(torch.finfo(dtype).eps)
            assert all(abs(Decimal.from_float(a)-b)<=32*eps for a,b in zip(y,unit))
            assert all(abs(Decimal.from_float(a)-b)<=64*eps/length for a,b in zip(grad,expected))


@pytest.mark.parametrize('operation',['multiply','divide'])
def test_channel_broadcast_first_derivatives_gradcheck(operation):
    from gffx.torch import _reference_channels as channels
    x=torch.tensor([[.13,-.27,.51],[.33,.17,-.43]],dtype=torch.float64,requires_grad=True)
    w=torch.tensor([[.73],[1.13]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(getattr(channels,operation),(x,w),eps=1e-6,atol=1e-6,rtol=1e-5)
    assert torch.autograd.gradgradcheck(getattr(channels,operation),(x,w),eps=1e-6,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('shape',[(),(3,),(1,3,1)])
def test_material_field_expand_preserves_shared_derivatives(shape):
    from gffx.torch._reference_channels import expand
    x=torch.full(shape,.37,dtype=torch.float64,requires_grad=True)
    fn=lambda value:expand(value,(2,3,3))
    assert torch.autograd.gradcheck(fn,(x,),eps=1e-6,atol=1e-6,rtol=1e-5)
    assert torch.autograd.gradgradcheck(fn,(x,),eps=1e-6,atol=1e-6,rtol=1e-5)
