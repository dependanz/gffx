"""Independent grazing-angle Bezier coordinate derivative regression."""
from decimal import Decimal,localcontext
import math
import pytest,torch


def test_angular_bezier_grazing_coordinate_independent_first_vjp(record_property):
    from gffx.torch.material import PrincipledBudget
    from gffx.torch._loss_curve import _bezier
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    rows=[]
    for r in (.213,.427,.827,1.):
        _,_,_,coeff=budget._curve(torch.tensor([r],dtype=torch.float64))
        b=coeff[0,0].clone();scale=float(budget._data['_clog'][1]-budget._data['_clog'][0])*1e-10
        for u0 in (0.,math.nextafter(0.,1.),.125,.5,.875,1.):
            u=torch.tensor([[u0]],dtype=torch.float64,requires_grad=True);control=b[:,None,:]
            cot=torch.tensor([[.125,-.75]],dtype=torch.float64)/scale
            y=_bezier(control,u);g,=torch.autograd.grad(y,u,cot)
            with localcontext() as ctx:
                ctx.prec=150;D=Decimal.from_float;t=D(u0);v=1-t;exact=Decimal(0)
                for channel in range(2):
                    z=[D(float(x)) for x in b[:,channel]]
                    d=3*(v*v*(z[1]-z[0])+2*v*t*(z[2]-z[1])+t*t*(z[3]-z[2]))
                    exact+=D(float(cot[0,channel]))*d
                error=abs(D(float(g))-exact)
                rows.append((r,u0,str(error)))
                assert error<=Decimal('1e-10')+Decimal('1e-9')*abs(exact),(r,u0,float(g),str(exact))
    record_property('grazing_partial_rows',str(rows))


def test_angular_bezier_coefficients_and_cpu_cuda_values_vjps():
    from gffx.torch._loss_curve import _bezier
    base=torch.tensor([[.81,.53],[.78,.42],[.62,.31],[.51,.17]],dtype=torch.float64)
    rows=[]
    for device in ('cpu','cuda'):
        b=base.to(device).requires_grad_();u=torch.tensor([.31],dtype=torch.float64,device=device,requires_grad=True)
        y=_bezier(b,u);grads=torch.autograd.grad(y,(b,u),torch.tensor([.125,-.75],dtype=torch.float64,device=device));rows.append([x.detach().cpu() for x in (y,*grads)])
    for a,b in zip(*rows):assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    b=base.clone().requires_grad_();u=torch.tensor([.31],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(_bezier,(b,u),atol=1e-8,rtol=1e-6)
