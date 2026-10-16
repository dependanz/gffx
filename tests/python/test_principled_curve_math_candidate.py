"""Independent controls for a pre-production Principled arithmetic experiment."""
from decimal import Decimal,localcontext
import math
import pytest
import torch
from tools import principled_curve_math_candidate as candidate


def test_principled_exp_candidate_independent_value_and_vjp(record_property):
    values=[-745.,-744.4400719213812,-708.4,-37.,-1.,-.5,-.001,0.]
    values.extend(-745*i/511 for i in range(512))
    for n in (0,-1,-10,-100,-500,-1000):
        x=(n-.5)*math.log(2)
        values.extend((math.nextafter(x,-math.inf),x,math.nextafter(x,math.inf)))
    x=torch.tensor(values,dtype=torch.float64,requires_grad=True)
    y=candidate.exp_negative(x)
    g,=torch.autograd.grad(y,x,grad_outputs=torch.ones_like(y))
    max_scaled=0.
    with localcontext() as ctx:
        ctx.prec=100
        for value,actual,gradient in zip(values,y.detach().tolist(),g.tolist()):
            reference=Decimal.from_float(value).exp()
            bound=Decimal(128)*Decimal(2)**-53*reference+Decimal(2)**-1074
            for result in (actual,gradient):
                error=abs(Decimal.from_float(result)-reference)
                assert error<=bound,(value,result,str(reference),str(error/bound))
                max_scaled=max(max_scaled,float(error/bound))
    record_property('cases',len(values));record_property('max_bound_ratio',max_scaled)
    assert y[-1].isfinite()


def test_principled_exp_candidate_cuda_exact(record_property):
    values=torch.linspace(-745,0,1025,dtype=torch.float64)
    cot=torch.linspace(-.9,1.1,len(values),dtype=torch.float64)
    def evaluate(device):
        x=values.to(device).clone().requires_grad_();y=candidate.exp_negative(x)
        g,=torch.autograd.grad(y,x,grad_outputs=cot.to(device))
        return tuple(v.detach().cpu().contiguous().view(torch.uint8) for v in (y,g))
    rows=[evaluate(d) for d in ('cpu','cuda','cuda')]
    for index,(a,b,c) in enumerate(zip(*rows)):
        record_property(f'item{index}_different_bytes',int((a!=b).sum()))
        assert torch.equal(a,b) and torch.equal(b,c)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_principled_cubic_candidate_independent_gradient(device,record_property):
    controls=torch.tensor([-2.,-1.75,-1.25,-1.],dtype=torch.float64)[:,None,None,None].expand(4,2,3,2).clone()
    controls=controls-torch.arange(12,dtype=torch.float64).reshape(1,2,3,2)/32
    u=torch.tensor([.17,.83],dtype=torch.float64).reshape(2,1,1)
    cot=torch.linspace(-.7,.9,12,dtype=torch.float64).reshape(2,3,2)
    results=[]
    for lane in ('cpu',device,device):
        q=u.to(lane).clone().requires_grad_();value=candidate.roughness_interpolate(controls.to(lane),q)
        g,=torch.autograd.grad(value,q,grad_outputs=cot.to(lane))
        results.append((value.detach().cpu(),g.detach().cpu()))
    for a,b,c in zip(*results):
        assert torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))
        assert torch.equal(b.reshape(-1).view(torch.uint8),c.reshape(-1).view(torch.uint8))
    with localcontext() as ctx:
        ctx.prec=90
        for batch in range(2):
            t=Decimal.from_float(float(u[batch,0,0]));v=1-t;expected=Decimal(0)
            for j in range(3):
                for k in range(2):
                    b=[Decimal.from_float(float(controls[i,batch,j,k])) for i in range(4)]
                    y=v**3*b[0]+3*v*v*t*b[1]+3*v*t*t*b[2]+t**3*b[3]
                    derivative=3*(v*v*(b[1]-b[0])+2*v*t*(b[2]-b[1])+t*t*(b[3]-b[2]))
                    expected+=y.exp()*derivative*Decimal.from_float(float(cot[batch,j,k]))
            error=abs(Decimal.from_float(float(results[1][1][batch,0,0]))-expected)
            assert error<Decimal('1e-12')
            record_property(f'batch{batch}_absolute_vjp_error',str(error))
