"""Independent factorial moments for a source-pinned symmetric diagnostic rule."""
from fractions import Fraction as Q
import math,torch,pytest


def test_symmetric_triangle_has_positive_interior_rule_and_all_degree14_moments():
    from tools.opaque_symmetric_triangle import table
    rows=table();assert len(rows)==42 and all(min(r)>0 for r in rows)
    for r in rows:assert abs(sum(r[:3])-1)<3e-16
    for i in range(15):
        for j in range(15-i):
            actual=sum(Q(u)**i*Q(v)**j*Q(w) for _,u,v,w in rows);expected=Q(math.factorial(i)*math.factorial(j),math.factorial(i+j+2))
            assert abs(actual-expected)<=Q('2e-15')*expected,(i,j)


def test_symmetric_resampling_keeps_moving_triangle_moments():
    import numpy as np
    from tools.opaque_symmetric_triangle import refine
    from tools.opaque_radial_resample import N,W
    n=torch.tensor(N,dtype=torch.float64);w=torch.tensor(W,dtype=torch.float64);xx,ww=np.polynomial.legendre.leggauss(8);m=torch.tensor((xx+1)/2);mw=torch.tensor(ww/2);a,b=torch.meshgrid(n,m,indexing='ij');theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    x=a+theta;y=(1-a)*b;q=torch.stack((x,y,1-x-y,(1+theta)*(1-a)*w[:,None]*mw[None,:]),-1).reshape(-1,4);r=refine(q);assert len(r)==42
    for i,j in ((0,0),(1,0),(4,4),(12,0),(0,12)):
        value=(r[:,0]**i*r[:,1]**j*r[:,3]).sum();slope=torch.autograd.grad(value,theta,retain_graph=True)[0];expected=Q(math.factorial(i)*math.factorial(j),math.factorial(i+j+2));derivative=expected+(i*Q(math.factorial(i-1)*math.factorial(j),math.factorial(i+j+1)) if i else 0)
        assert abs(value.item()-float(expected))<2e-14 and abs(slope.item()-float(derivative))<3e-14
    bad=q.detach().clone();bad[3,0]+=.001
    with pytest.raises(ValueError):refine(bad)
    with pytest.raises(ValueError):refine(q.repeat(20,1))
