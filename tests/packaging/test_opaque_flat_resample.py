"""Research-only flat-fan quadrature refinement and first-motion controls."""
import math
from fractions import Fraction as Q
import pytest
import torch

@pytest.mark.parametrize('orders',[(5,6),(5,7),(6,5),(7,5),(5,8),(8,5),(8,8)])
def test_flat_refinement_preserves_closed_triangle_moments_and_motion(orders):
    from tools.opaque_flat_resample import refine
    from tools.opaque_radial_resample import N,W
    n=torch.tensor(N,dtype=torch.float64);w=torch.tensor(W,dtype=torch.float64);a,b=torch.meshgrid(n,n,indexing='ij');theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True);x=a+theta;y=(1-a)*b
    q=torch.stack((x,y,1-x-y,(1+theta)*(1-a)*w[:,None]*w[None,:]),-1).reshape(-1,4);r=refine(q,*orders)
    assert len(r)==orders[0]*orders[1] and bool((r[:,3]>0).all())
    for i,j in [(0,0),(1,0),(0,1),(4,0),(0,4),(4,4)]:
        value=(r[:,0]**i*r[:,1]**j*r[:,3]).sum();expected=Q(math.factorial(i)*math.factorial(j),math.factorial(i+j+2));slope=expected+(i*Q(math.factorial(i-1)*math.factorial(j),math.factorial(i+j+1)) if i else 0)
        derivative=torch.autograd.grad(value,theta,retain_graph=True)[0];assert abs(value.item()-float(expected))<2e-14 and abs(derivative.item()-float(slope))<3e-14

@pytest.mark.parametrize('bad',['shape','capacity','nonfinite','nonaffine','order'])
def test_flat_refinement_refuses_unqualified_rows_or_capacity(bad):
    from tools.opaque_flat_resample import refine
    from tools.opaque_radial_resample import N,W
    n=torch.tensor(N,dtype=torch.float64);w=torch.tensor(W,dtype=torch.float64);a,b=torch.meshgrid(n,n,indexing='ij');q=torch.stack((a,(1-a)*b,(1-a)*(1-b),(1-a)*w[:,None]*w[None,:]),-1).reshape(-1,4)
    if bad=='shape':q=q[:-1]
    if bad=='capacity':q=q.repeat(13,1)
    if bad=='nonfinite':q[0,0]=float('nan')
    if bad=='nonaffine':q[2,0]+=.01
    with pytest.raises(ValueError):refine(q,9 if bad=='order' else 8,8)


@pytest.mark.parametrize('orders',[(6,7),(7,6),(6,6)])
def test_flat_reallocation_from_existing_five_by_eight_keeps_moments(orders):
    import numpy as np
    from tools.opaque_flat_resample import refine
    from tools.opaque_radial_resample import N,W
    n=torch.tensor(N,dtype=torch.float64);w=torch.tensor(W,dtype=torch.float64);nn,ww=np.polynomial.legendre.leggauss(8);m=torch.tensor((nn+1)/2,dtype=torch.float64);mw=torch.tensor(ww/2,dtype=torch.float64)
    a,b=torch.meshgrid(n,m,indexing='ij');theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True);x=a+theta;y=(1-a)*b
    q=torch.stack((x,y,1-x-y,(1+theta)*(1-a)*w[:,None]*mw[None,:]),-1).reshape(-1,4)
    r=refine(q,*orders,source=(5,8));assert len(r)==orders[0]*orders[1]
    for i,j in ((0,0),(1,0),(4,4),(8,0),(0,8)):
        value=(r[:,0]**i*r[:,1]**j*r[:,3]).sum();slope=torch.autograd.grad(value,theta,retain_graph=True)[0]
        exact=Q(math.factorial(i)*math.factorial(j),math.factorial(i+j+2));derivative=exact+(i*Q(math.factorial(i-1)*math.factorial(j),math.factorial(i+j+1)) if i else 0)
        assert abs(value.item()-float(exact))<2e-14 and abs(slope.item()-float(derivative))<3e-14
