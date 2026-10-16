"""Experimental positive Gauss8, exact rational root isolation, Decimal60 bounds.

P8 has degree8; four disjoint positive sign-changing rational brackets and
negative partners account for all8 real simple roots. Gauss orthogonality gives
exactness through15; weights on[0,1] are1/((1-r*r)*P8prime(r)**2).
See https://dlmf.nist.gov/3.5.v (Gauss-Legendre formula).
Only the experimental high-order helpers are rebound, explicitly and serially.
No additional integrand smoothness and no order16 Jet are assumed.
"""
from contextlib import contextmanager
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I
_active=False

def _p(x):return (6435*x**8-12012*x**6+6930*x**4-1260*x**2+35)/128

def rule(*,checkpoint=lambda:None):
    roots=[]
    for lo,hi in [(Q(1,10),Q(3,10)),(Q(4,10),Q(6,10)),(Q(7,10),Q(9,10)),(Q(9,10),Q(1))]:
        a,b=_p(lo),_p(hi)
        if a*b>=0:raise ValueError('missing strict root bracket')
        for _ in range(200):
            checkpoint();mid=(lo+hi)/2;v=_p(mid)
            if v==0:lo=hi=mid;break
            if a*v<0:hi=mid;b=v
            else:lo=mid;a=v
        roots.append(I(I(lo).lo,I(hi).hi))
    roots=[-r for r in reversed(roots)]+roots;nodes=[];weights=[]
    for r in roots:
        checkpoint();derivative=(51480*r**7-72072*r**5+27720*r**3-2520*r)/128
        w=1/((1-r*r)*derivative**2);x=(1+r)/2
        if w.lo<=0 or x.lo<=0 or x.hi>=1:raise ValueError('positive interior quadrature required')
        nodes.append(x);weights.append(w)
    return nodes,weights

class Work:
    """Conservative sample-equivalent charge: ceil(64/25)=3 per attempt.

    This is in addition to the unchanged wall/memory/depth limits. Failed-domain
    attempts consume the same charge; callers must take before evaluating.
    """
    charge=3
    def __init__(self,limit=256):
        if type(limit)!=int or not 1<=limit<=256:raise ValueError('work limit1..256')
        self.limit=limit;self.units=0;self.regions=0
    def take(self):
        if self.units+self.charge>self.limit:raise ValueError('work budget')
        self.units+=self.charge;self.regions+=1

@contextmanager
def context(*,checkpoint=lambda:None):
    global _active
    if _active:raise ValueError('nested quadrature context')
    from tools import opaque_high_peano as h,opaque_high_transport as t,opaque_boundary_taylor as b
    modules=[h,t,b];saved=[m.gauss5 for m in modules]
    nodes,weights=rule(checkpoint=checkpoint)
    endpoints=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights))
    def selected():return [I(a,z) for a,z,_,_ in endpoints],[I(a,z) for _,_,a,z in endpoints]
    _active=True
    try:
        for m in modules:m.gauss5=selected
        yield selected
    finally:
        for m,f in zip(modules,saved):m.gauss5=f
        _active=False
