"""Outward Gauss12 with exact rational root isolation on [0,1].

Legendre recurrence defines P12 exactly. Six disjoint positive sign-change
brackets and their reflections account for all twelve simple roots. Weights
follow the Gauss-Legendre identity, as in opaque_gauss8; no sampled roots.
Only order-ten Peano bounds are used by the directional checker.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I

def _pair(x):
    a,b=1,x
    for n in range(2,13):a,b=b,((2*n-1)*x*b-(n-1)*a)/n
    return a,b

def rule(*,checkpoint=lambda:None):
    roots=[]
    for aa,bb in [(1,2),(3,4),(5,6),(7,8),(9,95),(98,100)]:
        denominator=100 if bb>10 else 10
        lo,hi=Q(aa,denominator),Q(bb,denominator)
        # The fifth bracket is 0.90..0.95.
        if (aa,bb)==(9,95):lo=Q(9,10)
        a,b=_pair(lo)[1],_pair(hi)[1]
        if a*b>=0:raise ValueError('missing strict Gauss12 root bracket')
        for _ in range(200):
            checkpoint();mid=(lo+hi)/2;v=_pair(mid)[1]
            if v==0:lo=hi=mid;break
            if a*v<0:hi=mid
            else:lo=mid;a=v
        roots.append(I(I(lo).lo,I(hi).hi))
    nodes=[];weights=[]
    for r in [-v for v in reversed(roots)]+roots:
        checkpoint();previous,current=_pair(r)
        derivative=12*(previous-r*current)/(1-r*r)
        w=1/((1-r*r)*derivative**2);x=(1+r)/2
        if w.lo<=0 or x.lo<=0 or x.hi>=1:raise ValueError('positive interior quadrature required')
        nodes.append(x);weights.append(w)
    return nodes,weights
