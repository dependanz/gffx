"""Exact endpoint reuse test, independent legacy formula and work count."""
import math,random
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I

def legacy(lo,hi,c):
    n=len(c)-1;h=hi-lo;p=[sum((c[k]*math.comb(k,j)*lo**(k-j) for k in range(j,n+1)),I(0))*h**j for j in range(n+1)]
    return [sum((p[j]*Q(math.comb(i,j),math.comb(n,j)) for j in range(i+1)),I(0)) for i in range(n+1)]

@pytest.mark.parametrize('order',range(1,11))
@pytest.mark.parametrize('seed',range(5))
def test_reused_powers_exact_endpoints(order,seed):
    from tools.opaque_high_peano import _bernstein
    r=random.Random(seed);lo=I(Q(r.randrange(-10,10),11));hi=lo+I(Q(r.randrange(1,10),13));c=[I(Q(r.randrange(-10,10),17)) for _ in range(order+1)]
    a,b=legacy(lo,hi,c),_bernstein(lo,hi,c)
    assert [(v.lo,v.hi) for v in a]==[(v.lo,v.hi) for v in b]

def test_polynomial_power_work_is_linear(monkeypatch):
    from tools.opaque_high_peano import _bernstein
    old=I.__pow__;calls=[]
    def power(a,n):calls.append(n);return old(a,n)
    monkeypatch.setattr(I,'__pow__',power)
    _bernstein(I('.1'),I('.2'),[I(1)]*11)
    assert len(calls)<=22
