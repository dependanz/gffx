"""Exact polynomial controls for direct Bernstein Peano kernel construction."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import itertools,math
import pytest
from tools.opaque_shading_intervals import I

@pytest.mark.parametrize('order',[1,2,3])
@pytest.mark.parametrize('domain',[(Q(0),Q(1)),(Q(1,8),Q(3,8)),(Q(1,2),Q(1,2)),(Q(999,1000),Q(1)),(Q(0),Q(1,10**20))])
@pytest.mark.parametrize('remaining',[0,1,3])
def test_direct_bernstein_contains_exact_rational_power_conversion(order,domain,remaining):
    from tools.opaque_peano import _kernel_segment_bernstein
    lo,hi=domain;nodes=[Q(1,4),Q(1,2),Q(3,4)][:remaining];weights=[Q(2,7),Q(3,7),Q(2,7)][:remaining]
    # Independent ordinary-power expansion, followed by power-to-Bernstein.
    coeff=[]
    for j in range(order+1):
        p=sum((w*Q(math.comb(order-1,j),math.factorial(order-1))*(x-lo)**(order-1-j)*(lo-hi)**j for x,w in zip(nodes,weights)),Q(0)) if j<order else Q(0)
        p-=Q(math.comb(order,j),math.factorial(order))*(1-lo)**(order-j)*(lo-hi)**j
        coeff.append(p)
    expected=[sum((coeff[j]*Q(math.comb(k,j),math.comb(order,j)) for j in range(k+1)),Q(0)) for k in range(order+1)]
    actual=_kernel_segment_bernstein(I(lo),I(hi),list(map(I,nodes)),list(map(I,weights)),order)
    assert len(actual)==order+1
    for q,b in zip(expected,actual):assert Q(b.lo)<=q<=Q(b.hi)

@pytest.mark.parametrize('order',[1,2,3])
def test_uncertain_nodes_weights_and_endpoints_retain_all_corner_polynomials(order):
    from tools.opaque_peano import _kernel_segment_bernstein
    lo=I('.1','.100001');hi=I('.2','.200001');nodes=[I('.3','.300001')];weights=[I('.4','.400001')]
    got=_kernel_segment_bernstein(lo,hi,nodes,weights,order)
    for a,b,x,w in itertools.product(*[(Q(v.lo),Q(v.hi)) for v in [lo,hi,*nodes,*weights]]):
        coeff=[(w*Q(math.comb(order-1,j),math.factorial(order-1))*(x-a)**(order-1-j)*(a-b)**j if j<order else Q(0))-Q(math.comb(order,j),math.factorial(order))*(1-a)**(order-j)*(a-b)**j for j in range(order+1)]
        for k,v in enumerate(got):
            exact=sum((coeff[j]*Q(math.comb(k,j),math.comb(order,j)) for j in range(k+1)),Q(0))
            assert Q(v.lo)<=exact<=Q(v.hi)

def test_local_kernel_no_longer_constructs_general_derivative_jets(monkeypatch):
    from tools import opaque_peano as m
    m.kernel_bounds();m._kernel_local_ranges_cached.cache_clear()
    def forbidden(*a,**k):raise AssertionError('general Jet in degree-three local kernel')
    monkeypatch.setattr(m,'Jet',forbidden)
    for interval in [I(0,1),I('.13','.1301'),*m.gauss5()[0]]:
        got=m.kernel_local_ranges(interval);assert len(got)==3

@pytest.mark.parametrize('order',[1,2,3])
def test_dense_independent_kernel_points_and_jump_sides(order):
    from tools.opaque_peano import kernel_local_range,gauss5
    with localcontext() as c:
        c.prec=110;a=(D(5)+2*(D(10)/7).sqrt()).sqrt()/6;b=(D(5)-2*(D(10)/7).sqrt()).sqrt()/6
        nodes=[D('.5')-a,D('.5')-b,D('.5'),D('.5')+b,D('.5')+a];u=(D(322)-13*D(70).sqrt())/1800;v=(D(322)+13*D(70).sqrt())/1800;weights=[u,v,D(64)/225,v,u]
        points=[D(k)/100 for k in range(101)]+[n+sign*D('1e-50') for n in nodes for sign in [-1,1]]
        for point in points:
            box=I(point);actual=kernel_local_range(box,order)
            value=sum((w*(x-point)**(order-1)/math.factorial(order-1) for x,w in zip(nodes,weights) if x>point),D(0))-(1-point)**order/math.factorial(order)
            assert actual.lo<=value<=actual.hi
    for node,weight in zip(*gauss5()):
        value=kernel_local_range(node,order)
        if order==1:assert value.width>=weight.lo
