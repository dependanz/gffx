"""Requested-order kernel controls against exact formulas and full bounds."""
from decimal import Decimal as D,localcontext
import math
import pytest
from tools.opaque_shading_intervals import I

@pytest.mark.parametrize('order',[1,2,3])
@pytest.mark.parametrize('domain',[(0,0),(1,1),(0,1),('.13','.13000001'),('.499999','.500001'),('.02','.1')])
def test_requested_order_matches_full_outward_polynomial(order,domain):
    from tools.opaque_peano import kernel_local_range,kernel_local_ranges
    p=I(*domain);a=kernel_local_range(p,order);b=kernel_local_ranges(p)[order-1]
    assert (a.lo,a.hi)==(b.lo,b.hi)

@pytest.mark.parametrize('order',[1,2,3])
@pytest.mark.parametrize('point',['0','.02','.13','.3','.7','.93','1'])
def test_requested_kernel_contains_independent_100digit_peano_value(order,point):
    from tools.opaque_peano import kernel_local_range
    actual=kernel_local_range(I(point),order)
    with localcontext() as c:
        c.prec=100;t=D(point);a=(D(5)+2*(D(10)/7).sqrt()).sqrt()/6;b=(D(5)-2*(D(10)/7).sqrt()).sqrt()/6
        nodes=[D('.5')-a,D('.5')-b,D('.5'),D('.5')+b,D('.5')+a];u=(D(322)-13*D(70).sqrt())/1800;v=(D(322)+13*D(70).sqrt())/1800;weights=[u,v,D(64)/225,v,u]
        value=sum((w*(x-t)**(order-1)/math.factorial(order-1) for x,w in zip(nodes,weights) if x>t),D(0))-(1-t)**order/math.factorial(order)
    assert actual.lo<=value<=actual.hi

def test_requested_order_keeps_both_sides_of_uncertain_gauss_nodes():
    from tools.opaque_peano import kernel_local_range,kernel_local_ranges,gauss5
    for node,weight in zip(*gauss5()):
        assert kernel_local_range(node,1).width>=weight.lo
        for order in (1,2,3):
            got=kernel_local_range(node,order);full=kernel_local_ranges(node)[order-1]
            assert (got.lo,got.hi)==(full.lo,full.hi)

def test_requested_only_evaluates_one_order_and_uses_shared_bounded_cache(monkeypatch):
    from tools import opaque_peano as m
    m._kernel_local_ranges_cached.cache_clear();original=m._kernel_local_ranges_uncached;calls=[]
    def observed(p,order=None):calls.append(order);return original(p) if order is None else original(p,order)
    monkeypatch.setattr(m,'_kernel_local_ranges_uncached',observed)
    got=m.kernel_local_range(I('.13'),3);assert calls==[3];got.lo=D(-999)
    fresh=m.kernel_local_range(I('.13'),3);assert fresh.lo!=D(-999) and calls==[3]
    for k in range(132):
        p=I(D(k)/D(200));m.kernel_local_range(p,k%3+1)
        if k%8==0:m.kernel_local_ranges(p)
    info=m._kernel_local_ranges_cached.cache_info();assert info.maxsize==128 and info.currsize<=128
    m.kernel_local_range(I('.130001'),3);assert calls[-1]==3

@pytest.mark.parametrize('order',[0,4,True,1.5,None])
def test_requested_order_refuses_invalid_order(order):
    from tools.opaque_peano import kernel_local_range
    with pytest.raises(ValueError):kernel_local_range(I('.13'),order)

@pytest.mark.parametrize('bounds',[(-1,0),(0,2)])
def test_requested_order_refuses_invalid_domain(bounds):
    from tools.opaque_peano import kernel_local_range
    with pytest.raises(ValueError):kernel_local_range(I(*bounds),3)

def test_weighted_path_requests_only_its_order(monkeypatch):
    from tools import opaque_transverse_peano as m
    original=m.kernel_local_range;calls=[]
    def observed(p,order):calls.append(order);return original(p,order)
    monkeypatch.setattr(m,'kernel_local_range',observed)
    for order in (1,2,3):
        calls.clear();m.weighted_kernel_parts(I('1','1.1'),I('.12','.120001'),I('.13','.130001'),order)
        assert calls and set(calls)=={order}


def test_requested_orders_share_one_position_entry_and_fill_only_missing_orders(monkeypatch):
    from tools import opaque_peano as m
    m._kernel_local_ranges_cached.cache_clear();original=m._kernel_local_ranges_uncached;calls=[]
    def observed(p,order=None):calls.append(order);return original(p) if order is None else original(p,order)
    monkeypatch.setattr(m,'_kernel_local_ranges_uncached',observed)
    p=I('.13','.1300001');m.kernel_local_range(p,3)
    assert calls==[3]
    full=m.kernel_local_ranges(p)
    assert calls==[3,1,2] and m._kernel_local_ranges_cached.cache_info().currsize==1
    for order in (1,2,3):
        got=m.kernel_local_range(p,order);assert (got.lo,got.hi)==(full[order-1].lo,full[order-1].hi)
    assert calls==[3,1,2]

def test_failed_order_fill_keeps_other_completed_orders(monkeypatch):
    from tools import opaque_peano as m
    m._kernel_local_ranges_cached.cache_clear();p=I('.14');third=m.kernel_local_range(p,3);original=m._kernel_local_ranges_uncached
    def stopped(p,order=None):raise RuntimeError('stop')
    monkeypatch.setattr(m,'_kernel_local_ranges_uncached',stopped)
    with pytest.raises(RuntimeError,match='stop'):m.kernel_local_range(p,1)
    got=m.kernel_local_range(p,3);assert (got.lo,got.hi)==(third.lo,third.hi)
    monkeypatch.setattr(m,'_kernel_local_ranges_uncached',original)
    assert m.kernel_local_range(p,1).lo==m.kernel_local_ranges(p)[0].lo
