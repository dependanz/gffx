"""Certified absolute Bernstein kernel mass, not a sampled estimate."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I

@pytest.mark.parametrize('row,width,expected',[
    ([-1,1],Q(1),Q(1,2)),([0,1],Q(3),Q(3,2)),([1,2,3],Q(2),Q(4)),
    ([-3,-2,-1],Q(2),Q(4)),([0,0,0],Q(4),Q(0)),([1,-1,1],Q(1),Q(1,3)),
    ([0,0,0,1],Q(2),Q(1,2)),([-1,-1,1,1],Q(1),Q(5,8))])
def test_absolute_bernstein_mass_contains_exact_integrals(row,width,expected):
    from tools.opaque_peano import _bernstein_absolute_mass
    result=_bernstein_absolute_mass(list(map(I,row)),I(width))
    assert result.lo>=0 and Q(result.hi)>=expected
    assert result.hi<=(I(width)*I(max(abs(x) for x in row))).hi

@pytest.mark.parametrize('order',[1,2,3])
@pytest.mark.parametrize('domain',[(0,0),(1,1),(0,1),('.12','.14'),('.48','.52'),('.1','.9')])
def test_mass_is_bounded_by_existing_global_and_rectangle_bounds(order,domain):
    from tools.opaque_peano import kernel_local_mass,kernel_local_range,kernel_bounds
    p=I(*domain);v=kernel_local_mass(p,order)
    assert 0<=v.lo<=v.hi<=kernel_bounds()[1][order-1].hi
    assert v.hi<=(I(kernel_local_range(p,order).bound)*I(p.width)).hi

def test_linear_kernel_mass_exact_in_first_segment_and_fresh_cache():
    from tools.opaque_peano import kernel_local_mass,_kernel_local_mass_cached
    # Before the first Gauss node, K1(t)=t; integral0..1/100=t^2/2.
    result=kernel_local_mass(I(0,'.01'),1)
    assert Q(result.hi)>=Q(1,20000) and Q(result.hi)<=Q(1,20000)+Q(1,10**50)
    result.hi=I(999).hi
    assert kernel_local_mass(I(0,'.01'),1).hi<I('.001').hi
    for i in range(140):kernel_local_mass(I(I(Q(i,200)).lo,I(Q(i+1,200)).hi),1)
    assert _kernel_local_mass_cached.cache_info().maxsize==128 and _kernel_local_mass_cached.cache_info().currsize<=128

def test_weighted_kernel_uses_integrated_mass_with_all_uncertain_support():
    from tools.opaque_transverse_peano import weighted_kernel_parts
    v=weighted_kernel_parts(I(0,2),I(0),I('.01'),1)
    assert v['absolute_mass'].hi<I('0.000051').hi
    assert v['value'].lo<=0 and v['value'].hi>=I('.0001').hi


def test_mass_does_not_rebuild_the_same_segment_polynomials_for_rectangle(monkeypatch):
    from tools import opaque_peano as m
    m.kernel_bounds();m._kernel_local_mass_cached.cache_clear()
    def forbidden(*args):raise AssertionError('duplicate range polynomial construction')
    monkeypatch.setattr(m,'kernel_local_range',forbidden)
    assert m.kernel_local_mass(I('.13','.15'),3).hi>=0
