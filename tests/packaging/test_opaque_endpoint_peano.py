"""Exact kernel controls for uncertain endpoints,including Gauss-node crossings."""
import math
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I
from tools.opaque_peano import gauss5


def natural(a,b,order):
    nodes,weights=gauss5()
    def p(t):return (1-t)**(order+1)/math.factorial(order+1)-sum((w*I(max(I(0).lo,(x-t).lo),max(I(0).hi,(x-t).hi))**order/math.factorial(order) for x,w in zip(nodes,weights)),I(0))
    return p(b)-p(a)


@pytest.mark.parametrize('order',[1,2,3])
@pytest.mark.parametrize('bounds',[('.29','.32','.99','1'),('.1','.21','.6','.65'),('.48','.52','.7','.8'),('0','.01','.99','1')])
def test_endpoint_mass_contains_point_primitives_and_never_widens(order,bounds,record_property):
    from tools.opaque_transverse_peano import kernel_integral
    a=I(*bounds[:2]);b=I(*bounds[2:]);got=kernel_integral(a,b,order);old=natural(a,b,order)
    assert got.lo>=old.lo and got.hi<=old.hi
    for j in range(5):
        for k in range(5):
            aa=I(a.lo)+(I(a.hi)-I(a.lo))*Q(j,4);bb=I(b.lo)+(I(b.hi)-I(b.lo))*Q(k,4)
            exact=natural(aa,bb,order)
            assert got.lo<=exact.lo<=exact.hi<=got.hi
    assert got.width*2<old.width
    record_property('endpoint_improvement',str(old.width/got.width))


def test_uncertain_endpoints_in_first_segment_enclose_exact_polynomial():
    from tools.opaque_transverse_peano import kernel_integral
    # Before the first Gauss node,K3=t^3/6 exactly by moments0,1,2.
    a=I('.01','.02');b=I('.03','.04');got=kernel_integral(a,b,3)
    exact=(b**4-a**4)/24
    assert got.lo<=exact.lo and got.hi>=exact.hi
    assert got.width<I('1e-7').lo


def test_signed_witness_intersects_two_valid_bounds_and_refuses_disjoint():
    from tools.opaque_transverse_peano import _error
    w=dict(position=I(0,1),jumps=[I(0),I(0),I(1)],signed_jumps=[I(0),I(0),I(1)],max_crossings=1,required=True,transverse_integral=I(-100,100))
    row=dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[w])
    old=_error(row,I(0,1),I(1),False)[0];new=_error(row,I(0,1),I(1),True)[0]
    assert new.lo==old.lo and new.hi==old.hi
    w['transverse_integral']=I(100)
    with pytest.raises(ValueError,match='disjoint'):_error(row,I(0,1),I(1),True)
