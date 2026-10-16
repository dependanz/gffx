"""Independent integral controls for signed ordinary remainders on fixed halves."""
import math
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('physical',[False,True])
@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('degree',[3,4,6,9])
def test_signed_partition_contains_exact_truncated_polynomial(axis,physical,sign,degree):
    from tools.opaque_smooth_peano import partitioned_ordinary
    from tools.opaque_peano import gauss5
    inner=I(2,4) if physical else I(0,1);outer=I(-3,0) if physical else I(0,1)
    box=[None,None];box[axis]=inner;box[1-axis]=outer
    hi=I(inner.hi)-I(inner.lo);ho=I(outer.hi)-I(outer.lo);k=I('.37');calls=[]
    def third(sub,active):
        calls.append(1);x=(sub[axis]-I(inner.lo))/hi;y=(sub[1-axis]-I(outer.lo))/ho;z=x-k
        if z.hi<0:return I(0)
        v=(I(0,1) if z.lo<=0 else I(1)) if degree==3 else I(max(I(0).lo,z.lo),max(I(0).hi,z.hi))**(degree-3)
        return v*(1+y)*(sign*math.factorial(degree)//math.factorial(degree-3))/hi**3
    got=partitioned_ordinary(third,tuple(box),axis);signed=got['signed'];assert len(calls)==4
    nodes,weights=gauss5();quad=I(0)
    for x,w in zip(nodes,weights):
        z=x-k;quad=quad+I(max(I(0).lo,z.lo),max(I(0).hi,z.hi))**degree*w*I(Q(3,2))*hi*ho*sign
    exact=I(Q(3,2))*I(Q(63,100))**(degree+1)/(degree+1)*hi*ho*sign
    known=quad-exact
    assert signed.lo<=known.lo<=known.hi<=signed.hi
    assert signed.lo>=-got['effective'].hi and signed.hi<=got['effective'].hi

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sign',[-1,1])
def test_piecewise_constant_third_has_nonzero_exact_signed_half_mass(axis,sign):
    from tools.opaque_smooth_peano import partitioned_ordinary
    from tools.opaque_peano import gauss5
    def third(box,active):return I(0) if box[axis].hi<=I('.5').hi else I(sign*6)
    got=partitioned_ordinary(third,(I(0,1),I(0,1)),axis)
    nodes,weights=gauss5();known=sum((I(max(I(0).lo,(x-I('.5')).lo),max(I(0).hi,(x-I('.5')).hi))**3*w*sign for x,w in zip(nodes,weights)),I(0))-I(Q(sign,64))
    assert got['signed'].lo<=known.lo<=known.hi<=got['signed'].hi
    assert got['signed'].width<I('1e-50').lo and not (got['signed'].lo<=0<=got['signed'].hi)

@pytest.mark.parametrize('value',[-7,0,13])
def test_constant_third_preserves_zero_and_four_calls(value):
    from tools.opaque_smooth_peano import partitioned_ordinary
    calls=[]
    def third(*args):calls.append(1);return I(value)
    got=partitioned_ordinary(third,(I(2,4),I(-3,0)),1)
    assert got['signed'].zero and got['effective'].zero and len(calls)==4

def test_integrated_signed_bound_keeps_sign_mass_and_jumps():
    from tools.opaque_transverse_peano import _error
    row=dict(third=I(100),jumps=[I(0)]*3,ordinary_integral=I(0,'.001'),ordinary_integral_signed=I('.0002','.0003'))
    a=_error(row,I(0,1),I(2),True)[0];b=_error(row,I(0,1),I(7),True)[0]
    assert a.lo==I('.0002').lo and a.hi==I('.0003').hi and (a.lo,a.hi)==(b.lo,b.hi)
    row['jumps']=[I('.1'),I('.2'),I('.3')];wide,parts=_error(row,I(0,1),I(2),True)
    assert wide.lo<a.lo and wide.hi>a.hi and parts['global_jumps'].hi>0
    row['ordinary_integral_signed']=I(1)
    with pytest.raises(ValueError,match='disjoint'):_error(row,I(0,1),I(2),True)


def test_material_producer_supplies_signed_integrated_bound():
    from tests.packaging.test_opaque_boundary_diagnostic import fixture
    c=fixture();c.smooth_partition=True
    row=c.integrated_bounds((I(0,1),I(0,1)),1)[1]
    assert 'ordinary_integral_signed' in row
    assert row['ordinary_integral_signed'].lo>=-row['ordinary_integral'].hi
    assert row['ordinary_integral_signed'].hi<=row['ordinary_integral'].hi
