"""Geometric value bounds can exclude false knots without hiding a true turn."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('axis',[0,1])
def test_centered_values_reduce_dependency_and_keep_exact_polynomial(axis):
    from tools.opaque_centered_gradient import CenteredInputs
    def f(x,y):
        a,b=(x,y) if axis==0 else (y,x)
        return [(a-b)*(a-b)+a-b+b-a]
    box=[I('.999','1.001'),I(1)];box=box if axis==0 else box[::-1]
    args=[Jet(v,order=4) for v in box];args[axis]=Jet.variable(box[axis],order=4)
    raw=f(*args)[0];q=CenteredInputs(f,refine_values=True)(*args)[0]
    assert q[0,0].width<raw[0,0].width/100
    for v in [Q(999,1000),Q(1),Q(1001,1000)]:assert Q(q[0,0].lo)<=(v-1)**2<=Q(q[0,0].hi)
    assert q[1,0].lo<0<q[1,0].hi
    assert all((q[k].lo,q[k].hi)==(raw[k].lo,raw[k].hi) for k in raw.c if k not in ((0,0),(1,0)))

@pytest.mark.parametrize('flag',[None,1,'yes'])
def test_centered_values_refuse_ambiguous_opt_in(flag):
    from tools.opaque_centered_gradient import CenteredInputs
    with pytest.raises(ValueError):CenteredInputs(None,refine_values=flag)

def test_centered_values_reject_incompatible_point_value():
    from tools.opaque_centered_gradient import CenteredInputs
    calls=[0]
    def bad(x,y):
        calls[0]+=1
        return [Jet(order=x.order,coeff={(0,0):I(0) if calls[0]==1 else I(10),(1,0):I(-1,1)})]
    with pytest.raises(ValueError,match='disjoint geometric value'):CenteredInputs(bad,refine_values=True)(Jet.variable(I(-1,1),order=2),Jet(0,order=2))

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_value_centered_boundary_keeps_independent_closed_integral(monkeypatch,axis,sx,sy):
    from tools.opaque_centered_gradient import ValueCenteredBoundary
    from tools import opaque_boundary_material as boundary
    from tests.packaging import test_opaque_boundary_material as f
    from tools.opaque_decimal_convolution import decimal_convolution
    monkeypatch.setattr(boundary,'BoundaryMaterial',ValueCenteredBoundary)
    with decimal_convolution():f.test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,2)
