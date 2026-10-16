"""Fourth-order implicit transport, independently closed tests."""
import pytest
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q

@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('point',[Q(1,10),Q(1,4),Q(1,2)])
def test_fourth_circle_coefficient_independent_formula(sign,point):
    from tools.opaque_boundary_taylor import curve_jet
    from tools.opaque_shading_intervals import I
    x=I(point);y=(1-x*x).sqrt()*sign
    z=curve_jet(lambda x,y:x*x+y*y,x,y,order=4)
    with localcontext() as c:
        c.prec=90;q=D(point.numerator)/D(point.denominator);v=sign*(1-q*q).sqrt();expected=-(1+4*q*q)/(8*v**7)
        assert z[4,0].lo<=expected<=z[4,0].hi

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_order4_complete_material_closed_integral(monkeypatch,axis,sx,sy):
    from tools import opaque_boundary_material as module
    from tests.packaging.test_opaque_boundary_material import test_transport_keeps_independent_log_squared_rectangle
    original=module.BoundaryMaterial
    monkeypatch.setattr(module,'BoundaryMaterial',lambda *a,**k:original(*a,**k,boundary_order=4))
    test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,4)

def test_order4_is_explicit_and_invalid_degrees_refuse():
    from tools.opaque_boundary_material import BoundaryMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    assert BoundaryMaterial(fixture()).boundary_order==3
    assert BoundaryMaterial(fixture(),boundary_order=4).boundary_order==4
    for n in [0,2,5,True,'4']:
        with pytest.raises(ValueError):BoundaryMaterial(fixture(),boundary_order=n)

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_order3_identity_retains_complete_material(monkeypatch,axis,sx,sy):
    from tools import opaque_boundary_material as m
    from tests.packaging.test_opaque_boundary_material import test_transport_keeps_independent_log_squared_rectangle
    original=m.BoundaryMaterial
    def selected(*a,**k):k['order']=3;return original(*a,**k)
    monkeypatch.setattr(m,'BoundaryMaterial',selected)
    test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy,4)
