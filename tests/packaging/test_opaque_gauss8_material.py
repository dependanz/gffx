"""Independent material integrals and fail-closed controls with Gauss8."""
import pytest
from tests.packaging.test_opaque_boundary_material import *
@pytest.fixture(autouse=True)
def selected_rule():
    from tools.opaque_gauss8 import context
    with context():yield

@pytest.mark.parametrize('order',[3,4,5])
@pytest.mark.parametrize('axis',[0,1])
def test_lower_orders_retain_independent_material_integral(order,axis):
    from tools.opaque_boundary_material import BoundaryMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    from decimal import Decimal as D,localcontext
    got=BoundaryMaterial(fixture(),order=order)((I(0,1),I(0,1)),integrated_axis=axis)[1]
    with localcontext() as ctx:
        ctx.prec=90;k=D('.5');a=D('.3');b=D('.01');d=D('.4')
        def f(q):
            if q<=k:return D(0)
            z=(q/k).ln();return q*q/2*(z*z-3*z+D('3.5'))-2*k*q+k*k/4
        expected=(f(a+b+d)-f(a+b)-f(a+d)+f(a))/(b*d)*(1+D(2).ln()**2)
        assert got.lo<=expected<=got.hi
