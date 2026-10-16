"""Closed integrals and refusal controls for finite-table signed transport."""
from fractions import Fraction as Q
import pytest

@pytest.mark.parametrize('order',range(1,7))
@pytest.mark.parametrize('a,b',[(Q(0),Q(1)),(Q(1,7),Q(3,4)),(Q(1,2),Q(1,2))])
def test_signed_kernel_primitive_contains_independent_truncated_moments(order,a,b):
    from decimal import Decimal as D,localcontext
    import math
    from tools.opaque_high_transport import kernel_integral
    from tools.opaque_shading_intervals import I
    got=kernel_integral(I(a),I(b),order)
    with localcontext() as c:
        c.prec=90
        a,b=[D(q.numerator)/D(q.denominator) for q in (a,b)]
        x=(D(5)+2*(D(10)/7).sqrt()).sqrt()/6;y=(D(5)-2*(D(10)/7).sqrt()).sqrt()/6
        ns=[D('.5')-x,D('.5')-y,D('.5'),D('.5')+y,D('.5')+x];u=(322-13*D(70).sqrt())/1800;v=(322+13*D(70).sqrt())/1800
        expected=sum(w*(max(D(0),n-a)**order-max(D(0),n-b)**order)/math.factorial(order) for n,w in zip(ns,[u,v,D(64)/225,v,u]))-((1-a)**(order+1)-(1-b)**(order+1))/math.factorial(order+1)
        assert got.lo-D('1e-80')<=expected<=got.hi+D('1e-80')

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
def test_transport_keeps_independent_log_squared_rectangle(axis,sx,sy):
    from tools.opaque_high_transport import TransportMaterial
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_shading_intervals import I
    from decimal import Decimal as D,localcontext
    got=TransportMaterial(fixture(sx,sy),order=6)((I(0,1),I(0,1)),integrated_axis=axis)[1]
    with localcontext() as c:
        c.prec=90;k=D('.5');a=D('.3');b=D('.01');d=D('.4')
        def f(q):
            if q<=k:return D(0)
            z=(q/k).ln();return q*q/2*(z*z-3*z+D('3.5'))-2*k*q+k*k/4
        expected=(f(a+b+d)-f(a+b)-f(a+d)+f(a))/(b*d)*(1+D(2).ln()**2)
        assert got.lo<=expected<=got.hi

def test_transport_rejects_disjoint_integrated_evidence():
    from tools.opaque_high_peano import high_error
    from tools.opaque_shading_intervals import I
    row=dict(order=3,derivative=I(0),jumps=[dict(position=I('.5'),values=[I(0)]*3,required=True,transverse_integral=I(1))])
    with pytest.raises(ValueError,match='disjoint'):high_error(row,I(0,1),I(1),integrated=True)

@pytest.mark.parametrize('case',['steps','simultaneous','refusal'])
def test_transport_preserves_discontinuous_and_refusal_controls(case,monkeypatch):
    from tools.opaque_high_transport import TransportMaterial
    from tools import opaque_aggregate_peano as module
    from tests.packaging import test_opaque_aggregate_peano as controls
    from tools import opaque_high_material as hm
    from tests.packaging.test_opaque_high_material import high_steps
    monkeypatch.setattr(module,'AggregateMaterial',TransportMaterial)
    if case=='steps':
        monkeypatch.setattr(hm,'HighMaterial',TransportMaterial);high_steps()
    elif case=='simultaneous':controls.test_simultaneous_view_and_light_steps_preserve_product_jumps()
    else:controls.test_unknown_monotonicity_and_cancellation_refuse()
