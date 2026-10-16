"""Integration by parts with discontinuous tables and complete boundary terms."""
from fractions import Fraction as Q
import pytest

@pytest.mark.parametrize('order',[1,2,3,4,5,6])
@pytest.mark.parametrize('direction',[-1,1])
def test_all_boundary_terms_preserve_independent_step_polynomial_integral(order,direction):
    from tools.opaque_primitive_parts import terms
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_angular_primitives import step_curve
    f=Primitives(step_curve(),max_order=6);a,b=I('.25'),I(1)
    def at(x):
        x=Jet.variable(x,order=order+1);q=x if direction==1 else x.lift('1.25')-x
        weight=q**(order-1)
        return terms(q,weight,lambda z,n:f.integral(z,I('.25'),n)[0],order=order)
    left,lr=at(a);right,rr=at(b);_,rem=at(I(a.lo,b.hi));got=right-left+rem*(b-a)
    # Integral f(q)*q^(order-1) on[.25,1], independent of orientation.
    exact=I((2*(Q(1,2)**order-Q(1,4)**order)+4*(1-Q(1,2)**order))/order)
    assert got.lo<=exact.lo<=exact.hi<=got.hi
    assert got.width<I('1e-45').lo and rem.bound<I('1e-45').lo

def test_nonlinear_coordinate_keeps_jacobian_and_complete_remainder():
    from tools.opaque_primitive_parts import terms
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_angular_primitives import step_curve
    f=Primitives(step_curve(),max_order=6);a,b=I('.1'),I('.5')
    def at(v):
        x=Jet.variable(v,order=4);q=x*x+I('.25');return terms(q,2*x*q*q,lambda z,n:f.integral(z,I('.25'),n)[0],order=3)
    left,_=at(a);right,_=at(b);_,remainder=at(I(a.lo,b.hi));got=right-left+remainder*(b-a)
    exact=2*(I('.5')**3-I('.26')**3)/3
    assert got.lo<=exact.lo<=exact.hi<=got.hi

def test_unknown_coordinate_direction_and_parameter_tangent_are_refused():
    from tools.opaque_primitive_parts import terms
    from tools.opaque_shading_intervals import I,Jet
    x=Jet.variable(I(-1,1),order=4)
    with pytest.raises(ValueError):terms(x*x,x.lift(1),lambda z,n:I(1),order=3)
    x=Jet.variable(I('.5'),order=4);x.c[0,1]=I(1)
    with pytest.raises(ValueError):terms(x,x.lift(1),lambda z,n:I(1),order=3)
    with pytest.raises(ValueError):terms(Jet(1),Jet(1),lambda z,n:I(1),order=3)

def test_parts_propagates_cancellation():
    from tools.opaque_primitive_parts import terms
    from tools.opaque_shading_intervals import I,Jet,Unresolved
    def stop():raise Unresolved('stop parts')
    with pytest.raises(Unresolved):terms(Jet.variable(I('.5'),order=4),Jet(1,order=3),lambda z,n:I(1),order=3,checkpoint=stop)


@pytest.mark.parametrize('order',[1,2,3,4,5,6])
def test_missing_highest_coordinate_derivative_is_rejected(order):
    from tools.opaque_primitive_parts import terms
    from tools.opaque_shading_intervals import I,Jet
    with pytest.raises(ValueError,match='coordinate order'):
        terms(Jet.variable(I('.5'),order=order),Jet(1,order=order),lambda z,n:I(0),order=order)

@pytest.mark.parametrize('order',[1,2,3,4,5,6])
def test_nonzero_remainder_is_required_for_exact_rational_integral(order):
    from tools.opaque_primitive_parts import terms
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_angular_primitives import step_curve
    f=Primitives(step_curve(),max_order=6);a,b=I('.6'),I('.8')
    def at(v):
        x=Jet.variable(v,order=order+1);return terms(x,1/(1+x)**2,lambda z,n:f.integral(z,I('.5'),n)[0],order=order)
    left,_=at(a);right,_=at(b);_,rem=at(I(a.lo,b.hi));boundary=right-left;got=boundary+rem*(b-a);exact=4*(1/(1+a)-1/(1+b))
    assert got.lo<=exact.lo<=exact.hi<=got.hi
    assert boundary.hi<exact.lo or boundary.lo>exact.hi
