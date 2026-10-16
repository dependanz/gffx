"""Independent derivative and jump-flux controls, including discontinuities."""
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet
from tests.packaging.test_opaque_loss_extension import Curve

def test_piecewise_residual_first_derivative_contains_independent_polynomials():
    from tools.opaque_knot_transport import residual_first
    from tools.opaque_loss_extension import LossExtension
    e=LossExtension(Curve());q=Jet(I(.3,.9),tangent=-.7);got=residual_first(e,q,1)
    with localcontext() as c:
        c.prec=90
        for f in (.31,.4,.51,.6,.71,.8,.89):
            v=D.from_float(f);j=min(sum(n.lo<=v for n in Curve.cn)-1,3)
            def point(i,ch):
                h=Curve.x[i+1].lo-Curve.x[i].lo;u=(v.ln()-Curve.x[i].lo)/h;b=[x.lo for x in Curve.controls[i][ch]]
                value=(1-u)**3*b[0]+3*(1-u)**2*u*b[1]+3*(1-u)*u*u*b[2]+u**3*b[3]
                slope=3*((1-u)**2*(b[1]-b[0])+2*u*(1-u)*(b[2]-b[1])+u*u*(b[3]-b[2]))/(h*v)
                return value,slope
            for ch in range(2):
                a,da=point(j,ch);b,db=point(1,ch);t=(da-db)*D.from_float(-.7)
                assert got[ch][0,0].lo<=a-b<=got[ch][0,0].hi
                assert got[ch][0,1].lo<=t<=got[ch][0,1].hi
    assert any(not x[0,1].zero for x in got)

def test_transversal_flux_bounds_a_moving_discontinuous_step():
    from tools.opaque_knot_transport import transversal_flux
    # q=.2+.6*x+theta; a unit jump at q=.5 traverses the unit right triangle.
    # Ordinary pointwise derivative=0, but exact integral derivative=5/6.
    field=lambda x,y:x*Q(3,5)+x.lift(Q(1,5),tangent=1)
    result=transversal_flux(field,(I(0,1),I(0,1)),[(I('.5'),I(1))])
    exact=I(Q(5,6));assert result['bound'].hi>=exact.hi>0
    assert result['axis']==0 and result['crossings']==1
    assert result['bound'].hi<=I(Q(5,3)).hi

def test_transversal_flux_refuses_stationary_crossing_and_ignores_absent_knots():
    from tools.opaque_knot_transport import transversal_flux
    field=lambda x,y:(x-Q(1,2))**2+x.lift(Q(1,2),tangent=1)
    with pytest.raises(ValueError,match='transversal'):transversal_flux(field,(I(0,1),I(0,1)),[(I('.5'),I(1))])
    assert transversal_flux(field,(I(0,1),I(0,1)),[(I(2),I(1))])['bound'].zero

def test_residual_derivative_refuses_horizon_and_spatial_jet_misuse():
    from tools.opaque_knot_transport import residual_first
    from tools.opaque_loss_extension import LossExtension
    e=LossExtension(Curve())
    with pytest.raises(ValueError):residual_first(e,Jet(I('.1','.3'),tangent=1),1)
    with pytest.raises(ValueError):residual_first(e,Jet.variable(.5,order=2),1)
